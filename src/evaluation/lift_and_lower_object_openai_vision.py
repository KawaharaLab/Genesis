import argparse
import base64
import csv
import io
import json
import os
import random
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import genesis as gs
import genesis.utils.geom as gu
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

try:
    from openai import OpenAI
except ImportError as exc:
    raise RuntimeError("openai package is required. Install with: uv pip install openai") from exc


ROOT = Path(__file__).resolve().parents[2]
PANDA_XML_PATH = ROOT / "genesis" / "assets" / "xml" / "franka_emika_panda" / "panda.xml"
YCB_TARGET_OBJ_PATH = ROOT / "data" / "objects" / "ycb" / "010_potted_meat_can" / "model.obj"

DT = 0.01
SIM_HZ = int(round(1.0 / DT))
HOLD_SECONDS = 1.0
HOLD_STEPS = int(HOLD_SECONDS / DT)
RENDER_EVERY = 1
SIM_STEP_COUNT = 0
TRACE_RECORDER = {
    "enabled": False,
    "images_dir": None,
    "image_stride": 80,
}
OPENAI_JPEG_QUALITY = 90

TARGET_TILE_SIZE = 0.20
TARGET_TILE_THICKNESS = 0.004
TARGET_TILE_COLOR = (0.10, 0.35, 0.90, 1.0)


def infer_finger_length_from_mjcf(xml_path: Path) -> float:
    try:
        root = ET.parse(xml_path).getroot()
        z_max = 0.053
        for default_node in root.findall(".//default"):
            cls = default_node.attrib.get("class", "")
            if not cls.startswith("fingertip_pad_collision_"):
                continue
            geom = default_node.find("geom")
            if geom is None:
                continue
            pos = [float(v) for v in geom.attrib.get("pos", "0 0 0").split()]
            size = [float(v) for v in geom.attrib.get("size", "0 0 0").split()]
            if len(pos) == 3 and len(size) == 3:
                z_max = max(z_max, pos[2] + size[2])
        return float(z_max)
    except Exception:
        return 0.053


def _bounds_from_obj(obj_path: Path):
    min_corner = np.array([np.inf, np.inf, np.inf], dtype=float)
    max_corner = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
    vertex_found = False
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                x, y, z = map(float, parts[1:4])
                vertex = np.array([x, y, z], dtype=float)
                min_corner = np.minimum(min_corner, vertex)
                max_corner = np.maximum(max_corner, vertex)
                vertex_found = True
    if not vertex_found:
        raise ValueError(f"No vertex data found in OBJ file: {obj_path}")
    return min_corner, max_corner


def _vertices_from_obj(obj_path: Path) -> np.ndarray:
    vertices = []
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if not line.startswith("v "):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError(f"No vertex data found in OBJ file: {obj_path}")
    return np.asarray(vertices, dtype=float)


def _scale_to_vec(scale):
    if np.isscalar(scale):
        return np.array([float(scale), float(scale), float(scale)], dtype=float)
    arr = np.asarray(scale, dtype=float)
    if arr.size == 1:
        return np.repeat(arr.item(), 3)
    if arr.size != 3:
        raise ValueError(f"Scale must be scalar or length 3, got shape {arr.shape}")
    return arr


def set_grasp_for_obj(obj_path: Path):
    gripper_min_width, gripper_max_width = 0.002, 0.075
    min_corner, max_corner = _bounds_from_obj(obj_path)
    bbox = (max_corner - min_corner).tolist()
    scale = 1.0
    if gripper_min_width < bbox[0] < gripper_max_width:
        euler = (0, 0, 90)
    elif gripper_min_width < bbox[1] < gripper_max_width:
        euler = (0, 0, 0)
    else:
        scale = (0.080 / (bbox[0] + 0.01)) if bbox[0] < bbox[1] else (0.080 / (bbox[1] + 0.01))
        euler = (0, 0, 90) if bbox[0] < bbox[1] else (0, 0, 0)
    return scale, euler, min_corner


def ik_pose(franka, end_effector, pos, quat=(0, 1, 0, 0)):
    pos_vec = np.asarray(pos, dtype=float).reshape(3)
    quat_vec = np.asarray(quat, dtype=float).reshape(4)
    return franka.inverse_kinematics(link=end_effector, pos=pos_vec[None, :], quat=quat_vec[None, :])


def control_pose(franka, motors_dof, fingers_dof, qpos, gripper_opening):
    franka.control_dofs_position(qpos[0, :-2], motors_dof)
    franka.control_dofs_position(np.array([gripper_opening, gripper_opening]), fingers_dof)


def sim_step(scene, cam=None, record=False):
    global SIM_STEP_COUNT
    scene.step()
    trace_enabled = bool(TRACE_RECORDER["enabled"] and TRACE_RECORDER["images_dir"] is not None)
    trace_step = trace_enabled and (SIM_STEP_COUNT % max(1, int(TRACE_RECORDER["image_stride"])) == 0)
    render_step = SIM_STEP_COUNT % max(1, RENDER_EVERY) == 0
    should_render = bool((record and render_step) or trace_step)
    if should_render and cam is not None:
        try:
            rgb, _, _, _ = cam.render(rgb=True)
        except Exception as exc:
            if trace_enabled:
                print(f"[warn] camera render failed, disabling trial image capture: {exc}")
                TRACE_RECORDER["enabled"] = False
                TRACE_RECORDER["images_dir"] = None
            rgb = None
        if rgb is not None and trace_step:
            frame_path = TRACE_RECORDER["images_dir"] / f"frame_{SIM_STEP_COUNT:06d}.png"
            plt.imsave(frame_path, rgb)
    SIM_STEP_COUNT += 1


def save_dict_rows_to_csv(rows: list[dict], csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["empty"])
        return
    keys = sorted(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_force_plot_from_score_trace(rows: list[dict], out_path: Path):
    if not rows:
        return
    steps = [r["step"] for r in rows]
    pred = [1.0 if r.get("pred_touch", False) else 0.0 for r in rows]
    gt = [1.0 if r.get("touching_support_gt", False) else 0.0 for r in rows]
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(steps, pred, label="pred_touch(0/1)")
    ax.plot(steps, gt, label="gt_touch(0/1)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Touch Flag")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def scaled_steps(base_steps: int, motion_speed: float) -> int:
    return max(8, int(round(base_steps / max(motion_speed, 1e-3))))


def move_ee(scene, franka, end_effector, motors_dof, fingers_dof, pos, gripper_opening, steps=120, cam=None, record=False):
    current = franka.get_links_pos([8])[0].cpu().numpy().reshape(3)
    target = np.array(pos, dtype=float)
    for i in range(max(1, steps)):
        alpha = (i + 1) / max(1, steps)
        interp = (1.0 - alpha) * current + alpha * target
        qpos = ik_pose(franka, end_effector, interp)
        control_pose(franka, motors_dof, fingers_dof, qpos, gripper_opening=gripper_opening)
        sim_step(scene, cam=cam, record=record)


def set_gripper(scene, franka, motors_dof, fingers_dof, start_open, end_open, steps=60, cam=None, record=False):
    arm_hold = franka.get_dofs_position(motors_dof)[0].cpu().numpy()
    for i in range(max(1, steps)):
        alpha = (i + 1) / max(1, steps)
        opening = (1.0 - alpha) * start_open + alpha * end_open
        franka.control_dofs_position(arm_hold, motors_dof)
        franka.control_dofs_position(np.array([opening, opening]), fingers_dof)
        sim_step(scene, cam=cam, record=record)


def hold_pose(scene, franka, end_effector, motors_dof, fingers_dof, pos, gripper_opening, steps, cam=None, record=False):
    qpos = ik_pose(franka, end_effector, pos)
    for _ in range(max(1, steps)):
        control_pose(franka, motors_dof, fingers_dof, qpos, gripper_opening=gripper_opening)
        sim_step(scene, cam=cam, record=record)


def get_aabb_min_max(obj_entity):
    aabb = obj_entity.get_AABB().cpu().numpy().reshape(-1, 3)
    return aabb[0], aabb[1]


def compute_grasp_heights(franka, hand_idx, left_finger_idx, obj_entity, finger_len, hover_z):
    lower, upper = get_aabb_min_max(obj_entity)
    obj_top = float(upper[2])
    obj_h = float(max(upper[2] - lower[2], 1e-4))

    links_pos = franka.get_links_pos([hand_idx, left_finger_idx])[0].cpu().numpy()
    hand_z = float(links_pos[0, 2])
    root_z = float(links_pos[1, 2])
    hand_to_root_dz = root_z - hand_z
    sign = -1.0 if hand_to_root_dz < 0.0 else 1.0
    hand_to_tip_dz = hand_to_root_dz + sign * finger_len

    root_clearance = max(0.008, 0.15 * obj_h)
    tip_target_depth = min(0.010, 0.20 * obj_h)

    z_for_tip = (obj_top - tip_target_depth) - hand_to_tip_dz
    z_for_root = (obj_top + root_clearance) - hand_to_root_dz
    grasp_hand_z = max(z_for_tip, z_for_root)
    grasp_hand_z = min(grasp_hand_z, hover_z - 0.01)

    clamp_hand_z = max(grasp_hand_z - 0.004, z_for_root)
    safe_retract_z = max(grasp_hand_z + 0.03, obj_top + 0.12)
    lift_z = max(0.30, safe_retract_z + 0.10)
    lower_back_z = clamp_hand_z
    return grasp_hand_z, clamp_hand_z, safe_retract_z, lift_z, lower_back_z


def compute_grasp_xy_from_aabb(obj_entity, fallback_xy):
    lower, upper = get_aabb_min_max(obj_entity)
    center_xy = 0.5 * (lower[:2] + upper[:2])
    span_xy = np.maximum(upper[:2] - lower[:2], 1e-6)
    margin_xy = np.minimum(0.010, 0.20 * span_xy)
    low_safe = lower[:2] + margin_xy
    high_safe = upper[:2] - margin_xy
    x = float(np.clip(center_xy[0], low_safe[0], high_safe[0]))
    y = float(np.clip(center_xy[1], low_safe[1], high_safe[1]))
    if not np.isfinite(x) or not np.isfinite(y):
        return float(fallback_xy[0]), float(fallback_xy[1])
    return x, y


def align_object_yaw_to_gripper(
    scene,
    obj_entity,
    cam=None,
    record=False,
    vertices_local: np.ndarray | None = None,
    obj_scale=1.0,
    min_adjust_deg=3.0,
    min_anisotropy=0.08,
    settle_steps=40,
):
    quat_raw = obj_entity.get_quat()
    if hasattr(quat_raw, "detach"):
        quat = quat_raw.detach().cpu().numpy().reshape(-1, 4)[0]
    else:
        quat = np.asarray(quat_raw, dtype=float).reshape(-1, 4)[0]
    euler_deg = np.asarray(gu.quat_to_xyz(quat, rpy=True, degrees=True), dtype=float).reshape(3)

    if vertices_local is not None and len(vertices_local) >= 8:
        scale_vec = _scale_to_vec(obj_scale).reshape(1, 3)
        verts_scaled = np.asarray(vertices_local, dtype=float) * scale_vec
        rot = np.asarray(gu.quat_to_R(quat), dtype=float).reshape(3, 3)
        verts_world = verts_scaled @ rot.T
        xy = verts_world[:, :2]
        xy_centered = xy - np.mean(xy, axis=0, keepdims=True)
        cov = (xy_centered.T @ xy_centered) / max(1, xy_centered.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major_idx = int(np.argmax(eigvals))
        lam_max = float(max(eigvals[major_idx], 1e-12))
        lam_min = float(max(eigvals[1 - major_idx], 0.0))
        anisotropy = (lam_max - lam_min) / lam_max
        if anisotropy < float(min_anisotropy):
            return
        major = eigvecs[:, major_idx]
        yaw_deg = float(np.degrees(np.arctan2(major[1], major[0])))
    else:
        yaw_deg = float(euler_deg[2])

    aligned_yaw_deg = float(np.round(yaw_deg / 90.0) * 90.0)
    delta_deg = float(((aligned_yaw_deg - yaw_deg + 180.0) % 360.0) - 180.0)
    if abs(delta_deg) < float(min_adjust_deg):
        return

    euler_deg[2] = float(euler_deg[2] + delta_deg)
    quat_aligned = gu.xyz_to_quat(euler_deg, rpy=True, degrees=True)
    obj_entity.set_quat(quat_aligned)
    for _ in range(max(1, int(settle_steps))):
        sim_step(scene, cam=cam, record=record)


def sample_object_params(object_type: str, rng: random.Random) -> dict:
    rho = 200.0 * rng.uniform(1.0, 10.0)
    if object_type == "cube":
        size = rng.uniform(0.030, 0.075)
        return {"rho": float(rho), "size": float(size), "scale": None}
    if object_type == "bottle":
        scale = rng.uniform(0.075, 0.105)
        return {"rho": float(rho), "size": None, "scale": float(scale)}
    return {"rho": float(rho), "size": None, "scale": None}


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _encode_rgb_to_data_url(rgb: np.ndarray) -> str:
    rgb_u8 = np.asarray(np.clip(rgb, 0, 255), dtype=np.uint8)
    img = Image.fromarray(rgb_u8)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(max(30, min(95, OPENAI_JPEG_QUALITY))))
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _parse_touch_json(text: str) -> bool:
    try:
        payload = json.loads(text)
        val = payload.get("touching_ground", False)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() in ("true", "yes", "1")
        return bool(val)
    except Exception:
        low = text.strip().lower()
        return "true" in low or "yes" in low


def classify_touching_ground_with_openai(client, model_name: str, rgb: np.ndarray) -> tuple[bool, str]:
    image_url = _encode_rgb_to_data_url(rgb)
    prompt = (
        "You are judging a robot placement scene. "
        "Answer whether the grasped object is already touching the support surface (floor or blue tile). "
        "Return strict JSON only: {\"touching_ground\": true/false, \"confidence\": 0..1, \"reason\": \"short\"}."
    )
    resp = client.responses.create(
        model=model_name,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ],
        temperature=0,
        max_output_tokens=120,
    )
    text = getattr(resp, "output_text", "") or ""
    return _parse_touch_json(text), text


def _normalize_api_key(raw_key: str | None) -> str | None:
    if raw_key is None:
        return None
    key = raw_key.strip()
    if (key.startswith('"') and key.endswith('"')) or (key.startswith("'") and key.endswith("'")):
        key = key[1:-1].strip()
    return key or None


def safe_render_rgb(cam, retries: int = 3, backoff_sec: float = 0.03):
    for attempt in range(1, retries + 1):
        try:
            rgb, _, _, _ = cam.render(rgb=True)
            if rgb is not None:
                return np.asarray(rgb), True, ""
        except Exception as exc:
            if attempt >= retries:
                raise RuntimeError(
                    f"camera render failed after {retries} retries: {type(exc).__name__}: {exc}"
                ) from exc
            time.sleep(backoff_sec * attempt)
    raise RuntimeError("camera render failed with unknown error")


def object_touching_support(obj_entity, support_entity) -> bool:
    contacts = obj_entity.get_contacts(with_entity=support_entity)
    valid = contacts.get("valid_mask", None)
    if valid is not None:
        if hasattr(valid, "detach"):
            valid = valid.detach().cpu().numpy()
        return bool(np.asarray(valid).astype(bool).any())
    link_a = contacts.get("link_a", [])
    if hasattr(link_a, "detach"):
        link_a = link_a.detach().cpu().numpy()
    return np.asarray(link_a).size > 0


def pick_lift_and_visual_release(
    scene,
    franka,
    end_effector,
    motors_dof,
    fingers_dof,
    hand_idx,
    left_finger_idx,
    obj_entity,
    support_entity,
    finger_len,
    obj_xy,
    motion_speed,
    post_descend_wait_steps,
    vision_cam,
    render_cam,
    record,
    client,
    openai_model,
    align_vertices_local=None,
    align_scale=1.0,
    eval_interval_steps=10,
):
    x, y = obj_xy
    hover_z = 0.24
    grasp_z, clamp_z, safe_retract_z, lift_z, lower_back_z = compute_grasp_heights(
        franka=franka,
        hand_idx=hand_idx,
        left_finger_idx=left_finger_idx,
        obj_entity=obj_entity,
        finger_len=finger_len,
        hover_z=hover_z,
    )

    align_object_yaw_to_gripper(
        scene,
        obj_entity,
        cam=render_cam,
        record=record,
        vertices_local=align_vertices_local,
        obj_scale=align_scale,
    )
    x, y = compute_grasp_xy_from_aabb(obj_entity, fallback_xy=obj_xy)

    move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, hover_z), 0.04, scaled_steps(150, motion_speed), render_cam, record)
    move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, grasp_z), 0.04, scaled_steps(100, motion_speed), render_cam, record)
    move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, clamp_z), 0.04, scaled_steps(35, motion_speed), render_cam, record)
    set_gripper(scene, franka, motors_dof, fingers_dof, 0.04, 0.0, scaled_steps(60, motion_speed), render_cam, record)
    move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, safe_retract_z), 0.0, scaled_steps(50, motion_speed), render_cam, record)
    move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, lift_z), 0.0, scaled_steps(120, motion_speed), render_cam, record)
    hold_pose(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, lift_z), 0.0, HOLD_STEPS, render_cam, record)
    move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, safe_retract_z), 0.0, scaled_steps(80, motion_speed), render_cam, record)

    start_pos = franka.get_links_pos([8])[0].cpu().numpy().reshape(3)
    target_pos = np.array([x, y, lower_back_z], dtype=float)
    descend_steps = scaled_steps(140, motion_speed)

    support_touch_step = None
    release_step = None
    release_reason = "no_visual_touch_detected"
    eval_records = []
    render_failures = 0

    eval_interval_steps = max(1, int(eval_interval_steps))
    for i in range(descend_steps):
        alpha = (i + 1) / descend_steps
        interp = (1.0 - alpha) * start_pos + alpha * target_pos
        qpos = ik_pose(franka, end_effector, interp)
        control_pose(franka, motors_dof, fingers_dof, qpos, gripper_opening=0.0)
        sim_step(scene, cam=render_cam, record=record)

        touching_support = object_touching_support(obj_entity, support_entity)
        if touching_support and support_touch_step is None:
            support_touch_step = int(scene.t)

        if i % eval_interval_steps == 0:
            rgb, ok_render, render_note = safe_render_rgb(vision_cam)
            if not ok_render:
                render_failures += 1
            pred_touch, raw_text = classify_touching_ground_with_openai(client, openai_model, rgb)
            eval_records.append(
                {
                    "step": int(scene.t),
                    "pred_touch": bool(pred_touch),
                    "touching_support_gt": bool(touching_support),
                    "raw_response": raw_text,
                    "render_ok": bool(ok_render),
                    "render_note": render_note,
                }
            )
            if pred_touch:
                release_step = int(scene.t)
                release_reason = "openai_visual_touch_during_descent"
                break

    if release_step is None:
        wait_steps = max(1, int(post_descend_wait_steps))
        qpos_hold = ik_pose(franka, end_effector, (x, y, lower_back_z))
        for i in range(wait_steps):
            control_pose(franka, motors_dof, fingers_dof, qpos_hold, gripper_opening=0.0)
            sim_step(scene, cam=render_cam, record=record)

            touching_support = object_touching_support(obj_entity, support_entity)
            if touching_support and support_touch_step is None:
                support_touch_step = int(scene.t)

            if i % eval_interval_steps == 0:
                rgb, ok_render, render_note = safe_render_rgb(vision_cam)
                if not ok_render:
                    render_failures += 1
                pred_touch, raw_text = classify_touching_ground_with_openai(client, openai_model, rgb)
                eval_records.append(
                    {
                        "step": int(scene.t),
                        "pred_touch": bool(pred_touch),
                        "touching_support_gt": bool(touching_support),
                        "raw_response": raw_text,
                        "render_ok": bool(ok_render),
                        "render_note": render_note,
                    }
                )
                if pred_touch:
                    release_step = int(scene.t)
                    release_reason = "openai_visual_touch_after_descent"
                    break

    released = release_step is not None
    if released:
        set_gripper(scene, franka, motors_dof, fingers_dof, 0.0, 0.04, scaled_steps(40, motion_speed), render_cam, record)

    delay_steps = None
    delay_sec = None
    if (support_touch_step is not None) and (release_step is not None):
        delay_steps = max(0, release_step - support_touch_step)
        delay_sec = delay_steps * DT

    move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, hover_z), 0.04, scaled_steps(100, motion_speed), render_cam, record)

    return {
        "release_reason": release_reason,
        "release_step": release_step,
        "support_touch_step": support_touch_step,
        "released": bool(released),
        "support_touched_at_release": bool(released and support_touch_step is not None and release_step >= support_touch_step),
        "delay_steps_after_touch": delay_steps,
        "delay_sec_after_touch": delay_sec,
        "vision_render_failures": int(render_failures),
        "score_trace": eval_records,
        "score_trace_tail": eval_records[-12:],
    }


def _add_wrist_camera(scene, franka, res: int = 640):
    cam_wrist = scene.add_camera(
        model="thinlens",
        res=(res, res),
        pos=(0.0, 0.0, 0.0),
        lookat=(0.0, 0.0, 0.0),
        fov=80.0,
        GUI=False,
    )
    roll = np.deg2rad(-10)
    pitch = np.deg2rad(180)
    yaw = np.deg2rad(-90)
    r_roll = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    r_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    r_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    r = r_yaw @ r_pitch @ r_roll
    trans = np.array([0.14, 0.0, -0.08])
    wrist_mount_link = franka.get_link("hand")
    cam_wrist.attach(wrist_mount_link, gu.trans_R_to_T(trans, r))
    return cam_wrist


def run_single_trial(trial_idx, args, params, output_dir, camera_name: str, use_tile: bool, client):
    global SIM_STEP_COUNT
    SIM_STEP_COUNT = 0
    global TRACE_RECORDER

    trial_dir = output_dir / f"trial_{trial_idx + 1:02d}"
    images_dir = trial_dir / "images" / camera_name
    csv_dir = trial_dir / "csv"
    images_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    TRACE_RECORDER["enabled"] = bool(args.save_trial_images)
    TRACE_RECORDER["images_dir"] = images_dir if args.save_trial_images else None
    TRACE_RECORDER["image_stride"] = int(args.trial_image_stride)

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.cuda_device))
        gs.init(backend=gs.gpu, logging_level="warning")
    else:
        gs.init(backend=gs.cpu, logging_level="warning")

    scene = None
    render_cam = None
    should_record = False
    video_path = None
    try:
        scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.6, -1.1, 1.3),
                camera_lookat=(0.55, 0.0, 0.20),
                camera_fov=35,
                max_FPS=60,
            ),
            rigid_options=gs.options.RigidOptions(dt=DT),
            show_viewer=args.show_viewer,
        )
        floor_entity = scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        obj_xy = (0.62, 0.0)
        align_vertices_local = None
        align_scale = 1.0
        tile_entity = None
        tile_top_z = 0.0
        if use_tile:
            tile_entity = scene.add_entity(
                material=gs.materials.Rigid(rho=800, friction=1.0, coup_friction=1.0),
                morph=gs.morphs.Box(
                    size=(TARGET_TILE_SIZE, TARGET_TILE_SIZE, TARGET_TILE_THICKNESS),
                    pos=(obj_xy[0], obj_xy[1], TARGET_TILE_THICKNESS * 0.5),
                    fixed=True,
                ),
                surface=gs.surfaces.Default(color=TARGET_TILE_COLOR),
            )
            tile_top_z = TARGET_TILE_THICKNESS

        if args.object_type == "cube":
            cube_size = params["size"]
            obj_entity = scene.add_entity(
                material=gs.materials.Rigid(rho=params["rho"], friction=1.0),
                morph=gs.morphs.Box(size=(cube_size, cube_size, cube_size), pos=(obj_xy[0], obj_xy[1], tile_top_z + cube_size * 0.5)),
                surface=gs.surfaces.Default(color=(0.35, 0.75, 0.95, 1.0)),
            )
        elif args.object_type == "bottle":
            spawn_z = tile_top_z + 0.036 * (params["scale"] / 0.09)
            obj_entity = scene.add_entity(
                material=gs.materials.Rigid(rho=params["rho"]),
                morph=gs.morphs.URDF(
                    file="urdf/3763/mobility_vhacd.urdf",
                    scale=params["scale"],
                    pos=(obj_xy[0], obj_xy[1], spawn_z),
                    euler=(0, 90, 0),
                ),
            )
        else:
            object_scale, object_euler, min_corner = set_grasp_for_obj(YCB_TARGET_OBJ_PATH)
            align_vertices_local = _vertices_from_obj(YCB_TARGET_OBJ_PATH)
            align_scale = object_scale
            scale_vec = _scale_to_vec(object_scale)
            spawn_z = tile_top_z + 0.001 + max(0.0, -scale_vec[2] * min_corner[2])
            obj_entity = scene.add_entity(
                material=gs.materials.Rigid(rho=params["rho"], friction=1.0),
                morph=gs.morphs.Mesh(
                    file=str(YCB_TARGET_OBJ_PATH),
                    scale=object_scale,
                    pos=(obj_xy[0], obj_xy[1], float(spawn_z)),
                    euler=object_euler,
                ),
                surface=gs.surfaces.Default(color=(0.0, 1.0, 0.0, 1.0)),
            )

        franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
        cam_0 = scene.add_camera(
            model="thinlens",
            res=(args.vision_resolution, args.vision_resolution),
            pos=(1.7, 0.0, 0.7),
            lookat=(0.60, 0.0, 0.16),
            fov=35,
            GUI=False,
        )
        cam_wrist = _add_wrist_camera(scene, franka, res=args.vision_resolution)

        scene.build(n_envs=1)

        support_entity = tile_entity if use_tile else floor_entity
        vision_cam = cam_0 if camera_name == "camera_0" else cam_wrist
        render_cam = vision_cam

        motors_dof = np.arange(7)
        fingers_dof = np.arange(7, 9)
        end_effector = franka.get_link("hand")
        hand_idx = next(i for i, link in enumerate(franka.links) if link.name == "hand")
        left_finger_idx = next(i for i, link in enumerate(franka.links) if link.name == "left_finger")
        finger_len = infer_finger_length_from_mjcf(PANDA_XML_PATH)

        franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
        franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
        franka.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )

        home_q = ik_pose(franka, end_effector, pos=(0.45, 0.0, 0.28))
        home_q[0, -2:] = 0.04
        franka.set_dofs_position(home_q[0, :-2], motors_dof)
        franka.set_dofs_position(home_q[0, -2:], fingers_dof)

        should_record = bool(args.video) and (args.save_all_videos or trial_idx == 0)
        if should_record:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = output_dir / f"{camera_name}_{'tile' if use_tile else 'no_tile'}_trial{trial_idx + 1:02d}_{ts}.mp4"
            render_cam.start_recording()

        for _ in range(120):
            control_pose(franka, motors_dof, fingers_dof, home_q, gripper_opening=0.04)
            sim_step(scene, cam=render_cam, record=should_record)

        metrics = pick_lift_and_visual_release(
            scene=scene,
            franka=franka,
            end_effector=end_effector,
            motors_dof=motors_dof,
            fingers_dof=fingers_dof,
            hand_idx=hand_idx,
            left_finger_idx=left_finger_idx,
            obj_entity=obj_entity,
            support_entity=support_entity,
            finger_len=finger_len,
            obj_xy=obj_xy,
            motion_speed=args.motion_speed,
            post_descend_wait_steps=args.post_descend_wait_steps,
            vision_cam=vision_cam,
            render_cam=render_cam,
            record=should_record,
            client=client,
            openai_model=args.openai_model,
            align_vertices_local=align_vertices_local,
            align_scale=align_scale,
            eval_interval_steps=args.eval_interval_steps,
        )

        move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (0.45, 0.0, 0.30), 0.04, scaled_steps(140, args.motion_speed), render_cam, should_record)
        for _ in range(120):
            control_pose(franka, motors_dof, fingers_dof, ik_pose(franka, end_effector, (0.45, 0.0, 0.30)), gripper_opening=0.04)
            sim_step(scene, cam=render_cam, record=should_record)

        if should_record:
            render_cam.stop_recording(save_to_filename=str(video_path), fps=args.effective_fps)
            should_record = False

        score_trace_csv = csv_dir / "score_trace.csv"
        save_dict_rows_to_csv(metrics.get("score_trace", []), score_trace_csv)
        save_force_plot_from_score_trace(metrics.get("score_trace", []), csv_dir / "force_plot.png")

        return {
            "trial_index": trial_idx + 1,
            "camera": camera_name,
            "use_tile": bool(use_tile),
            "object_params": params,
            "trial_dir": str(trial_dir),
            "score_trace_csv": str(score_trace_csv),
            "video_path": str(video_path) if video_path is not None else None,
            "metrics": metrics,
        }
    finally:
        try:
            if should_record and (render_cam is not None):
                render_cam.stop_recording(save_to_filename=str(video_path), fps=args.effective_fps)
        except Exception:
            pass
        gs.destroy()


def summarize_trials(trials: list[dict]) -> dict:
    total = len(trials)
    support_touched = [t for t in trials if t["metrics"]["support_touched_at_release"]]
    released_by_vision = [t for t in trials if t["metrics"]["release_reason"].startswith("openai_visual_touch")]

    delay_vals = [t["metrics"]["delay_sec_after_touch"] for t in trials if t["metrics"]["delay_sec_after_touch"] is not None]
    rho_vals = [float(t["object_params"]["rho"]) for t in trials]
    sizes = [float(t["object_params"]["size"]) for t in trials if t["object_params"]["size"] is not None]
    scales = [float(t["object_params"]["scale"]) for t in trials if t["object_params"]["scale"] is not None]

    return {
        "trials": total,
        "support_touched_at_release_ratio": float(len(support_touched) / max(1, total)),
        "vision_release_ratio": float(len(released_by_vision) / max(1, total)),
        "delay_sec_after_touch": _stats(delay_vals),
        "rho": _stats(rho_vals),
        "size": _stats(sizes),
        "scale": _stats(scales),
    }


def is_valid_evaluation_trial(trial: dict) -> tuple[bool, str]:
    metrics = trial.get("metrics", {})
    trace = metrics.get("score_trace", [])
    if not trace:
        return False, "empty_score_trace"
    first = trace[0]
    if bool(first.get("touching_support_gt", False)):
        return False, "already_touching_support_at_first_eval"
    return True, "valid"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-type", choices=["cube", "bottle", "ycb_wood_block"], default="ycb_wood_block")
    parser.add_argument("--video", default="")
    parser.add_argument("--save-all-videos", action="store_true")
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--motion-speed", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "data" / "evaluation_outputs" / "openai_vision_baseline"))
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument(
        "--max-attempt-factor",
        type=float,
        default=3.0,
        help="Max attempts = ceil(trials * factor) for replacing invalid trials.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--openai-model", type=str, default="gpt-4o")
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key. If omitted, OPENAI_API_KEY env var is used.",
    )
    parser.add_argument("--cuda-device", type=int, default=0, help="CUDA device index used by Genesis (e.g., 1 for cuda:1).")
    parser.add_argument(
        "--eval-interval-steps",
        type=int,
        default=10,
        help="Run OpenAI visual touch classification every N sim steps (larger is faster).",
    )
    parser.add_argument(
        "--vision-resolution",
        type=int,
        default=640,
        help="Square camera resolution for vision inference (pixels).",
    )
    parser.add_argument(
        "--openai-jpeg-quality",
        type=int,
        default=75,
        help="JPEG quality (30-95) for OpenAI image upload; lower is faster/smaller.",
    )
    parser.add_argument(
        "--save-trial-images",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--trial-image-stride", type=int, default=10)
    parser.add_argument("--post-descend-wait-steps", type=int, default=1200)
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=["camera_0:no_tile", "camera_0:tile", "camera_wrist:no_tile", "camera_wrist:tile"],
        help="Pattern list: camera_0:no_tile, camera_0:tile, camera_wrist:no_tile, camera_wrist:tile",
    )
    args = parser.parse_args()
    global OPENAI_JPEG_QUALITY
    OPENAI_JPEG_QUALITY = int(args.openai_jpeg_quality)

    if args.cuda_device < 0:
        raise ValueError("--cuda-device must be >= 0")
    if args.eval_interval_steps < 1:
        raise ValueError("--eval-interval-steps must be >= 1")
    if args.vision_resolution < 128:
        raise ValueError("--vision-resolution must be >= 128")
    if args.max_attempt_factor < 1.0:
        raise ValueError("--max-attempt-factor must be >= 1.0")

    if not args.video and args.save_all_videos:
        raise ValueError("--save-all-videos requires --video")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    api_key = _normalize_api_key(args.openai_api_key) or _normalize_api_key(os.getenv("OPENAI_API_KEY"))
    if not api_key:
        raise RuntimeError(
            "OpenAI API key not found. Pass --openai-api-key or set OPENAI_API_KEY in the same shell running this script."
        )
    client = OpenAI(api_key=api_key)

    render_every = max(1, int(round(SIM_HZ / max(1, args.video_fps))))
    global RENDER_EVERY
    RENDER_EVERY = render_every
    args.effective_fps = int(round(SIM_HZ / render_every))

    base_rng = random.Random(args.seed)
    pattern_payloads = []

    for p in args.patterns:
        cam_key, tile_key = p.split(":")
        if cam_key not in ("camera_0", "camera_wrist"):
            raise ValueError(f"Unknown camera pattern: {cam_key}")
        if tile_key not in ("tile", "no_tile"):
            raise ValueError(f"Unknown tile pattern: {tile_key}")
        use_tile = tile_key == "tile"

        scenario_name = f"{cam_key}_{'tile' if use_tile else 'no_tile'}"
        scenario_dir = output_root / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        rng = random.Random(base_rng.randint(0, 10**9))
        valid_trials = []
        invalid_trials = []
        max_attempts = max(args.trials, int(np.ceil(args.trials * args.max_attempt_factor)))
        trial_idx = 0
        while len(valid_trials) < args.trials and trial_idx < max_attempts:
            params = sample_object_params(args.object_type, rng)
            trial = run_single_trial(
                trial_idx=trial_idx,
                args=args,
                params=params,
                output_dir=scenario_dir,
                camera_name=cam_key,
                use_tile=use_tile,
                client=client,
            )
            is_valid, invalid_reason = is_valid_evaluation_trial(trial)
            trial["valid_for_eval"] = bool(is_valid)
            trial["invalid_reason"] = None if is_valid else invalid_reason
            if is_valid:
                valid_trials.append(trial)
            else:
                invalid_trials.append(trial)
            delay = trial["metrics"]["delay_sec_after_touch"]
            print(
                f"[{scenario_name}][attempt {trial_idx + 1:02d}/{max_attempts}] "
                f"rho={params['rho']:.1f} "
                f"size={params['size'] if params['size'] is not None else '-'} "
                f"scale={params['scale'] if params['scale'] is not None else '-'} "
                f"valid={is_valid} "
                f"reason={trial['metrics']['release_reason']} "
                f"delay_sec={delay if delay is not None else 'None'}"
            )
            trial_idx += 1

        if len(valid_trials) < args.trials:
            raise RuntimeError(
                f"[{scenario_name}] valid trials shortage: {len(valid_trials)}/{args.trials} "
                f"after {trial_idx} attempts. Increase --max-attempt-factor."
            )

        summary = summarize_trials(valid_trials)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = scenario_dir / f"lift_and_lower_openai_vision_{args.object_type}_{scenario_name}_summary_{ts}.json"
        payload = {
            "openai_model": args.openai_model,
            "object_type": args.object_type,
            "camera": cam_key,
            "use_tile": use_tile,
            "motion_speed": args.motion_speed,
            "sim_hz": SIM_HZ,
            "render_every": render_every,
            "effective_fps": args.effective_fps,
            "trials": len(valid_trials),
            "valid_trials_target": args.trials,
            "attempts_total": trial_idx,
            "invalid_trials_count": len(invalid_trials),
            "seed": args.seed,
            "trial_results": valid_trials,
            "invalid_trial_results": invalid_trials,
            "summary": summary,
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        pattern_payloads.append({"scenario": scenario_name, "log_path": str(log_path), "summary": summary})

    consolidated_path = output_root / f"lift_and_lower_openai_vision_all_patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(consolidated_path, "w", encoding="utf-8") as f:
        json.dump({"patterns": pattern_payloads}, f, ensure_ascii=False, indent=2)

    print("all pattern summaries:")
    print(json.dumps(pattern_payloads, ensure_ascii=False, indent=2))
    print(f"saved consolidated log -> {consolidated_path}")


if __name__ == "__main__":
    main()
