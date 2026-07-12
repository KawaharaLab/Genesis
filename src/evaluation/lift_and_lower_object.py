import argparse
import csv
import json
import random
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import genesis as gs
import genesis.utils.geom as gu
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
PATCHTST_ROOT = ROOT / "external" / "PatchTST_tmp" / "PatchTST_self_supervised"
if str(PATCHTST_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCHTST_ROOT))

from src.models.patchTST import PatchTST  # noqa: E402
from src.callback.patch_mask import create_patch  # noqa: E402


DT = 0.01
HOLD_SECONDS = 1.0
HOLD_STEPS = int(HOLD_SECONDS / DT)
SIM_HZ = int(round(1.0 / DT))
SEQ_LEN = 80
RENDER_EVERY = 1
SIM_STEP_COUNT = 0
TRACE_RECORDER = {
    "enabled": False,
    "images_dir": None,
    "image_stride": 80,
}
STEP_TRACE_RECORDER = {
    "enabled": False,
    "rows": [],
    "franka": None,
    "obj_entity": None,
    "support_entity": None,
}

PANDA_XML_PATH = ROOT / "genesis" / "assets" / "xml" / "franka_emika_panda" / "panda.xml"
DEFAULT_CKPT = ROOT / "data" / "PatchTST" / "twilight-music-38_epoch4000.pth"
TEXT_EMB_DIR = ROOT / "data" / "text_emb"
YCB_TARGET_OBJ_PATH = ROOT / "data" / "objects" / "ycb" / "009_gelatin_box" / "model.obj"


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


def build_patchtst_model(nvars=12, context_points=80, patch_len=10, stride=10, n_layers=3, n_heads=16, d_model=128, d_ff=768):
    num_patch = (max(context_points, patch_len) - patch_len) // stride + 1
    model = PatchTST(
        c_in=nvars,
        target_dim=96,
        patch_len=patch_len,
        stride=stride,
        num_patch=num_patch,
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        shared_embedding=True,
        d_ff=d_ff,
        dropout=0.0,
        head_dropout=0.0,
        act="gelu",
        head_type="pretrain",
        res_attention=False,
    )
    return model


def load_contrastive_patchtst(model_path: Path, device: torch.device):
    model = build_patchtst_model().to(device)
    proj = None

    ckpt_all = torch.load(model_path, map_location=device)
    state = ckpt_all.get("model", ckpt_all)

    if any(k.startswith("proj.") for k in state.keys()):
        if "proj.3.weight" in state and "proj.0.weight" in state:
            in_dim = state["proj.0.weight"].shape[1]
            hidden = state["proj.0.weight"].shape[0]
            out_dim = state["proj.3.weight"].shape[0]
            proj = torch.nn.Sequential(
                torch.nn.Linear(in_dim, hidden),
                torch.nn.GELU(),
                torch.nn.Dropout(0.0),
                torch.nn.Linear(hidden, out_dim),
            ).to(device)
        elif "proj.0.weight" in state:
            in_dim = state["proj.0.weight"].shape[1]
            out_dim = state["proj.0.weight"].shape[0]
            proj = torch.nn.Sequential(torch.nn.Linear(in_dim, out_dim)).to(device)

    cur = model.state_dict()
    for k, v in state.items():
        if k.startswith("backbone.") and k in cur and cur[k].shape == v.shape:
            cur[k] = v
    model.load_state_dict(cur, strict=False)

    if proj is not None:
        p_state = proj.state_dict()
        for k, v in state.items():
            if not k.startswith("proj."):
                continue
            k2 = k.replace("proj.", "")
            if k2 in p_state and p_state[k2].shape == v.shape:
                p_state[k2] = v
        proj.load_state_dict(p_state, strict=False)
        proj.eval()

    model.eval()
    return model, proj


def load_text_embedding(label: str) -> torch.Tensor:
    path = TEXT_EMB_DIR / f"{label}.pt"
    t = torch.load(path, map_location="cpu")
    if isinstance(t, torch.Tensor):
        if t.ndim == 2 and t.shape[0] == 1:
            t = t.squeeze(0)
        t = t.float()
    else:
        t = torch.tensor(t, dtype=torch.float32)
    t = t / (t.norm() + 1e-8)
    return t


def embed_force_segment(model, proj, force_seq_80x12: np.ndarray) -> torch.Tensor:
    x = force_seq_80x12.astype(np.float32)[None, ...]
    x_t = torch.from_numpy(x)
    device = next(model.parameters()).device
    with torch.no_grad():
        xb_patch, _ = create_patch(x_t.to(device), patch_len=10, stride=10)
        z = model.backbone(xb_patch)
        z_pool = z.mean(dim=(1, 3))
        if proj is not None:
            z_pool = proj(z_pool)
        z_pool = z_pool / (z_pool.norm(dim=-1, keepdim=True) + 1e-8)
    return z_pool[0].detach().cpu()


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
    if STEP_TRACE_RECORDER["enabled"]:
        franka = STEP_TRACE_RECORDER["franka"]
        obj_entity = STEP_TRACE_RECORDER["obj_entity"]
        support_entity = STEP_TRACE_RECORDER["support_entity"]
        force12 = get_force_vector_12(franka)
        touching_support = False
        if obj_entity is not None and support_entity is not None:
            contacts = obj_entity.get_contacts(with_entity=support_entity)
            valid = contacts.get("valid_mask", None)
            if valid is not None:
                if hasattr(valid, "detach"):
                    valid = valid.detach().cpu().numpy()
                touching_support = bool(np.asarray(valid).astype(bool).any())
        STEP_TRACE_RECORDER["rows"].append(
            {
                "step": int(scene.t),
                "sim_step_count": int(SIM_STEP_COUNT),
                "touching_support": bool(touching_support),
                "left_fx": float(force12[0]),
                "left_fy": float(force12[1]),
                "left_fz": float(force12[2]),
                "left_tx": float(force12[3]),
                "left_ty": float(force12[4]),
                "left_tz": float(force12[5]),
                "right_fx": float(force12[6]),
                "right_fy": float(force12[7]),
                "right_fz": float(force12[8]),
                "right_tx": float(force12[9]),
                "right_ty": float(force12[10]),
                "right_tz": float(force12[11]),
            }
        )
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
    hold = [r["hold_score"] for r in rows]
    place = [r["place_score"] for r in rows]
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(steps, hold, label="hold_score")
    ax.plot(steps, place, label="place_score")
    ax.set_xlabel("Step")
    ax.set_ylabel("Score")
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


def get_force_vector_12(franka) -> np.ndarray:
    f = np.asarray(franka.get_links_contact_force([9, 10], sensor=True).cpu().numpy(), dtype=np.float32)
    t = np.asarray(franka.get_links_contact_torque([9, 10], sensor=True).cpu().numpy(), dtype=np.float32)
    if f.ndim == 3:
        f = f[0]
    if t.ndim == 3:
        t = t[0]
    return np.concatenate([f[0], t[0], f[1], t[1]], axis=0).astype(np.float32)


def get_contact_pairs(obj_entity) -> set[int]:
    contacts = obj_entity.get_contacts()
    link_a = contacts.get("link_a", [])
    link_b = contacts.get("link_b", [])
    out = set()
    for arr in (link_a, link_b):
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        for v in np.asarray(arr).reshape(-1).tolist():
            try:
                out.add(int(v))
            except (TypeError, ValueError):
                pass
    return out


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


def classify_hold_place(model, proj, seq_buffer, hold_vec, place_vec):
    seq = np.asarray(seq_buffer, dtype=np.float32)
    if seq.shape[0] < SEQ_LEN:
        pad = np.repeat(seq[-1:, :], SEQ_LEN - seq.shape[0], axis=0)
        seq = np.concatenate([seq, pad], axis=0)
    else:
        seq = seq[-SEQ_LEN:, :]
    emb = embed_force_segment(model, proj, seq)
    hold_score = float(torch.dot(emb, hold_vec))
    place_score = float(torch.dot(emb, place_vec))
    return hold_score, place_score


def pick_lift_and_adaptive_release(
    scene,
    franka,
    end_effector,
    motors_dof,
    fingers_dof,
    hand_idx,
    left_finger_idx,
    obj_entity,
    finger_len,
    obj_xy,
    model,
    proj,
    hold_vec,
    place_vec,
    motion_speed=1.0,
    post_descend_wait_steps=1200,
    cam=None,
    record=False,
    align_vertices_local=None,
    align_scale=1.0,
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
        cam=cam,
        record=record,
        vertices_local=align_vertices_local,
        obj_scale=align_scale,
    )
    x, y = compute_grasp_xy_from_aabb(obj_entity, fallback_xy=obj_xy)

    move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, hover_z), 0.04, scaled_steps(150, motion_speed), cam, record)
    move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, grasp_z), 0.04, scaled_steps(100, motion_speed), cam, record)
    move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, clamp_z), 0.04, scaled_steps(35, motion_speed), cam, record)
    set_gripper(scene, franka, motors_dof, fingers_dof, 0.04, 0.0, scaled_steps(60, motion_speed), cam, record)

    move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, safe_retract_z), 0.0, scaled_steps(50, motion_speed), cam, record)
    move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, lift_z), 0.0, scaled_steps(120, motion_speed), cam, record)
    hold_pose(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, lift_z), 0.0, HOLD_STEPS, cam, record)

    move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, safe_retract_z), 0.0, scaled_steps(80, motion_speed), cam, record)

    # Adaptive release while descending.
    start_pos = franka.get_links_pos([8])[0].cpu().numpy().reshape(3)
    target_pos = np.array([x, y, lower_back_z], dtype=float)
    descend_steps = scaled_steps(140, motion_speed)

    floor_touch_step = None
    release_step = None
    release_reason = "place_gt_hold"
    buffer = []
    force_between_touch_and_release = []
    eval_records = []

    for i in range(descend_steps):
        alpha = (i + 1) / descend_steps
        interp = (1.0 - alpha) * start_pos + alpha * target_pos
        qpos = ik_pose(franka, end_effector, interp)
        control_pose(franka, motors_dof, fingers_dof, qpos, gripper_opening=0.0)
        sim_step(scene, cam=cam, record=record)

        force12 = get_force_vector_12(franka)
        buffer.append(force12)
        if len(buffer) > SEQ_LEN:
            buffer = buffer[-SEQ_LEN:]

        pairs = get_contact_pairs(obj_entity)
        touching_floor = 0 in pairs
        if touching_floor and floor_touch_step is None:
            floor_touch_step = int(scene.t)

        if floor_touch_step is not None and release_step is None:
            force_between_touch_and_release.append(force12)

        # Evaluate every 10 sim steps (= 0.1 s at DT=0.01) once we have enough temporal context.
        if len(buffer) >= max(20, SEQ_LEN // 2) and (i % 10 == 0):
            hold_score, place_score = classify_hold_place(model, proj, buffer, hold_vec, place_vec)
            eval_records.append(
                {
                    "step": int(scene.t),
                    "hold_score": hold_score,
                    "place_score": place_score,
                    "touching_floor": bool(touching_floor),
                }
            )
            if place_score > hold_score:
                release_step = int(scene.t)
                release_reason = "place_gt_hold_during_descent"
                break

    # If not released yet, keep descending-end pose and continue evaluating.
    if release_step is None:
        wait_steps = max(1, int(post_descend_wait_steps))
        qpos_hold = ik_pose(franka, end_effector, (x, y, lower_back_z))
        for i in range(wait_steps):
            control_pose(franka, motors_dof, fingers_dof, qpos_hold, gripper_opening=0.0)
            sim_step(scene, cam=cam, record=record)

            force12 = get_force_vector_12(franka)
            buffer.append(force12)
            if len(buffer) > SEQ_LEN:
                buffer = buffer[-SEQ_LEN:]

            pairs = get_contact_pairs(obj_entity)
            touching_floor = 0 in pairs
            if touching_floor and floor_touch_step is None:
                floor_touch_step = int(scene.t)

            if floor_touch_step is not None and release_step is None:
                force_between_touch_and_release.append(force12)

            if len(buffer) >= max(20, SEQ_LEN // 2) and (i % 10 == 0):
                hold_score, place_score = classify_hold_place(model, proj, buffer, hold_vec, place_vec)
                eval_records.append(
                    {
                        "step": int(scene.t),
                        "hold_score": hold_score,
                        "place_score": place_score,
                        "touching_floor": bool(touching_floor),
                    }
                )
                if place_score > hold_score:
                    release_step = int(scene.t)
                    release_reason = "place_gt_hold_after_descent"
                    break

    released = release_step is not None
    if released:
        set_gripper(scene, franka, motors_dof, fingers_dof, 0.0, 0.04, scaled_steps(40, motion_speed), cam, record)
    else:
        release_reason = "hold_dominant_no_release"

    if (floor_touch_step is not None) and (release_step is not None):
        delay_steps = max(0, release_step - floor_touch_step)
        delay_sec = delay_steps * DT
    else:
        delay_steps = None
        delay_sec = None

    if len(force_between_touch_and_release) > 0:
        arr = np.asarray(force_between_touch_and_release, dtype=np.float32)
        f_mag = np.linalg.norm(arr[:, [0, 1, 2]], axis=1)
        t_mag = np.linalg.norm(arr[:, [3, 4, 5]], axis=1)
        force_stats = {
            "samples": int(arr.shape[0]),
            "left_force_mean": float(np.mean(f_mag)),
            "left_force_max": float(np.max(f_mag)),
            "left_torque_mean": float(np.mean(t_mag)),
            "left_torque_max": float(np.max(t_mag)),
            "left_force_impulse_approx": float(np.sum(f_mag) * DT),
        }
    else:
        force_stats = {
            "samples": 0,
            "left_force_mean": None,
            "left_force_max": None,
            "left_torque_mean": None,
            "left_torque_max": None,
            "left_force_impulse_approx": None,
        }

    move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, hover_z), 0.04, scaled_steps(100, motion_speed), cam, record)

    return {
        "release_reason": release_reason,
        "release_step": release_step,
        "floor_touch_step": floor_touch_step,
        "released": bool(released),
        "floor_touched_at_release": bool(released and floor_touch_step is not None and release_step >= floor_touch_step),
        "delay_steps_after_touch": delay_steps,
        "delay_sec_after_touch": delay_sec,
        "force_stats_between_touch_and_release": force_stats,
        "score_trace": eval_records,
        "score_trace_tail": eval_records[-12:],
    }


def sample_object_params(object_type: str, rng: random.Random) -> dict:
    # Match run_genesis_sim style random density: rho = 200 * U(1,10)
    rho = 200.0 * rng.uniform(1.0, 10.0)

    if object_type == "cube":
        # Keep side length inside gripper capture range (roughly < 0.08 m total opening).
        size = rng.uniform(0.030, 0.075)
        return {
            "rho": float(rho),
            "size": float(size),
            "scale": None,
        }

    if object_type == "bottle":
        scale = rng.uniform(0.075, 0.105)
        return {
            "rho": float(rho),
            "size": None,
            "scale": float(scale),
        }
    return {
        "rho": float(rho),
        "size": None,
        "scale": None,
    }


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


def run_single_trial(trial_idx, args, model, proj, hold_vec, place_vec, params, output_dir):
    global SIM_STEP_COUNT
    SIM_STEP_COUNT = 0
    global TRACE_RECORDER, STEP_TRACE_RECORDER

    trial_dir = output_dir / f"trial_{trial_idx + 1:02d}"
    images_dir = trial_dir / "images" / "camera_0"
    csv_dir = trial_dir / "csv"
    images_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    TRACE_RECORDER["enabled"] = bool(args.save_trial_images)
    TRACE_RECORDER["images_dir"] = images_dir if args.save_trial_images else None
    TRACE_RECORDER["image_stride"] = int(args.trial_image_stride)

    if torch.cuda.is_available():
        gs.init(backend=gs.gpu, logging_level="warning")
    else:
        gs.init(backend=gs.cpu, logging_level="warning")

    scene = None
    cam = None
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
        if args.object_type == "cube":
            cube_size = params["size"]
            obj_entity = scene.add_entity(
                material=gs.materials.Rigid(rho=params["rho"], friction=1.0),
                morph=gs.morphs.Box(size=(cube_size, cube_size, cube_size), pos=(obj_xy[0], obj_xy[1], cube_size * 0.5)),
                surface=gs.surfaces.Default(color=(0.35, 0.75, 0.95, 1.0)),
            )
        elif args.object_type == "bottle":
            obj_entity = scene.add_entity(
                material=gs.materials.Rigid(rho=params["rho"]),
                morph=gs.morphs.URDF(
                    file="urdf/3763/mobility_vhacd.urdf",
                    scale=params["scale"],
                    pos=(obj_xy[0], obj_xy[1], 0.036 * (params["scale"] / 0.09)),
                    euler=(0, 90, 0),
                ),
            )
        else:
            object_scale, object_euler, min_corner = set_grasp_for_obj(YCB_TARGET_OBJ_PATH)
            align_vertices_local = _vertices_from_obj(YCB_TARGET_OBJ_PATH)
            align_scale = object_scale
            scale_vec = _scale_to_vec(object_scale)
            spawn_z = 0.001 + max(0.0, -scale_vec[2] * min_corner[2])
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
        cam = scene.add_camera(
            model="thinlens",
            res=(960, 960),
            pos=(1.7, 0.0, 0.7),
            lookat=(0.60, 0.0, 0.16),
            fov=35,
            GUI=False,
        )

        scene.build(n_envs=1)

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
        STEP_TRACE_RECORDER["enabled"] = True
        STEP_TRACE_RECORDER["rows"] = []
        STEP_TRACE_RECORDER["franka"] = franka
        STEP_TRACE_RECORDER["obj_entity"] = obj_entity
        STEP_TRACE_RECORDER["support_entity"] = floor_entity

        home_q = ik_pose(franka, end_effector, pos=(0.45, 0.0, 0.28))
        home_q[0, -2:] = 0.04
        franka.set_dofs_position(home_q[0, :-2], motors_dof)
        franka.set_dofs_position(home_q[0, -2:], fingers_dof)

        should_record = bool(args.video) and (args.save_all_videos or trial_idx == 0)
        if should_record:
            if args.trials == 1 and args.video:
                video_path = Path(args.video)
                if not video_path.is_absolute():
                    video_path = output_dir / video_path
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_path = output_dir / f"lift_and_lower_patchtst_{args.object_type}_trial{trial_idx + 1:02d}_{ts}.mp4"
            cam.start_recording()

        for _ in range(120):
            control_pose(franka, motors_dof, fingers_dof, home_q, gripper_opening=0.04)
            sim_step(scene, cam=cam, record=should_record)

        metrics = pick_lift_and_adaptive_release(
            scene,
            franka,
            end_effector,
            motors_dof,
            fingers_dof,
            hand_idx,
            left_finger_idx,
            obj_entity,
            finger_len,
            obj_xy,
            model,
            proj,
            hold_vec,
            place_vec,
            motion_speed=args.motion_speed,
            post_descend_wait_steps=args.post_descend_wait_steps,
            cam=cam,
            record=should_record,
            align_vertices_local=align_vertices_local,
            align_scale=align_scale,
        )

        move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (0.45, 0.0, 0.30), 0.04, scaled_steps(140, args.motion_speed), cam, should_record)

        for _ in range(120):
            control_pose(franka, motors_dof, fingers_dof, ik_pose(franka, end_effector, (0.45, 0.0, 0.30)), gripper_opening=0.04)
            sim_step(scene, cam=cam, record=should_record)

        if should_record:
            cam.stop_recording(save_to_filename=str(video_path), fps=args.effective_fps)
            should_record = False

        score_trace_csv = csv_dir / "score_trace.csv"
        save_dict_rows_to_csv(metrics.get("score_trace", []), score_trace_csv)
        force_stats_csv = csv_dir / "force_stats.csv"
        save_dict_rows_to_csv([metrics.get("force_stats_between_touch_and_release", {})], force_stats_csv)
        force_contact_trace_csv = csv_dir / "force_contact_trace.csv"
        save_dict_rows_to_csv(STEP_TRACE_RECORDER["rows"], force_contact_trace_csv)
        save_force_plot_from_score_trace(metrics.get("score_trace", []), csv_dir / "force_plot.png")

        return {
            "trial_index": trial_idx + 1,
            "object_params": params,
            "trial_dir": str(trial_dir),
            "score_trace_csv": str(score_trace_csv),
            "force_stats_csv": str(force_stats_csv),
            "force_contact_trace_csv": str(force_contact_trace_csv),
            "video_path": str(video_path) if video_path is not None else None,
            "metrics": metrics,
        }
    finally:
        STEP_TRACE_RECORDER["enabled"] = False
        STEP_TRACE_RECORDER["rows"] = []
        STEP_TRACE_RECORDER["franka"] = None
        STEP_TRACE_RECORDER["obj_entity"] = None
        STEP_TRACE_RECORDER["support_entity"] = None
        try:
            if should_record and cam is not None:
                cam.stop_recording(save_to_filename=str(video_path), fps=args.effective_fps)
        except Exception:
            pass
        gs.destroy()


def summarize_trials(trials: list[dict]) -> dict:
    total = len(trials)
    floor_touched = [t for t in trials if t["metrics"]["floor_touched_at_release"]]
    released_by_classifier = [t for t in trials if t["metrics"]["release_reason"].startswith("place_gt_hold")]

    delay_vals = [t["metrics"]["delay_sec_after_touch"] for t in trials if t["metrics"]["delay_sec_after_touch"] is not None]
    force_mean_vals = [
        t["metrics"]["force_stats_between_touch_and_release"]["left_force_mean"]
        for t in trials
        if t["metrics"]["force_stats_between_touch_and_release"]["left_force_mean"] is not None
    ]
    rho_vals = [float(t["object_params"]["rho"]) for t in trials]

    sizes = [float(t["object_params"]["size"]) for t in trials if t["object_params"]["size"] is not None]
    scales = [float(t["object_params"]["scale"]) for t in trials if t["object_params"]["scale"] is not None]

    return {
        "trials": total,
        "floor_touched_at_release_ratio": float(len(floor_touched) / max(1, total)),
        "classifier_release_ratio": float(len(released_by_classifier) / max(1, total)),
        "delay_sec_after_touch": _stats(delay_vals),
        "left_force_mean_between_touch_and_release": _stats(force_mean_vals),
        "rho": _stats(rho_vals),
        "size": _stats(sizes),
        "scale": _stats(scales),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-type", choices=["cube", "bottle", "ycb_wood_block"], default="cube")
    parser.add_argument("--video", default="")
    parser.add_argument("--save-all-videos", action="store_true", help="If --video is enabled, save every trial video.")
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--motion-speed", type=float, default=1.0, help=">1.0 makes robot motions faster.")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "data" / "evaluation_outputs" / "lift_and_lower"))
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-trial-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-trial camera snapshots.",
    )
    parser.add_argument("--trial-image-stride", type=int, default=20, help="Save one image every N sim steps.")
    parser.add_argument(
        "--post-descend-wait-steps",
        type=int,
        default=1200,
        help="Max steps to keep waiting at the lowered pose while hold similarity dominates.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, proj = load_contrastive_patchtst(Path(args.model_path), device)
    hold_vec = load_text_embedding("hold")
    place_vec = load_text_embedding("place")

    render_every = max(1, int(round(SIM_HZ / max(1, args.video_fps))))
    global RENDER_EVERY
    RENDER_EVERY = render_every
    effective_fps = int(round(SIM_HZ / render_every))
    args.effective_fps = effective_fps

    rng = random.Random(args.seed)
    trials = []
    for trial_idx in range(args.trials):
        params = sample_object_params(args.object_type, rng)
        trial = run_single_trial(trial_idx, args, model, proj, hold_vec, place_vec, params, output_dir)
        trials.append(trial)
        delay = trial["metrics"]["delay_sec_after_touch"]
        print(
            f"[trial {trial_idx + 1:02d}/{args.trials}] "
            f"rho={params['rho']:.1f} "
            f"size={params['size'] if params['size'] is not None else '-'} "
            f"scale={params['scale'] if params['scale'] is not None else '-'} "
            f"reason={trial['metrics']['release_reason']} "
            f"delay_sec={delay if delay is not None else 'None'}"
        )

    summary = summarize_trials(trials)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"lift_and_lower_patchtst_{args.object_type}_summary_{ts}.json"
    payload = {
        "model_path": args.model_path,
        "object_type": args.object_type,
        "motion_speed": args.motion_speed,
        "sim_hz": SIM_HZ,
        "render_every": render_every,
        "effective_fps": effective_fps,
        "trials": args.trials,
        "seed": args.seed,
        "trial_results": trials,
        "summary": summary,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved log -> {log_path}")


if __name__ == "__main__":
    main()
