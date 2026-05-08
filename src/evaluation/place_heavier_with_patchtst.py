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
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
PATCHTST_ROOT = ROOT / "external" / "PatchTST_tmp" / "PatchTST_self_supervised"
if str(PATCHTST_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCHTST_ROOT))

from src.models.patchTST import PatchTST  # noqa: E402
from src.callback.patch_mask import create_patch  # noqa: E402


DT = 0.01
SIM_HZ = int(round(1.0 / DT))
HOLD_SECONDS = 1.0
HOLD_STEPS = int(HOLD_SECONDS / DT)
SEQ_LEN = 80
SIM_STEP_COUNT = 0
TRACE_RECORDER = {
    "enabled": False,
    "images_dir": None,
    "image_stride": 20,
}

PANDA_XML_PATH = ROOT / "genesis" / "assets" / "xml" / "franka_emika_panda" / "panda.xml"
DEFAULT_CKPT = ROOT / "data" / "PatchTST" / "twilight-music-38_epoch4000.pth"
TEXT_EMB_DIR = ROOT / "data" / "text_emb"


def scaled_steps(base_steps: int, motion_speed: float) -> int:
    # motion_speed > 1.0 makes motions faster by reducing interpolation steps.
    return max(8, int(round(base_steps / max(motion_speed, 1e-3))))


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
    should_render = record or TRACE_RECORDER["enabled"]
    if should_render and cam is not None:
        rgb, _, _, _ = cam.render(rgb=True)
        if TRACE_RECORDER["enabled"] and TRACE_RECORDER["images_dir"] is not None:
            if SIM_STEP_COUNT % max(1, int(TRACE_RECORDER["image_stride"])) == 0:
                frame_path = TRACE_RECORDER["images_dir"] / f"frame_{SIM_STEP_COUNT:06d}.png"
                plt.imsave(frame_path, rgb)
    SIM_STEP_COUNT += 1


def save_matrix_csv(matrix: np.ndarray, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(matrix, dtype=np.float32)
    header = [f"ch_{i}" for i in range(arr.shape[1])]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(arr.tolist())


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


def hold_and_collect_force(
    scene,
    franka,
    end_effector,
    motors_dof,
    fingers_dof,
    pos,
    gripper_opening,
    steps,
    cam=None,
    record=False,
):
    qpos = ik_pose(franka, end_effector, pos)
    seq = []
    for _ in range(max(1, steps)):
        control_pose(franka, motors_dof, fingers_dof, qpos, gripper_opening=gripper_opening)
        sim_step(scene, cam=cam, record=record)
        f = np.asarray(franka.get_links_contact_force([9, 10], sensor=True).cpu().numpy(), dtype=np.float32)
        t = np.asarray(franka.get_links_contact_torque([9, 10], sensor=True).cpu().numpy(), dtype=np.float32)
        if f.ndim == 3:
            f = f[0]
        if t.ndim == 3:
            t = t[0]
        v12 = np.concatenate([f[0], t[0], f[1], t[1]], axis=0).astype(np.float32)
        seq.append(v12)

    arr = np.asarray(seq, dtype=np.float32)
    if arr.shape[0] < SEQ_LEN:
        pad = np.repeat(arr[-1:, :], SEQ_LEN - arr.shape[0], axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    return arr[-SEQ_LEN:, :]


def get_aabb_min_max(obj_entity):
    aabb = obj_entity.get_AABB().cpu().numpy().reshape(-1, 3)
    return aabb[0], aabb[1]


def get_obj_xy(obj_entity):
    lower, upper = get_aabb_min_max(obj_entity)
    center = 0.5 * (lower + upper)
    return float(center[0]), float(center[1])


def compute_grasp_heights(franka, hand_idx, left_finger_idx, obj_entity, finger_len, hover_z=0.24):
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
    return hover_z, grasp_hand_z, clamp_hand_z, safe_retract_z, lift_z


def pick_lift_hold_and_putback(
    scene,
    franka,
    end_effector,
    motors_dof,
    fingers_dof,
    hand_idx,
    left_finger_idx,
    obj_entity,
    finger_len,
    xy,
    motion_speed=1.0,
    cam=None,
    record=False,
):
    x, y = xy
    hover_z, grasp_z, clamp_z, safe_retract_z, lift_z = compute_grasp_heights(
        franka, hand_idx, left_finger_idx, obj_entity, finger_len
    )

    move_ee(
        scene, franka, end_effector, motors_dof, fingers_dof, (x, y, hover_z), 0.04, scaled_steps(130, motion_speed), cam, record
    )
    move_ee(
        scene, franka, end_effector, motors_dof, fingers_dof, (x, y, grasp_z), 0.04, scaled_steps(100, motion_speed), cam, record
    )
    move_ee(
        scene, franka, end_effector, motors_dof, fingers_dof, (x, y, clamp_z), 0.04, scaled_steps(35, motion_speed), cam, record
    )

    set_gripper(scene, franka, motors_dof, fingers_dof, 0.04, 0.0, scaled_steps(60, motion_speed), cam, record)
    move_ee(
        scene, franka, end_effector, motors_dof, fingers_dof, (x, y, safe_retract_z), 0.0, scaled_steps(50, motion_speed), cam, record
    )
    move_ee(
        scene, franka, end_effector, motors_dof, fingers_dof, (x, y, lift_z), 0.0, scaled_steps(100, motion_speed), cam, record
    )

    force_seq = hold_and_collect_force(
        scene,
        franka,
        end_effector,
        motors_dof,
        fingers_dof,
        (x, y, lift_z),
        0.0,
        HOLD_STEPS,
        cam,
        record,
    )

    move_ee(
        scene, franka, end_effector, motors_dof, fingers_dof, (x, y, safe_retract_z), 0.0, scaled_steps(70, motion_speed), cam, record
    )
    move_ee(
        scene, franka, end_effector, motors_dof, fingers_dof, (x, y, clamp_z), 0.0, scaled_steps(110, motion_speed), cam, record
    )
    set_gripper(scene, franka, motors_dof, fingers_dof, 0.0, 0.04, scaled_steps(60, motion_speed), cam, record)
    move_ee(
        scene, franka, end_effector, motors_dof, fingers_dof, (x, y, hover_z), 0.04, scaled_steps(80, motion_speed), cam, record
    )
    return force_seq


def pick_and_place_to_tile(
    scene,
    franka,
    end_effector,
    motors_dof,
    fingers_dof,
    hand_idx,
    left_finger_idx,
    obj_entity,
    finger_len,
    src_xy,
    dst_xy,
    motion_speed=1.0,
    cam=None,
    record=False,
):
    x, y = src_xy
    hover_z, grasp_z, clamp_z, safe_retract_z, lift_z = compute_grasp_heights(
        franka, hand_idx, left_finger_idx, obj_entity, finger_len
    )

    move_ee(
        scene, franka, end_effector, motors_dof, fingers_dof, (x, y, hover_z), 0.04, scaled_steps(120, motion_speed), cam, record
    )
    move_ee(
        scene, franka, end_effector, motors_dof, fingers_dof, (x, y, grasp_z), 0.04, scaled_steps(100, motion_speed), cam, record
    )
    move_ee(
        scene, franka, end_effector, motors_dof, fingers_dof, (x, y, clamp_z), 0.04, scaled_steps(35, motion_speed), cam, record
    )
    set_gripper(scene, franka, motors_dof, fingers_dof, 0.04, 0.0, scaled_steps(60, motion_speed), cam, record)
    move_ee(
        scene, franka, end_effector, motors_dof, fingers_dof, (x, y, safe_retract_z), 0.0, scaled_steps(50, motion_speed), cam, record
    )
    move_ee(
        scene, franka, end_effector, motors_dof, fingers_dof, (x, y, lift_z), 0.0, scaled_steps(100, motion_speed), cam, record
    )

    tx, ty = dst_xy
    move_ee(
        scene, franka, end_effector, motors_dof, fingers_dof, (tx, ty, lift_z), 0.0, scaled_steps(160, motion_speed), cam, record
    )
    place_z = max(clamp_z, 0.09)
    move_ee(
        scene, franka, end_effector, motors_dof, fingers_dof, (tx, ty, place_z), 0.0, scaled_steps(120, motion_speed), cam, record
    )
    set_gripper(scene, franka, motors_dof, fingers_dof, 0.0, 0.04, scaled_steps(60, motion_speed), cam, record)
    move_ee(
        scene, franka, end_effector, motors_dof, fingers_dof, (tx, ty, lift_z), 0.04, scaled_steps(90, motion_speed), cam, record
    )


def sample_object_params(object_type: str, rng: random.Random) -> dict:
    if object_type == "cube":
        size = float(rng.uniform(0.030, 0.075))
        left_rho = 200.0 * rng.uniform(1.0, 10.0)
        right_rho = 200.0 * rng.uniform(1.0, 10.0)
        return {"size": size, "left_rho": float(left_rho), "right_rho": float(right_rho), "left_scale": None, "right_scale": None}
    left_scale = float(rng.uniform(0.075, 0.105))
    right_scale = float(rng.uniform(0.075, 0.105))
    left_rho = 200.0 * rng.uniform(1.0, 10.0)
    right_rho = 200.0 * rng.uniform(1.0, 10.0)
    return {"size": None, "left_rho": float(left_rho), "right_rho": float(right_rho), "left_scale": left_scale, "right_scale": right_scale}


def run_single_trial(trial_idx, args, model, proj, heavy_vec, light_vec, params, run_root: Path):
    global SIM_STEP_COUNT
    global TRACE_RECORDER
    SIM_STEP_COUNT = 0

    trial_dir = run_root / f"trial_{trial_idx + 1:02d}"
    images_dir = trial_dir / "images" / "camera_0"
    csv_dir = trial_dir / "csv"
    images_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    TRACE_RECORDER["enabled"] = bool(args.save_trial_images)
    TRACE_RECORDER["images_dir"] = images_dir if args.save_trial_images else None
    TRACE_RECORDER["image_stride"] = int(args.trial_image_stride)

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
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

    left_xy = (0.62, -0.14)
    right_xy = (0.62, 0.14)
    tile_xy = (0.45, 0.38)

    if args.object_type == "cube":
        cube_size = params["size"]
        left_obj = scene.add_entity(
            material=gs.materials.Rigid(rho=params["left_rho"], friction=1.0),
            morph=gs.morphs.Box(size=(cube_size, cube_size, cube_size), pos=(left_xy[0], left_xy[1], cube_size * 0.5)),
            surface=gs.surfaces.Default(color=(0.3, 0.7, 0.9, 1.0)),
        )
        right_obj = scene.add_entity(
            material=gs.materials.Rigid(rho=params["right_rho"], friction=1.0),
            morph=gs.morphs.Box(size=(cube_size, cube_size, cube_size), pos=(right_xy[0], right_xy[1], cube_size * 0.5)),
            surface=gs.surfaces.Default(color=(0.9, 0.6, 0.3, 1.0)),
        )
    else:
        left_obj = scene.add_entity(
            material=gs.materials.Rigid(rho=params["left_rho"]),
            morph=gs.morphs.URDF(
                file="urdf/3763/mobility_vhacd.urdf",
                scale=params["left_scale"],
                pos=(left_xy[0], left_xy[1], 0.036 * (params["left_scale"] / 0.09)),
                euler=(0, 90, 0),
            ),
        )
        right_obj = scene.add_entity(
            material=gs.materials.Rigid(rho=params["right_rho"]),
            morph=gs.morphs.URDF(
                file="urdf/3763/mobility_vhacd.urdf",
                scale=params["right_scale"],
                pos=(right_xy[0], right_xy[1], 0.036 * (params["right_scale"] / 0.09)),
                euler=(0, 90, 0),
            ),
        )

    scene.add_entity(
        material=gs.materials.Rigid(rho=800, friction=1.0),
        morph=gs.morphs.Box(size=(0.16, 0.16, 0.004), pos=(tile_xy[0], tile_xy[1], 0.002), fixed=True),
        surface=gs.surfaces.Default(color=(0.1, 0.35, 0.9, 1.0)),
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
    home_q = ik_pose(franka, end_effector, pos=(0.45, 0.0, 0.28))
    home_q[0, -2:] = 0.04
    franka.set_dofs_position(home_q[0, :-2], motors_dof)
    franka.set_dofs_position(home_q[0, -2:], fingers_dof)

    for _ in range(120):
        control_pose(franka, motors_dof, fingers_dof, home_q, gripper_opening=0.04)
        sim_step(scene, cam=cam, record=False)

    left_force = pick_lift_hold_and_putback(
        scene, franka, end_effector, motors_dof, fingers_dof, hand_idx, left_finger_idx, left_obj, finger_len, left_xy, args.motion_speed, cam, False
    )
    right_force = pick_lift_hold_and_putback(
        scene, franka, end_effector, motors_dof, fingers_dof, hand_idx, left_finger_idx, right_obj, finger_len, right_xy, args.motion_speed, cam, False
    )

    left_emb = embed_force_segment(model, proj, left_force)
    right_emb = embed_force_segment(model, proj, right_force)
    left_heavy = float(torch.dot(left_emb, heavy_vec))
    left_light = float(torch.dot(left_emb, light_vec))
    right_heavy = float(torch.dot(right_emb, heavy_vec))
    right_light = float(torch.dot(right_emb, light_vec))
    predicted_heavy = "left" if left_heavy >= right_heavy else "right"
    gt_heavy = "left" if params["left_rho"] >= params["right_rho"] else "right"
    correct = bool(predicted_heavy == gt_heavy)

    if predicted_heavy == "left":
        pick_and_place_to_tile(
            scene, franka, end_effector, motors_dof, fingers_dof, hand_idx, left_finger_idx, left_obj, finger_len, left_xy, tile_xy, args.motion_speed, cam, False
        )
        placed_obj = left_obj
    else:
        pick_and_place_to_tile(
            scene, franka, end_effector, motors_dof, fingers_dof, hand_idx, left_finger_idx, right_obj, finger_len, right_xy, tile_xy, args.motion_speed, cam, False
        )
        placed_obj = right_obj

    move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (0.45, 0.0, 0.30), 0.04, scaled_steps(120, args.motion_speed), cam, False)
    for _ in range(40):
        sim_step(scene, cam=cam, record=False)

    placed_x, placed_y = get_obj_xy(placed_obj)
    place_dist = float(np.linalg.norm(np.array([placed_x - tile_xy[0], placed_y - tile_xy[1]], dtype=np.float32)))
    place_threshold = 0.10
    placed_at_target = bool(place_dist <= place_threshold)
    task_success = bool(correct and placed_at_target)

    save_matrix_csv(left_force, csv_dir / "left_force_seq.csv")
    save_matrix_csv(right_force, csv_dir / "right_force_seq.csv")
    score_row = {
        "trial_index": trial_idx + 1,
        "left_rho": float(params["left_rho"]),
        "right_rho": float(params["right_rho"]),
        "size": float(params["size"]) if params["size"] is not None else None,
        "left_scale": float(params["left_scale"]) if params["left_scale"] is not None else None,
        "right_scale": float(params["right_scale"]) if params["right_scale"] is not None else None,
        "left_heavy_score": left_heavy,
        "left_light_score": left_light,
        "right_heavy_score": right_heavy,
        "right_light_score": right_light,
        "predicted_heavy_side": predicted_heavy,
        "ground_truth_heavy_side": gt_heavy,
        "heavy_selection_correct": correct,
        "placed_at_target": placed_at_target,
        "place_distance_to_target_xy": place_dist,
        "place_distance_threshold": place_threshold,
        "task_success": task_success,
    }
    save_dict_rows_to_csv([score_row], csv_dir / "trial_scores.csv")

    return {
        "trial_index": trial_idx + 1,
        "trial_dir": str(trial_dir),
        "object_params": params,
        "scores": score_row,
    }


def summarize_trials(trials: list[dict]) -> dict:
    total = len(trials)
    select_correct = sum(1 for t in trials if t["scores"]["heavy_selection_correct"])
    place_success = sum(1 for t in trials if t["scores"]["placed_at_target"])
    overall_success = sum(1 for t in trials if t["scores"]["task_success"])
    return {
        "trials": total,
        "heavy_selection_accuracy": float(select_correct / max(1, total)),
        "place_success_rate": float(place_success / max(1, total)),
        "task_success_rate": float(overall_success / max(1, total)),
        "heavy_selection_correct_count": int(select_correct),
        "place_success_count": int(place_success),
        "task_success_count": int(overall_success),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "data" / "evaluation_outputs"))
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--object-type", choices=["cube", "bottle"], default="cube")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--motion-speed", type=float, default=1.8, help=">1.0 makes robot motions faster.")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-trial-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-trial camera snapshots.",
    )
    parser.add_argument("--trial-image-stride", type=int, default=20, help="Save one image every N sim steps.")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        gs.init(backend=gs.gpu, logging_level="warning")
    else:
        gs.init(backend=gs.cpu, logging_level="warning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, proj = load_contrastive_patchtst(Path(args.model_path), device)
    heavy_vec = load_text_embedding("heavy")
    light_vec = load_text_embedding("light")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = output_dir / f"place_heavier_patchtst_{args.object_type}_trials_{ts}"
    run_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    trials = []
    for trial_idx in range(args.trials):
        params = sample_object_params(args.object_type, rng)
        trial = run_single_trial(trial_idx, args, model, proj, heavy_vec, light_vec, params, run_root)
        trials.append(trial)
        print(
            f"[trial {trial_idx + 1:02d}/{args.trials}] "
            f"left_rho={params['left_rho']:.1f} right_rho={params['right_rho']:.1f} "
            f"pred={trial['scores']['predicted_heavy_side']} gt={trial['scores']['ground_truth_heavy_side']} "
            f"select_ok={trial['scores']['heavy_selection_correct']} "
            f"place_ok={trial['scores']['placed_at_target']} "
            f"task_ok={trial['scores']['task_success']}"
        )

    summary = summarize_trials(trials)
    payload = {
        "model_path": args.model_path,
        "object_type": args.object_type,
        "motion_speed": args.motion_speed,
        "trials": args.trials,
        "seed": args.seed,
        "results_root": str(run_root),
        "trial_results": trials,
        "summary": summary,
    }
    summary_path = run_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
