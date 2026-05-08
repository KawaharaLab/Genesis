import random
import shutil
import time
import json
from pathlib import Path
from multiprocessing import Process

import pandas as pd
import numpy as np
import torch
import genesis as gs
import matplotlib.pyplot as plt

import genesis.utils.geom as gu

import master_movement as mm
from make_step import (
    EarlyDropDetected,
    final_make_step,
    make_step,
    set_object_contact_targets,
    set_early_drop_monitor,
    set_intentional_release,
)


class EarlyDropSimulationError(RuntimeError):
    def __init__(self, message, paths, force_df, deform_df, segment_df, metadata):
        super().__init__(message)
        self.paths = paths
        self.force_df = force_df
        self.deform_df = deform_df
        self.segment_df = segment_df
        self.metadata = metadata


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

PHOTO_INTERVAL = 80
MATERIAL_TYPE = "Rigid"
TARGET_CHOICES = ["gentle","firm", "gentle_simple"]
GRASP_TO_LIFT_WAIT_STEPS = 100

MAX_PARALLEL_PROCESSES = 8
DATASET = "train_04272026"
DATASET_TYPE = "ycb" if "eval" in DATASET else "gso"  # Options: "ycb", "gso"
OUTPUT_VARIANT_LABEL = "vary"  # e.g. "heavy", "light", "v2"; set "" to disable
PLACE_TARGET_X_RANGE = (0.20, 0.70)
PLACE_TARGET_Y_RANGE = (0.20, 0.70)
MIN_PLACE_DISTANCE_FROM_SPAWN = 0.14
MIN_PLACE_DISTANCE_FROM_ROBOT = 0.42
MAX_PLACE_DISTANCE_FROM_ROBOT = 0.82
TARGET_TILE_SIZE = 0.20
TARGET_TILE_THICKNESS = 0.004
TARGET_TILE_COLOR = (0.10, 0.35, 0.90, 1.0)
BUMP_WALL_SIZE = (0.06, 0.06, 0.035)
BUMP_WALL_COLOR = (0.90, 0.25, 0.20, 1.0)
BUMP_PUSH_STEPS = 100
TOPPLE_THRESHOLD_DEG = 35.0


## -------------------------- PATH SETUP -------------------------- ##
def setup_paths(object_name: str, target_choice: str) -> dict:
    input_obj_path = DATA_ROOT / "objects" / DATASET_TYPE / object_name / "model.obj"
    if DATASET_TYPE == "hugging_face":
        input_obj_path = DATA_ROOT / "objects" / DATASET_TYPE / object_name / f"{object_name}.glb"
    if not input_obj_path.exists():
        raise FileNotFoundError(f"Input file not found at: {input_obj_path}")

    output_dir = DATA_ROOT / DATASET / "csv" / object_name / MATERIAL_TYPE / target_choice
    image_root = DATA_ROOT / DATASET / "images" / object_name / MATERIAL_TYPE / target_choice
    image_dirs = [image_root / "camera_0", image_root / "camera_wrist"]

    output_dir.mkdir(parents=True, exist_ok=True)
    for image_dir in image_dirs:
        image_dir.mkdir(parents=True, exist_ok=True)

    return {
        "input_obj": input_obj_path,
        "output_dir": output_dir,
        "images_dir": image_root,
        "force_data": output_dir / f"{object_name}_{MATERIAL_TYPE}_{target_choice}.csv",
        "deformation_data": output_dir / f"{object_name}_{MATERIAL_TYPE}_deform_{target_choice}.csv",
        "segmentation_data": output_dir / f"{object_name}_{MATERIAL_TYPE}_steps_{target_choice}.csv",
        "metadata_data": output_dir / f"{object_name}_{MATERIAL_TYPE}_metadata_{target_choice}.json",
        "object_name": object_name,
        "plot": output_dir / f"{object_name}_force_graph",
    }


def get_obj_bounding_box(obj_path):
    """Return the axis-aligned bounding box extents for OBJ or GLB meshes."""
    min_corner, max_corner = get_obj_bounds(obj_path)
    return (max_corner - min_corner).tolist()


def get_obj_bounds(obj_path):
    """Return the axis-aligned bounding box (min, max) for OBJ or GLB meshes."""
    obj_path = Path(obj_path)
    suffix = obj_path.suffix.lower()
    if suffix == ".obj":
        return _bounds_from_obj(obj_path)
    if suffix == ".glb":
        return _bounds_from_glb(obj_path)
    raise ValueError(f"Unsupported mesh format: {suffix}")


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


def _bounds_from_glb(obj_path: Path):
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("trimesh is required to compute bounding boxes for .glb files.") from exc

    mesh = trimesh.load(obj_path, force="scene")
    bounds = mesh.bounds
    if bounds is None:
        raise ValueError(f"Could not compute bounds for {obj_path}")
    min_corner, max_corner = bounds
    return np.asarray(min_corner, dtype=float), np.asarray(max_corner, dtype=float)


def _scale_to_vec(scale):
    """Return a length-3 numpy array regardless of scalar/tuple scale input."""
    if np.isscalar(scale):
        return np.array([float(scale), float(scale), float(scale)], dtype=float)
    arr = np.asarray(scale, dtype=float)
    if arr.size == 1:
        return np.repeat(arr.item(), 3)
    if arr.size != 3:
        raise ValueError(f"Scale must be scalar or length 3, got shape {arr.shape}")
    return arr


def set_grasp(obj_path):
    gripper_min_width, gripper_max_width = 0.002, 0.075
    bbox = get_obj_bounding_box(obj_path)
    scale = 1.0
    if gripper_min_width < bbox[0] < gripper_max_width:
        euler = (0, 0, 90)
    elif gripper_min_width < bbox[1] < gripper_max_width:
        euler = (0, 0, 0)
    else:
        scale = (0.080 / (bbox[0] + 0.01)) if bbox[0] < bbox[1] else (0.080 / (bbox[1] + 0.01))
        euler = (0, 0, 90) if bbox[0] < bbox[1] else (0, 0, 0)
    return scale, euler


## -------------------------- SIMULATION CORE -------------------------- ##
def sample_place_target(source_xy, robot_xy=(0.0, 0.0)):
    source_xy = np.asarray(source_xy, dtype=float)
    robot_xy = np.asarray(robot_xy, dtype=float)

    def _valid_target(candidate_xy):
        d_spawn = np.linalg.norm(candidate_xy - source_xy)
        d_robot = np.linalg.norm(candidate_xy - robot_xy)
        return (
            d_spawn >= MIN_PLACE_DISTANCE_FROM_SPAWN
            and MIN_PLACE_DISTANCE_FROM_ROBOT <= d_robot <= MAX_PLACE_DISTANCE_FROM_ROBOT
        )

    for _ in range(300):
        tx = random.uniform(*PLACE_TARGET_X_RANGE)
        ty = random.uniform(*PLACE_TARGET_Y_RANGE)
        candidate = np.array([tx, ty], dtype=float)
        if _valid_target(candidate):
            return float(tx), float(ty)

    # Fallback: sample from an annulus around the robot and clamp to valid workspace.
    for _ in range(300):
        theta = random.uniform(-np.pi, np.pi)
        radius = random.uniform(MIN_PLACE_DISTANCE_FROM_ROBOT, MAX_PLACE_DISTANCE_FROM_ROBOT)
        candidate = robot_xy + radius * np.array([np.cos(theta), np.sin(theta)], dtype=float)
        candidate[0] = np.clip(candidate[0], PLACE_TARGET_X_RANGE[0], PLACE_TARGET_X_RANGE[1])
        candidate[1] = np.clip(candidate[1], PLACE_TARGET_Y_RANGE[0], PLACE_TARGET_Y_RANGE[1])
        if _valid_target(candidate):
            return float(candidate[0]), float(candidate[1])

    # Final deterministic fallback near workspace center.
    fallback = np.array(
        [
            0.5 * (PLACE_TARGET_X_RANGE[0] + PLACE_TARGET_X_RANGE[1]),
            0.5 * (PLACE_TARGET_Y_RANGE[0] + PLACE_TARGET_Y_RANGE[1]),
        ],
        dtype=float,
    )
    return float(fallback[0]), float(fallback[1])


def _is_simple_mode(place_mode: str) -> bool:
    return place_mode in ("drop_simple", "gentle_simple")


def _plan_bump_wall(source_xy, target_xy):
    source = np.asarray(source_xy, dtype=float)
    target = np.asarray(target_xy, dtype=float)
    delta = target - source
    distance = float(np.linalg.norm(delta))
    if distance < 1e-6:
        direction = np.array([1.0, 0.0], dtype=float)
        distance = 1e-6
    else:
        direction = delta / distance

    min_along = 0.08
    max_along = max(min_along, distance - 0.06)
    along = float(np.clip(distance * 0.45, min_along, max_along))
    wall_xy = source + direction * along
    return {
        "center_xy": (float(wall_xy[0]), float(wall_xy[1])),
        "size_xyz": tuple(float(v) for v in BUMP_WALL_SIZE),
        "distance_from_source": along,
    }


def create_scene(obj_path: str, place_mode: str = "gentle"):
    if torch.cuda.is_available():
        gs.init(backend=gs.gpu)
    else:
        gs.init(backend=gs.cpu)

    object_scale, object_euler = set_grasp(obj_path)
    obj_min_corner, obj_max_corner = get_obj_bounds(obj_path)
    scale_vec = _scale_to_vec(object_scale)
    drop_center = np.array([0.45, 0.45], dtype=float)
    if Path(obj_path).suffix.lower() == ".glb":
        local_center = 0.5 * (obj_min_corner[:2] + obj_max_corner[:2])
        drop_center -= scale_vec[:2] * local_center

    spawn_z = 0.001 + max(0.0, -scale_vec[2] * obj_min_corner[2])
    object_spawn_pos = (float(drop_center[0]), float(drop_center[1]), float(spawn_z))
    color = (0, 255, 0)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1e-2),
        viewer_options=gs.options.ViewerOptions(camera_pos=(3, -1, 1.5), camera_lookat=(0, 0, 0), camera_fov=30),
        show_viewer=False,
    )

    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(coup_friction=0.0))
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        material=gs.materials.Rigid(coup_friction=3.0, friction=1.0),
    )

    cam = scene.add_camera(res=(1080, 1080), pos=(0.9, 0.9, 0.35), lookat=(0.0, 0.0, 0.35), fov=30)
    cam_wrist = scene.add_camera(
        model="thinlens",
        res=(1024, 1024),
        pos=(0.0, 0.0, 0.0),  # overwritten by attach below
        lookat=(0.0, 0.0, 0.0),
        fov=80.0,
        GUI=False,
    )
    # Mount wrist camera near the gripper tip with a small forward offset and downward pitch.
    roll = np.deg2rad(-10)   # keep camera upright relative to tool flange
    pitch = np.deg2rad(180)  # 初期位置のグリッパから根本への軸周り
    yaw = np.deg2rad(-90)    # 鉛直上向きの軸周り
    R_roll = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    R = R_yaw @ R_pitch @ R_roll
    trans = np.array([0.14, 0.0, -0.08])  # z: 鉛直下向き
    wrist_mount_link = franka.get_link("hand")
    cam_wrist.attach(wrist_mount_link, gu.trans_R_to_T(trans, R))

    rho = 200.0 * random.uniform(1.0, 10.0)
    gso_object = scene.add_entity(
        material=gs.materials.Rigid(rho=rho),
        morph=gs.morphs.Mesh(file=obj_path, scale=object_scale, pos=object_spawn_pos, euler=object_euler),
        surface=gs.surfaces.Default(color=color),
    )

    # Simple modes: object starts already on the tile and is placed/dropped back onto it.
    if _is_simple_mode(place_mode):
        target_xy = (float(drop_center[0]), float(drop_center[1]))
    else:
        target_xy = sample_place_target(drop_center, robot_xy=(0.0, 0.0))
    target_tile = scene.add_entity(
        material=gs.materials.Rigid(rho=800, friction=1.0, coup_friction=1.0),
        morph=gs.morphs.Box(
            size=(TARGET_TILE_SIZE, TARGET_TILE_SIZE, TARGET_TILE_THICKNESS),
            pos=(target_xy[0], target_xy[1], TARGET_TILE_THICKNESS * 0.5),
            fixed=True,
        ),
        surface=gs.surfaces.Default(color=TARGET_TILE_COLOR),
    )
    wall_plan = None
    obstacle_entity = None
    if place_mode == "bump":
        wall_plan = _plan_bump_wall(drop_center, target_xy)
        wall_x, wall_y = wall_plan["center_xy"]
        wall_sx, wall_sy, wall_sz = wall_plan["size_xyz"]
        obstacle_entity = scene.add_entity(
            material=gs.materials.Rigid(rho=1200, friction=1.2, coup_friction=1.2),
            morph=gs.morphs.Box(
                size=(wall_sx, wall_sy, wall_sz),
                pos=(wall_x, wall_y, wall_sz * 0.5),
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=BUMP_WALL_COLOR),
        )

    scene.build()
    return scene, cam, cam_wrist, franka, gso_object, target_xy, target_tile, obstacle_entity, wall_plan


def _settle_after_release(scene, cam, cam_wrist, franka, gso_object, df, deform_csv, paths, grip_force, steps=100):
    name = paths["object_name"]
    for _ in range(steps):
        make_step(
            scene=scene,
            cam=cam,
            franka=franka,
            df=df,
            deform_csv=deform_csv,
            photo_path=str(paths["images_dir"]),
            photo_interval=PHOTO_INTERVAL,
            gso_object=gso_object,
            name=name,
            gripper_force=grip_force,
            cam_wrist=cam_wrist,
        )


def _to_numpy(arr):
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr)


def _flatten_quat_wxyz(quat):
    quat = _to_numpy(quat).reshape(-1)
    if quat.size >= 4:
        return quat[:4].astype(float)
    raise ValueError(f"Unexpected quaternion shape: {quat.shape}")


def _angle_diff_deg(a, b):
    delta = a - b
    return (delta + 180.0) % 360.0 - 180.0


def _compute_max_xy_tilt_change_deg(quat_ref_wxyz, quat_now_wxyz):
    e_ref = _to_numpy(gs.utils.geom.quat_to_xyz(np.asarray(quat_ref_wxyz), rpy=True, degrees=True)).reshape(-1)
    e_now = _to_numpy(gs.utils.geom.quat_to_xyz(np.asarray(quat_now_wxyz), rpy=True, degrees=True)).reshape(-1)
    dx = abs(_angle_diff_deg(float(e_now[0]), float(e_ref[0])))
    dy = abs(_angle_diff_deg(float(e_now[1]), float(e_ref[1])))
    return max(dx, dy)


def _retreat_up_to_height(
    scene,
    cam,
    cam_wrist,
    franka,
    gso_object,
    df,
    deform_csv,
    paths,
    end_effector,
    motors_dof,
    fingers_dof,
    grip_force,
    target_z,
    fixed_x,
    fixed_y,
    steps=120,
):
    name = paths["object_name"]
    eef_pos = franka.get_links_pos([8]).tolist()[0]
    z_start = eef_pos[2]
    for step_idx in range(max(1, steps)):
        alpha = (step_idx + 1) / max(1, steps)
        z_target = (1.0 - alpha) * z_start + alpha * target_z
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=np.array([fixed_x, fixed_y, z_target]),
            quat=np.array([0, 1, 0, 0]),
        )
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([0.04, 0.04]), fingers_dof)
        make_step(
            scene=scene,
            cam=cam,
            franka=franka,
            df=df,
            deform_csv=deform_csv,
            photo_path=str(paths["images_dir"]),
            photo_interval=PHOTO_INTERVAL,
            gso_object=gso_object,
            name=name,
            gripper_force=grip_force,
            cam_wrist=cam_wrist,
        )


def _run_place_phase(
    scene,
    cam,
    cam_wrist,
    franka,
    gso_object,
    df,
    deform_csv,
    seg_df,
    paths,
    end_effector,
    place_x,
    place_y,
    place_mode,
    place_eef_z_nominal,
    retreat_target_z,
    motors_dof,
    fingers_dof,
    current_force,
    metadata,
):
    name = paths["object_name"]

    if place_mode in ("drop", "drop_simple"):
        seg_df.loc[len(seg_df)] = ["drop", int(scene.t)]
        release_z = place_eef_z_nominal + random.uniform(0.05, 0.08)
        mm.descend_to_place_cautiously(
            scene=scene,
            cam=cam,
            franka=franka,
            gso_object=gso_object,
            df=df,
            deform_csv=deform_csv,
            photo_path=str(paths["images_dir"]),
            photo_interval=PHOTO_INTERVAL,
            name=name,
            end_effector=end_effector,
            x=place_x,
            y=place_y,
            z=release_z,
            motors_dof=motors_dof,
            fingers_dof=fingers_dof,
            grip_force=-current_force,
            steps=120,
            cam_wrist=cam_wrist,
        )
        set_intentional_release(True)
        mm.release_object(
            scene=scene,
            cam=cam,
            franka=franka,
            gso_object=gso_object,
            df=df,
            deform_csv=deform_csv,
            photo_path=str(paths["images_dir"]),
            photo_interval=PHOTO_INTERVAL,
            name=name,
            fingers_dof=fingers_dof,
            grip_force=-current_force,
            steps=35,
            cam_wrist=cam_wrist,
        )
        quat_at_release = _flatten_quat_wxyz(gso_object.get_quat())
        _retreat_up_to_height(
            scene,
            cam,
            cam_wrist,
            franka,
            gso_object,
            df,
            deform_csv,
            paths,
            end_effector,
            motors_dof,
            fingers_dof,
            -current_force,
            retreat_target_z,
            place_x,
            place_y,
            steps=100,
        )
        _settle_after_release(scene, cam, cam_wrist, franka, gso_object, df, deform_csv, paths, -current_force, steps=180)
        quat_after_retreat = _flatten_quat_wxyz(gso_object.get_quat())
        tilt_change_deg = _compute_max_xy_tilt_change_deg(quat_at_release, quat_after_retreat)
        metadata["tilt_xy_deg"] = round(float(tilt_change_deg), 4)
        metadata["toppled"] = bool(tilt_change_deg >= TOPPLE_THRESHOLD_DEG)
        return

    if place_mode in ("gentle", "gentle_simple"):
        seg_df.loc[len(seg_df)] = ["gentle_place", int(scene.t)]
        mm.descend_to_place_cautiously(
            scene=scene,
            cam=cam,
            franka=franka,
            gso_object=gso_object,
            df=df,
            deform_csv=deform_csv,
            photo_path=str(paths["images_dir"]),
            photo_interval=PHOTO_INTERVAL,
            name=name,
            end_effector=end_effector,
            x=place_x,
            y=place_y,
            z=place_eef_z_nominal,
            motors_dof=motors_dof,
            fingers_dof=fingers_dof,
            grip_force=-current_force,
            steps=320,
            cam_wrist=cam_wrist,
        )
        for _ in range(20):
            make_step(
                scene=scene,
                cam=cam,
                franka=franka,
                df=df,
                deform_csv=deform_csv,
                photo_path=str(paths["images_dir"]),
                photo_interval=PHOTO_INTERVAL,
                gso_object=gso_object,
                name=name,
                gripper_force=-current_force,
                cam_wrist=cam_wrist,
            )
        set_intentional_release(True)
        mm.release_object(
            scene=scene,
            cam=cam,
            franka=franka,
            gso_object=gso_object,
            df=df,
            deform_csv=deform_csv,
            photo_path=str(paths["images_dir"]),
            photo_interval=PHOTO_INTERVAL,
            name=name,
            fingers_dof=fingers_dof,
            grip_force=-current_force,
            steps=60,
            cam_wrist=cam_wrist,
        )
        quat_at_release = _flatten_quat_wxyz(gso_object.get_quat())
        _retreat_up_to_height(
            scene,
            cam,
            cam_wrist,
            franka,
            gso_object,
            df,
            deform_csv,
            paths,
            end_effector,
            motors_dof,
            fingers_dof,
            -current_force,
            retreat_target_z,
            place_x,
            place_y,
            steps=120,
        )
        _settle_after_release(scene, cam, cam_wrist, franka, gso_object, df, deform_csv, paths, -current_force, steps=120)
        quat_after_retreat = _flatten_quat_wxyz(gso_object.get_quat())
        tilt_change_deg = _compute_max_xy_tilt_change_deg(quat_at_release, quat_after_retreat)
        metadata["tilt_xy_deg"] = round(float(tilt_change_deg), 4)
        metadata["toppled"] = bool(tilt_change_deg >= TOPPLE_THRESHOLD_DEG)
        return

    if place_mode in ("press", "firm"):
        seg_df.loc[len(seg_df)] = ["press_place", int(scene.t)]
        mm.descend_to_place_cautiously(
            scene=scene,
            cam=cam,
            franka=franka,
            gso_object=gso_object,
            df=df,
            deform_csv=deform_csv,
            photo_path=str(paths["images_dir"]),
            photo_interval=PHOTO_INTERVAL,
            name=name,
            end_effector=end_effector,
            x=place_x,
            y=place_y,
            z=place_eef_z_nominal,
            motors_dof=motors_dof,
            fingers_dof=fingers_dof,
            grip_force=-current_force,
            steps=260,
            cam_wrist=cam_wrist,
        )

        press_depth = 0.03
        press_steps = 120
        for step in range(press_steps):
            z_target = place_eef_z_nominal - press_depth * (step + 1) / press_steps
            qpos = franka.inverse_kinematics(link=end_effector, pos=np.array([place_x, place_y, z_target]), quat=np.array([0, 1, 0, 0]))
            franka.control_dofs_position(qpos[:-2], motors_dof)
            franka.control_dofs_position(np.array([0.0, 0.0]), fingers_dof)
            make_step(
                scene=scene,
                cam=cam,
                franka=franka,
                df=df,
                deform_csv=deform_csv,
                photo_path=str(paths["images_dir"]),
                photo_interval=PHOTO_INTERVAL,
                gso_object=gso_object,
                name=name,
                gripper_force=-current_force,
                cam_wrist=cam_wrist,
            )
            if len(df) > 0:
                left_fz = float(df.iloc[-1]["left_fz"])
                right_fz = float(df.iloc[-1]["right_fz"])
                if abs(left_fz) + abs(right_fz) > 30.0:
                    break

        for _ in range(25):
            make_step(
                scene=scene,
                cam=cam,
                franka=franka,
                df=df,
                deform_csv=deform_csv,
                photo_path=str(paths["images_dir"]),
                photo_interval=PHOTO_INTERVAL,
                gso_object=gso_object,
                name=name,
                gripper_force=-current_force,
                cam_wrist=cam_wrist,
            )
        set_intentional_release(True)
        mm.release_object(
            scene=scene,
            cam=cam,
            franka=franka,
            gso_object=gso_object,
            df=df,
            deform_csv=deform_csv,
            photo_path=str(paths["images_dir"]),
            photo_interval=PHOTO_INTERVAL,
            name=name,
            fingers_dof=fingers_dof,
            grip_force=-current_force,
            steps=45,
            cam_wrist=cam_wrist,
        )
        quat_at_release = _flatten_quat_wxyz(gso_object.get_quat())
        _retreat_up_to_height(
            scene,
            cam,
            cam_wrist,
            franka,
            gso_object,
            df,
            deform_csv,
            paths,
            end_effector,
            motors_dof,
            fingers_dof,
            -current_force,
            retreat_target_z,
            place_x,
            place_y,
            steps=120,
        )
        _settle_after_release(scene, cam, cam_wrist, franka, gso_object, df, deform_csv, paths, -current_force, steps=110)
        quat_after_retreat = _flatten_quat_wxyz(gso_object.get_quat())
        tilt_change_deg = _compute_max_xy_tilt_change_deg(quat_at_release, quat_after_retreat)
        metadata["tilt_xy_deg"] = round(float(tilt_change_deg), 4)
        metadata["toppled"] = bool(tilt_change_deg >= TOPPLE_THRESHOLD_DEG)
        return

    if place_mode == "bump":
        seg_df.loc[len(seg_df)] = ["bump_push", int(scene.t)]
        push_eef_z = franka.get_links_pos([8]).tolist()[0][2]
        for _ in range(BUMP_PUSH_STEPS):
            qpos = franka.inverse_kinematics(
                link=end_effector,
                pos=np.array([place_x, place_y, push_eef_z]),
                quat=np.array([0, 1, 0, 0]),
            )
            franka.control_dofs_position(qpos[:-2], motors_dof)
            franka.control_dofs_position(np.array([0.0, 0.0]), fingers_dof)
            make_step(
                scene=scene,
                cam=cam,
                franka=franka,
                df=df,
                deform_csv=deform_csv,
                photo_path=str(paths["images_dir"]),
                photo_interval=PHOTO_INTERVAL,
                gso_object=gso_object,
                name=name,
                gripper_force=-current_force,
                cam_wrist=cam_wrist,
            )

        set_intentional_release(True)
        mm.release_object(
            scene=scene,
            cam=cam,
            franka=franka,
            gso_object=gso_object,
            df=df,
            deform_csv=deform_csv,
            photo_path=str(paths["images_dir"]),
            photo_interval=PHOTO_INTERVAL,
            name=name,
            fingers_dof=fingers_dof,
            grip_force=-current_force,
            steps=45,
            cam_wrist=cam_wrist,
        )
        quat_at_release = _flatten_quat_wxyz(gso_object.get_quat())
        _retreat_up_to_height(
            scene,
            cam,
            cam_wrist,
            franka,
            gso_object,
            df,
            deform_csv,
            paths,
            end_effector,
            motors_dof,
            fingers_dof,
            -current_force,
            retreat_target_z,
            place_x,
            place_y,
            steps=110,
        )
        _settle_after_release(scene, cam, cam_wrist, franka, gso_object, df, deform_csv, paths, -current_force, steps=120)
        quat_after_retreat = _flatten_quat_wxyz(gso_object.get_quat())
        tilt_change_deg = _compute_max_xy_tilt_change_deg(quat_at_release, quat_after_retreat)
        metadata["tilt_xy_deg"] = round(float(tilt_change_deg), 4)
        metadata["toppled"] = bool(tilt_change_deg >= TOPPLE_THRESHOLD_DEG)
        return

    raise ValueError(f"Unsupported place mode: {place_mode}")


def run_rotation(
    scene,
    cam,
    cam_wrist,
    franka,
    gso_object,
    df,
    deform_csv,
    seg_df,
    paths,
    place_target_xy,
    place_mode,
    metadata,
    wall_plan=None,
):
    name = paths["object_name"]
    motors_dof, fingers_dof = np.arange(7), np.arange(7, 9)
    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 1000, 1000]))
    franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 100, 100]))
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -10, -10]),
        np.array([87, 87, 87, 87, 12, 12, 12, 10, 10]),
    )
    end_effector = franka.get_link("hand")

    offset = 0.074
    lower_obj_bound, upper_obj_bound = gso_object.get_AABB().cpu().numpy()
    object_height = float(upper_obj_bound[2] - lower_obj_bound[2])
    obj_center = 0.5 * (lower_obj_bound + upper_obj_bound)
    x, y = obj_center[0], obj_center[1]
    z = upper_obj_bound[2] + offset
    place_x, place_y = place_target_xy
    tile_top_z = TARGET_TILE_THICKNESS
    place_eef_z_nominal = tile_top_z + object_height + offset

    qpos = franka.inverse_kinematics(link=end_effector, pos=np.array([x, y, z + 0.1]), quat=np.array([0, 1, 0, 0]))
    qpos[-2:] = 0.04
    set_intentional_release(False)
    set_early_drop_monitor(False)

    metadata["target_xy"] = {"x": float(place_x), "y": float(place_y)}
    metadata["place_mode"] = str(place_mode)
    if wall_plan is not None:
        metadata["bump_wall"] = wall_plan
    seg_df.loc[len(seg_df)] = ["grasp", int(scene.t)]
    mm.set_to_pose(
        scene=scene,
        cam=cam,
        franka=franka,
        gso_object=gso_object,
        df=df,
        deform_csv=deform_csv,
        photo_path=str(paths["images_dir"]),
        photo_interval=PHOTO_INTERVAL,
        name=name,
        qpos=qpos,
        motors_dof=motors_dof,
        fingers_dof=fingers_dof,
        steps=20,
        cam_wrist=cam_wrist,
    )
    mm.descend_to_object(
        scene=scene,
        cam=cam,
        franka=franka,
        gso_object=gso_object,
        df=df,
        deform_csv=deform_csv,
        photo_path=str(paths["images_dir"]),
        photo_interval=PHOTO_INTERVAL,
        name=name,
        end_effector=end_effector,
        x=x,
        y=y,
        z=z,
        motors_dof=motors_dof,
        fingers_dof=fingers_dof,
        steps=30,
        cam_wrist=cam_wrist,
    )

    current_force = 10.0
    mm.grasp_object_position(
        scene=scene,
        cam=cam,
        franka=franka,
        gso_object=gso_object,
        df=df,
        deform_csv=deform_csv,
        photo_path=str(paths["images_dir"]),
        photo_interval=PHOTO_INTERVAL,
        name=name,
        end_effector=end_effector,
        x=x,
        y=y,
        z=z,
        motors_dof=motors_dof,
        fingers_dof=fingers_dof,
        grip_force=-current_force,
        steps=200,
        cam_wrist=cam_wrist,
    )
    set_early_drop_monitor(True)

    seg_df.loc[len(seg_df)] = ["hold", int(scene.t)]
    hold_qpos = franka.inverse_kinematics(link=end_effector, pos=np.array([x, y, z]), quat=np.array([0, 1, 0, 0]))
    for _ in range(GRASP_TO_LIFT_WAIT_STEPS):
        franka.control_dofs_position(hold_qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([0.0, 0.0]), fingers_dof)
        make_step(
            scene=scene,
            cam=cam,
            franka=franka,
            gso_object=gso_object,
            df=df,
            deform_csv=deform_csv,
            photo_path=str(paths["images_dir"]),
            photo_interval=PHOTO_INTERVAL,
            name=name,
            gripper_force=-current_force,
            cam_wrist=cam_wrist,
        )

    seg_df.loc[len(seg_df)] = ["lift", int(scene.t)]
    lift_steps = 200
    lift_step_size = 0.00075
    if place_mode == "bump":
        # Keep transport height lower so only the lower part of the object tends to hit the low wall.
        lift_steps = 120
        lift_step_size = 0.00035
    for i in range(lift_steps):
        curr_z = z + (i * lift_step_size)
        if not mm.lift_object(
            scene=scene,
            cam=cam,
            franka=franka,
            gso_object=gso_object,
            df=df,
            deform_csv=deform_csv,
            photo_path=str(paths["images_dir"]),
            photo_interval=PHOTO_INTERVAL,
            name=name,
            end_effector=end_effector,
            x=x,
            y=y,
            z=curr_z,
            motors_dof=motors_dof,
            fingers_dof=fingers_dof,
            grip_force=-current_force,
            steps=1,
            cam_wrist=cam_wrist,
        ):
            break

    seg_df.loc[len(seg_df)] = ["move_to_target", int(scene.t)]
    if _is_simple_mode(place_mode):
        # Keep the object above the same tile: no long XY transport.
        hold_qpos = franka.inverse_kinematics(link=end_effector, pos=np.array([x, y, z + 0.12]), quat=np.array([0, 1, 0, 0]))
        for _ in range(80):
            franka.control_dofs_position(hold_qpos[:-2], motors_dof)
            franka.control_dofs_position(np.array([0.0, 0.0]), fingers_dof)
            make_step(
                scene=scene,
                cam=cam,
                franka=franka,
                gso_object=gso_object,
                df=df,
                deform_csv=deform_csv,
                photo_path=str(paths["images_dir"]),
                photo_interval=PHOTO_INTERVAL,
                name=name,
                gripper_force=-current_force,
                cam_wrist=cam_wrist,
            )
        place_x, place_y = x, y
    else:
        mm.move_to_place_xy(
            scene=scene,
            cam=cam,
            franka=franka,
            gso_object=gso_object,
            df=df,
            deform_csv=deform_csv,
            photo_path=str(paths["images_dir"]),
            photo_interval=PHOTO_INTERVAL,
            name=name,
            end_effector=end_effector,
            x=place_x,
            y=place_y,
            motors_dof=motors_dof,
            fingers_dof=fingers_dof,
            grip_force=-current_force,
            steps=220 if place_mode == "bump" else 300,
            cam_wrist=cam_wrist,
        )
    retreat_target_z = franka.get_links_pos([8]).tolist()[0][2]

    seg_df.loc[len(seg_df)] = ["place", int(scene.t)]
    _run_place_phase(
        scene,
        cam,
        cam_wrist,
        franka,
        gso_object,
        df,
        deform_csv,
        seg_df,
        paths,
        end_effector,
        place_x,
        place_y,
        place_mode,
        place_eef_z_nominal,
        retreat_target_z,
        motors_dof,
        fingers_dof,
        current_force,
        metadata,
    )
    set_early_drop_monitor(False)

    final_make_step(
        scene=scene,
        cam=cam,
        franka=franka,
        df=df,
        deform_csv=deform_csv,
        photo_path=str(paths["images_dir"]),
        photo_interval=PHOTO_INTERVAL,
        gso_object=gso_object,
        name=name,
        cam_wrist=cam_wrist,
    )


## -------------------------- GENERATE PLOTS -------------------------- ##
def generate_plots(df, paths):
    """Generates and saves the plots for the simulation results."""
    name = paths["object_name"]
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    axs[0].axis("off")

    force_columns = ["left_fx", "left_fy", "left_fz", "right_fx", "right_fy", "right_fz"]
    for col in force_columns:
        axs[1].plot(df["step"], df[col], marker=".", label=col)

    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("Force (N)")
    axs[1].set_title("Force Components Over Time")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(paths["plot"], dpi=300, bbox_inches="tight")
    print(f"Saved plot -> {paths['plot']}")
    plt.close(fig)


def save_results(force_df, deform_df, segment_df, metadata, paths):
    print(f"💾 Saving results to {paths['output_dir']}")
    force_df.to_csv(paths["force_data"], index=False)
    deform_df.to_csv(paths["deformation_data"], index=False)
    segment_df.to_csv(paths["segmentation_data"], index=False)
    with open(paths["metadata_data"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, sort_keys=True)
    generate_plots(df=force_df, paths=paths)


def move_images_to_target_paths(source_paths, target_paths):
    src = Path(source_paths["images_dir"])
    dst = Path(target_paths["images_dir"])
    if not src.exists():
        return
    if dst.exists():
        raise FileExistsError(f"Refusing to overwrite existing image directory: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def _accidental_choice_exists(object_name: str, choice_name: str) -> bool:
    csv_dir = DATA_ROOT / DATASET / "csv" / object_name / MATERIAL_TYPE / choice_name
    img_dir = DATA_ROOT / DATASET / "images" / object_name / MATERIAL_TYPE / choice_name
    return csv_dir.exists() or img_dir.exists()


def _append_output_variant(name: str) -> str:
    label = OUTPUT_VARIANT_LABEL.strip()
    if not label:
        return name
    return f"{name}_{label}"


def _resolve_accidental_choice(object_name: str, target_choice: str, attempt_idx: int) -> str:
    # Preferred names: accidental_drop_1_{target}, accidental_drop_2_{target}
    base = _append_output_variant(f"accidental_drop_{attempt_idx}_{target_choice}")
    if not _accidental_choice_exists(object_name, base):
        return base

    # If the preferred name already exists (e.g. previous run), avoid overwrite.
    suffix = 2
    while True:
        candidate = f"{base}_run{suffix}"
        if not _accidental_choice_exists(object_name, candidate):
            return candidate
        suffix += 1


def _choice_exists(object_name: str, choice_name: str) -> bool:
    csv_dir = DATA_ROOT / DATASET / "csv" / object_name / MATERIAL_TYPE / choice_name
    img_dir = DATA_ROOT / DATASET / "images" / object_name / MATERIAL_TYPE / choice_name
    return csv_dir.exists() or img_dir.exists()


def _resolve_final_choice(object_name: str, target_choice: str) -> str:
    """
    Returns a safe destination choice for successful runs.
    If target_choice already exists from a previous run, append _runN.
    """
    base = _append_output_variant(target_choice)
    if not _choice_exists(object_name, base):
        return base

    suffix = 2
    while True:
        candidate = f"{base}_run{suffix}"
        if not _choice_exists(object_name, candidate):
            return candidate
        suffix += 1


def _resolve_attempt_choice(object_name: str, target_choice: str, attempt_idx: int) -> str:
    """
    Returns an isolated working directory name for each attempt to avoid image overwrite.
    """
    base = f"_attempt_{attempt_idx}_{target_choice}"
    if not _choice_exists(object_name, base):
        return base

    suffix = 2
    while True:
        candidate = f"{base}_run{suffix}"
        if not _choice_exists(object_name, candidate):
            return candidate
        suffix += 1


def _rename_attempt_outputs(object_name: str, source_choice: str, target_choice: str):
    csv_src = DATA_ROOT / DATASET / "csv" / object_name / MATERIAL_TYPE / source_choice
    img_src = DATA_ROOT / DATASET / "images" / object_name / MATERIAL_TYPE / source_choice
    csv_dst = DATA_ROOT / DATASET / "csv" / object_name / MATERIAL_TYPE / target_choice
    img_dst = DATA_ROOT / DATASET / "images" / object_name / MATERIAL_TYPE / target_choice

    if csv_dst.exists() or img_dst.exists():
        raise FileExistsError(f"Destination already exists: csv={csv_dst}, images={img_dst}")

    if csv_src.exists():
        csv_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(csv_src), str(csv_dst))
        # Rename files to match the new deformation name for downstream scripts.
        for p in csv_dst.iterdir():
            if source_choice in p.name:
                p.rename(p.with_name(p.name.replace(source_choice, target_choice, 1)))

    if img_src.exists():
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(img_src), str(img_dst))


def run_single_simulation(object_name: str, target_choice: str, output_choice: str | None = None):
    output_choice = target_choice if output_choice is None else output_choice
    paths = setup_paths(object_name, output_choice)
    force_df = pd.DataFrame(
        columns=[
            "step",
            "left_fx",
            "left_fy",
            "left_fz",
            "left_tx",
            "left_ty",
            "left_tz",
            "right_fx",
            "right_fy",
            "right_fz",
            "right_tx",
            "right_ty",
            "right_tz",
            "dof_0",
            "dof_1",
            "dof_2",
            "dof_3",
            "dof_4",
            "dof_5",
            "dof_6",
            "dof_7",
            "dof_8",
            "eef_x",
            "eef_y",
            "eef_z",
            "left_finger_x",
            "left_finger_y",
            "left_finger_z",
            "right_finger_x",
            "right_finger_y",
            "right_finger_z",
            "control_left_finger",
            "control_right_finger",
            "obj_COM_x",
            "obj_COM_y",
            "obj_COM_z",
            "obj_mass",
            "obj_min_x",
            "obj_min_y",
            "obj_min_z",
            "obj_max_x",
            "obj_max_y",
            "obj_max_z",
            "obj_left_finger",
            "obj_right_finger",
            "obj_plane",
            "obj_target_tile",
            "obj_obstacle",
        ]
    )
    deform_df = pd.DataFrame(columns=["step", "deformations", "grip_force"])
    segment_df = pd.DataFrame(columns=["action", "step"])
    metadata = {
        "object_name": object_name,
        "material_type": MATERIAL_TYPE,
        "target_choice": target_choice,
        "output_choice": output_choice,
        "topple_threshold_deg": float(TOPPLE_THRESHOLD_DEG),
        "target_xy": None,
        "place_mode": None,
        "tilt_xy_deg": None,
        "toppled": None,
    }

    scene = None
    cam = cam_wrist = franka = gso_object = None
    wall_plan = None
    try:
        scene, cam, cam_wrist, franka, gso_object, place_target_xy, target_tile, obstacle_entity, wall_plan = create_scene(
            str(paths["input_obj"]), target_choice
        )
        set_object_contact_targets(target_tile=target_tile, obstacle=obstacle_entity)
        run_rotation(
            scene,
            cam,
            cam_wrist,
            franka,
            gso_object,
            force_df,
            deform_df,
            segment_df,
            paths,
            place_target_xy,
            target_choice,
            metadata,
            wall_plan,
        )
    except EarlyDropDetected as exc:
        segment_df.loc[len(segment_df)] = [f"early_drop_{target_choice}", int(scene.t)]
        metadata["terminated_early_drop"] = True
        raise EarlyDropSimulationError(str(exc), paths, force_df, deform_df, segment_df, metadata) from exc
    finally:
        # Ensure each attempt fully tears down Genesis before any retry.
        set_object_contact_targets(target_tile=None, obstacle=None)
        gs.destroy()
    metadata["terminated_early_drop"] = False
    return paths, force_df, deform_df, segment_df, metadata


def main(object_name: str, target_choice: str = "gentle"):
    print(f"🚀 Starting simulation for '{object_name}' with target '{target_choice}'...")
    input_obj_path = DATA_ROOT / "objects" / DATASET_TYPE / object_name / "model.obj"
    if DATASET_TYPE == "hugging_face":
        input_obj_path = DATA_ROOT / "objects" / DATASET_TYPE / object_name / f"{object_name}.glb"
    if not input_obj_path.exists():
        print(f"❌ Aborting: Input file not found at: {input_obj_path}")
        return

    last_failed = None
    for attempt in range(2):
        attempt_idx = attempt + 1
        attempt_choice = _resolve_attempt_choice(object_name, target_choice, attempt_idx)
        try:
            paths, force_df, deform_df, segment_df, metadata = run_single_simulation(
                object_name, target_choice, output_choice=attempt_choice
            )
            final_choice = _resolve_final_choice(object_name, target_choice)
            metadata["target_choice"] = final_choice
            metadata["source_target_choice"] = target_choice
            metadata["attempt_choice"] = attempt_choice
            save_results(force_df, deform_df, segment_df, metadata, paths)
            _rename_attempt_outputs(object_name, attempt_choice, final_choice)
            if final_choice != target_choice:
                print(
                    f"ℹ️ Destination '{target_choice}' already exists. "
                    f"Saved to '{final_choice}' instead."
                )
            print(f"✅ Finished simulation for '{object_name}'.")
            return
        except EarlyDropSimulationError as exc:
            print(f"⚠️ Early drop detected for '{object_name}' / '{target_choice}' on attempt {attempt + 1}: {exc}")
            early_choice = _resolve_accidental_choice(object_name, target_choice, attempt + 1)
            early_metadata = dict(exc.metadata)
            early_metadata["target_choice"] = early_choice
            early_metadata["source_target_choice"] = target_choice
            early_metadata["attempt_choice"] = attempt_choice
            # 1) Save all outputs into the current attempt directory.
            save_results(exc.force_df, exc.deform_df, exc.segment_df, early_metadata, exc.paths)
            # 2) Rename the whole attempt directory to accidental_drop_*.
            _rename_attempt_outputs(object_name, attempt_choice, early_choice)
            print(f"⚠️ Saved early-drop result to '{early_choice}'.")
            last_failed = (early_choice, exc.force_df, exc.deform_df, exc.segment_df, early_metadata)
            if attempt == 0:
                print(f"🔁 Retrying '{object_name}' / '{target_choice}' once after early drop.")
                continue
            print(f"❌ Early drop persisted after retry for '{object_name}' / '{target_choice}'.")
            return

    if last_failed is not None:
        print(f"❌ Finished with early-drop result for '{object_name}' / '{target_choice}'.")


def get_tasks_to_run():
    if 0: # DO NOT CHANGE
        return [("001_chips_can", "bump")]
        # return [("001_chips_can", "drop"), ("001_chips_can", "gentle"), ("001_chips_can", "firm"), ("002_master_chef_can", "drop"), ("002_master_chef_can", "gentle"), ("002_master_chef_can", "firm"), ("003_cracker_box", "drop"), ("003_cracker_box", "gentle"), ("003_cracker_box", "firm"), ("004_sugar_box", "drop"), ("004_sugar_box", "gentle"), ("004_sugar_box", "firm"), ("005_tomato_soup_can", "drop"), ("005_tomato_soup_can", "gentle"), ("005_tomato_soup_can", "firm")]

    tasks = []
    objects_dir = DATA_ROOT / "objects" / DATASET_TYPE
    if not objects_dir.exists():
        print(f"❌ Error: Input directory '{objects_dir}' not found.")
        return []

    object_names = [d.name for d in objects_dir.iterdir() if d.is_dir()]
    print(f"🔍 Found {len(object_names)} objects in '{objects_dir}'.")
    for name in object_names:
        for target in TARGET_CHOICES:
            print(f"  - Queueing '{name}' with target '{target}' (no previous runs).")
            tasks.append((name, target))

    return tasks


if __name__ == "__main__":
    tasks_to_run = get_tasks_to_run()
    if not tasks_to_run:
        print("🎉 No new simulations to run.")
        exit()

    print(f"\nFound {len(tasks_to_run)} simulation task(s) to run.")
    processes = []
    for object_name, target_choice in tasks_to_run:
        while len(processes) >= MAX_PARALLEL_PROCESSES:
            processes = [p for p in processes if p.is_alive()]
            time.sleep(1)
        p = Process(target=main, args=(object_name, target_choice))
        p.start()
        processes.append(p)
        time.sleep(5)

    for p in processes:
        p.join()
    print("\n\nAll simulations completed.")
