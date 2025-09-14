import math
import random
import sys
import time
from pathlib import Path
from datetime import datetime
from multiprocessing import Process
import os

import pandas as pd
import numpy as np
import torch
# Ensure local 'genesis' package from repo is importable before any site-packages 'genesis'
_repo_root = str(Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
import genesis as gs
import matplotlib.pyplot as plt
# Assuming these are your custom modules within the src/ directory
import master_movement as mm
from make_step import make_step, final_make_step


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "hugging_face_object_data"

PHOTO_INTERVAL = 10
MATERIAL_TYPE = "Rigid" # Rigid or Elastic

if MATERIAL_TYPE == "Elastic":
    TARGET_CHOICES = ['soft', 'medium', 'hard']
else:
    TARGET_CHOICES = ['none']

MAX_PARALLEL_PROCESSES = 8
# DATA_TYPE = os.environ['DATA_TYPE']
DATA_TYPE = "eval_tmp"  # Options: "raw", "strong", "medium", "eval", "eval_medium", "eval_strong"

RUNNING_DROP_IN_BOX = False
## -------------------------- PATH SETUP -------------------------- ##

def setup_paths(object_name: str, target_choice: str) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    simulation_id = f"{target_choice}_{MATERIAL_TYPE.lower()}_{timestamp}"
    if "eval" in DATA_TYPE:
        #input_obj_path = DATA_ROOT / "objects" / object_name / "model.obj"
        input_obj_path = DATA_ROOT / "all_files" / object_name / "model.obj"

    else:
        input_obj_path = DATA_ROOT / "all_files" / object_name / "model.obj"
        #input_obj_path = DATA_ROOT / "objects/gso" / object_name / "model.obj"

    if not input_obj_path.exists():
        raise FileNotFoundError(f"Input file not found at: {input_obj_path}")

    output_dir = DATA_ROOT / DATA_TYPE / "csv" / object_name / MATERIAL_TYPE / target_choice
    image_root = DATA_ROOT / DATA_TYPE / "images" / object_name / MATERIAL_TYPE / target_choice
    image_dirs = [image_root / f"camera_{i}" for i in range(3)]

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
        "object_name": object_name,
        "plot": output_dir / f"{object_name}_force_graph"
    }


def get_obj_bounding_box(obj_path):
    min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
    max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')
    with open(obj_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                _, x, y, z = line.strip().split()
                x, y, z = float(x), float(y), float(z)
                min_x, max_x = min(min_x, x), max(max_x, x)
                min_y, max_y = min(min_y, y), max(max_y, y)
                min_z, max_z = min(min_z, z), max(max_z, z)
    return [max_x - min_x, max_y - min_y, max_z - min_z]


# -------------------------- ORIENTATION + SIZE HELPERS -------------------------- #
def _first_three_floats_after_v(tokens):
    """Return the first three numeric values after the 'v' label; ignore extras and comments."""
    coords = []
    for t in tokens[1:]:
        if t.startswith('#'):
            break
        try:
            coords.append(float(t))
        except ValueError:
            # Non-numeric token (e.g., stray text); skip it
            continue
        if len(coords) == 3:
            break
    return coords if len(coords) == 3 else None

def _read_vertices(obj_path):
    """Read vertices as Nx3 float64, ignoring optional w/rgb/comment tokens; drop non-finite rows."""
    vs = []
    with open(obj_path, 'r') as f:
        for line in f:
            # Match only position vertices; 'vn'/'vt' won't pass 'v ' with exact space after v
            if line.startswith('v '):
                tokens = line.strip().split()
                triplet = _first_three_floats_after_v(tokens)
                if triplet is not None:
                    vs.append(triplet)
    if not vs:
        return np.empty((0, 3), dtype=np.float64)
    V = np.array(vs, dtype=np.float64)
    # Drop any rows with NaN/Inf just in case
    if V.size:
        mask = np.isfinite(V).all(axis=1)
        V = V[mask]
    return V

def get_live_center_and_top(gso_object, mat_type: str):
    """Return (cx, cy, top_z) from the object's current state."""
    if mat_type == 'Elastic':
        P = gso_object.get_state().pos.detach().cpu().numpy()[0]
        lower = P.min(axis=0)
        upper = P.max(axis=0)
    else:  # Rigid
        lower, upper = gso_object.get_AABB().cpu().numpy()
    cx = 0.5 * (lower[0] + upper[0])
    cy = 0.5 * (lower[1] + upper[1])
    top_z = upper[2]
    return cx, cy, top_z

def min_z_after_euler(obj_path, euler_deg):
    """Min Z of the mesh after applying euler_deg (degrees)."""
    V = _read_vertices(obj_path)
    if V.size == 0:
        return 0.0
    R = euler_deg_to_matrix(*euler_deg)
    Vr = V @ R.T
    m = np.nanmin(Vr[:, 2])
    return float(m) if np.isfinite(m) else 0.0

def detect_down_axis(obj_path):
    """
    Heuristic: the 'down' axis tends to have many vertices near its minimum.
    Returns 0,1,2 for X,Y,Z being 'down' in file-space.
    """
    V = _read_vertices(obj_path)
    if V.size == 0:
        return 2
    mins = V.min(axis=0)
    spans = V.max(axis=0) - mins
    tol = 1e-3 * float(np.linalg.norm(spans))  # 0.1% of diag
    counts = [(V[:, i] <= mins[i] + tol).sum() for i in range(3)]
    return int(np.argmax(counts))

def canonical_euler_to_Z_up(down_axis):
    """
    Rotation (rx, ry, rz) deg that maps file-space 'down' to world -Z.
    - if down is Z: (0,0,0)
    - if down is Y: (-90,0,0)    (Y->Z)
    - if down is X: (0,90,0)     (X->Z)
    """
    if down_axis == 2:
        return (0.0, 0.0, 0.0)
    elif down_axis == 1:
        return (-90.0, 0.0, 0.0)
    else:
        return (0.0, 90.0, 0.0)

def euler_deg_to_matrix(rx_deg, ry_deg, rz_deg):
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0],[0, cx, -sx],[0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy],[0, 1, 0],[-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0],[sz,  cz, 0],[0,   0,  1]], dtype=np.float64)
    # XYZ intrinsic; for multiples of 90° this is robust
    return Rz @ Ry @ Rx

def yaw_from_xy_pca(obj_path, canon_euler, strategy="min_width_grasp"):
    """
    PCA on XY after applying canon_euler. Returns yaw (deg) so that:
      - strategy == "min_width_grasp": aligns MINOR axis with world X (smallest grasp width)
      - strategy == "face_long_edge":  aligns MAJOR axis with world X (long edge 'faces' robot)
    """
    V = _read_vertices(obj_path)
    if V.size == 0:
        return 0.0
    R = euler_deg_to_matrix(*canon_euler)
    Vr = V @ R.T
    XY = Vr[:, :2]
    XY = XY - XY.mean(axis=0)
    # Covariance and eigen-decomposition (eigh -> ascending eigenvalues)
    n = max(len(XY) - 1, 1)
    C = (XY.T @ XY) / n
    w, U = np.linalg.eigh(C)
    if not np.isfinite(w).all():
        return 0.0
    order = np.argsort(w)          # 0: minor, 1: major (in 2D)
    minor = U[:, order[0]]         # shape (2,)
    major = U[:, order[-1]]        # shape (2,)
    vec = minor if strategy == "min_width_grasp" else major
    yaw = -np.degrees(np.arctan2(vec[1], vec[0]))  # rotate so chosen axis aligns with +X
    return float(yaw)

def rotated_extents_after_euler(obj_path, euler_deg):
    V = _read_vertices(obj_path)
    if V.size == 0:
        return (0.0, 0.0, 0.0)
    R = euler_deg_to_matrix(*euler_deg)
    Vr = V @ R.T
    mins = Vr.min(axis=0)
    maxs = Vr.max(axis=0)
    ext = maxs - mins
    return tuple(map(float, ext))

# ---- Spawn height helpers (Z-up by default) ----
UP_AXIS = 2        # 2 => Z-up sims; use 1 if your sim is Y-up
GROUND_Z = 0.0
CLEARANCE = 0.005  # 5 mm of slack

def _scale_component(scale, axis_idx=UP_AXIS):
    """Return the scale along the up-axis (handles scalar or 3-tuple)."""
    if isinstance(scale, (tuple, list, np.ndarray)):
        if len(scale) > axis_idx:
            return float(scale[axis_idx])
        return float(scale[0])
    return float(scale)

def spawn_height_after_canon(obj_path, scale, down_axis, ground=GROUND_Z, clearance=CLEARANCE):
    """
    After canonicalization, world Z-height equals the file-space extent along 'down_axis'
    (which we rotate into Z). Use that to spawn above ground.
    """
    dx, dy, dz = get_obj_bounding_box(obj_path)
    extents = (dx, dy, dz)
    height = extents[down_axis]
    sz = _scale_component(scale, axis_idx=UP_AXIS)
    half_h = 0.5 * height * sz
    return ground + half_h + clearance

def set_grasp(obj_path,
              canon_euler=(0.0, 0.0, 0.0),
              GRIPPER_MIN_WIDTH=0.002,
              GRIPPER_MAX_WIDTH=0.075,
              TARGET_WIDTH=0.080,        # desired grasp width in XY (m)
              MAX_LONGEST_DIM=0.12,      # global size cap (m)
              yaw_strategy="face_long_edge"):  # or "face_long_edge"
    """
    Choose yaw from 2D PCA and scale uniformly so the post-rotation XY width
    matches TARGET_WIDTH (capped by MAX_LONGEST_DIM). Final clamp ensures width
    <= GRIPPER_MAX_WIDTH without shrinking otherwise-good objects.
    """
    # 1) Pick yaw from PCA on XY after canonicalization
    yaw = yaw_from_xy_pca(obj_path, canon_euler, strategy=yaw_strategy)
    grasp_euler = (0.0, 0.0, yaw)

    # 2) Extents AFTER applying (canon + yaw)
    total_euler = (canon_euler[0] + grasp_euler[0],
                   canon_euler[1] + grasp_euler[1],
                   canon_euler[2] + grasp_euler[2])
    ex, ey, ez = rotated_extents_after_euler(obj_path, total_euler)

    # 3) Width to regulate and safety cap
    #    For min-width grasping, width is the *smaller* of (ex, ey).
    #    For “face_long_edge”, width is the *larger* (you’re presenting the long side).
    width_xy = min(ex, ey) if yaw_strategy == "min_width_grasp" else max(ex, ey)
    s_width  = (TARGET_WIDTH / width_xy) if width_xy > 0 else 1.0
    longest  = max(ex, ey, ez)
    s_cap    = (MAX_LONGEST_DIM / longest) if longest > 0 else 1.0

    scale = float(np.clip(min(s_width, s_cap), 1e-3, 1e3))

    # 4) Final gripper-width clamp: only shrink offenders; leave others unchanged
    post_width = width_xy * scale
    max_width  = GRIPPER_MAX_WIDTH * 0.97  # tiny safety margin
    if post_width > max_width and post_width > 0:
        scale *= (max_width / post_width)

    return scale, grasp_euler


def adjust_force_with_pd_control(current_force, deform_csv, target_vel):
    if len(deform_csv) < 2: return current_force
    deform_velocity = deform_csv.iloc[-1, 1] - deform_csv.iloc[-2, 1]
    if deform_velocity > 1.2 * target_vel:
        current_force -= 0.1
    elif deform_velocity < 0.8 * target_vel:
        current_force += 0.1
    return max(0.1, current_force)


## -------------------------- SIMULATION CORE -------------------------- ##
def sample_drop_box_bounds(cx_range=(-0.55, 0.75),cy_range=(0.35, 0.85),inner_w=0.25,inner_h=0.25):
    cx = random.uniform(*cx_range)
    cy = random.uniform(*cy_range)
    x_min, x_max = cx - inner_w/2.0, cx + inner_w/2.0
    y_min, y_max = cy - inner_h/2.0, cy + inner_h/2.0
    return (x_min, x_max, y_min, y_max), (cx, cy)

def create_scene(obj_path: str):
    global RUNNING_DROP_IN_BOX
    gs.init(backend=gs.gpu if torch.cuda.is_available() else gs.cpu)

    # 1) Canonicalize to Z-up based on file-space 'down' axis
    down_axis   = detect_down_axis(obj_path)
    canon_euler = canonical_euler_to_Z_up(down_axis)

    # 2) Choose yaw & scale AFTER canonicalization (consistent XY width)
    object_scale, grasp_euler = set_grasp(obj_path, canon_euler=canon_euler)

    # 3) Compose rotations: canonicalization then grasp yaw
    object_euler = (canon_euler[0] + grasp_euler[0],
                    canon_euler[1] + grasp_euler[1],
                    canon_euler[2] + grasp_euler[2])

    # 4) Compute a spawn height that clears the ground by AABB/2 + clearance
    x0, y0 = 0.45, 0.45
    minz_local = min_z_after_euler(obj_path, object_euler)  # after canon + yaw
    spawn_z = GROUND_Z - (minz_local * object_scale) + CLEARANCE
    color = random.choice([(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255)])

    if MATERIAL_TYPE == 'Elastic':
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=1e-3, substeps=10),
            viewer_options=gs.options.ViewerOptions(camera_pos=(3,-1,1.5), camera_lookat=(0,0,0), camera_fov=30),
            show_viewer=False,
            mpm_options=gs.options.MPMOptions(lower_bound=(0,-0.1,-0.05), upper_bound=(0.75,1,1), grid_density=128)
        )
    else:
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=1e-2, substeps=5),
            viewer_options=gs.options.ViewerOptions(camera_pos=(3,-1,1.5), camera_lookat=(0,0,0), camera_fov=30),
            show_viewer=False,
        )
    cam = scene.add_camera(res=(1280, 720), pos=(-1.5, 1.5, 0.25), lookat=(0.45, 0.45, 0.4), fov=30)
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(coup_friction=0.0))
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"), material=gs.materials.Rigid(coup_friction=3.0, friction=1.0))

    if MATERIAL_TYPE == 'Elastic':
        gso_object = scene.add_entity(
            material=gs.materials.MPM.Elastic(),
            morph=gs.morphs.Mesh(file=obj_path, scale=object_scale, pos=(x0, y0, spawn_z), euler=object_euler),
            surface=gs.surfaces.Default(color=color)
        )
    elif MATERIAL_TYPE == 'Rigid':
        gso_object = scene.add_entity(
            material=gs.materials.Rigid(),
            morph=gs.morphs.Mesh(file=obj_path, scale=object_scale, pos=(x0, y0, spawn_z), euler=object_euler),
            surface=gs.surfaces.Default(color=color)
        )

    # ---------- HARD-CODED FLOOR ZONE MARKER ----------
    if RUNNING_DROP_IN_BOX == True:
        drop_box_bounds, (cx, cy) = sample_drop_box_bounds()
        print("HOLY", cx,cy)
        box_mesh_path = "/Users/nick/Desktop/Forked_Genesis/Genesis/data/vla_enviorment_objs/box.obj"  # e.g., PROJECT_ROOT / "assets" / "drop_box.obj"
        scene.add_entity(
                material=gs.materials.Rigid(coup_friction=3.0),
                morph=gs.morphs.Mesh(file=str(box_mesh_path), scale=1.0, pos=(cx, cy, 0.0), euler=(0,0,0)),
                surface=gs.surfaces.Default(color=(0, 200, 0))
        )
        RUNNING_DROP_IN_BOX = (cx, cy)
        # ---------------------------------------------------
    scene.build()
    for _ in range(100):  # ~0.6s at dt=1e-2; adjust if needed
        scene.step()
    return scene, cam, franka, gso_object

def run_rotation(scene, cam, franka, gso_object, df, deform_csv, seg_df, paths, target_choice):
    name, step_no = paths['object_name'], 0
    motors_dof, fingers_dof = np.arange(7), np.arange(7, 9)
    franka.set_dofs_kp(np.array([4500,4500,3500,3500,2000,2000,2000,10,10]))
    franka.set_dofs_kv(np.array([450,450,350,350,200,200,200,1,1]))
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -5, -5]),
        np.array([87, 87, 87, 87, 12, 12, 12, 5, 5]),
    )
    end_effector = franka.get_link("hand")
    vel_limits = {'soft': 0.0002, 'medium': 0.0006, 'hard': 0.0012}
    target_vel = vel_limits.get(target_choice, 0.0002)
    if MATERIAL_TYPE == 'Elastic':
        particle_positions_np = gso_object.get_state().pos.detach().cpu().numpy()[0]
        upper_obj_bound = np.max(particle_positions_np, axis=0)
    elif MATERIAL_TYPE == 'Rigid':
        aabb_min, upper_obj_bound = gso_object.get_AABB().cpu().numpy()
    #cam.start_recording()
    offset = 0.07
    cx, cy, topz = get_live_center_and_top(gso_object, MATERIAL_TYPE)
    x, y, z = cx, cy, topz + offset
    qpos = franka.inverse_kinematics(link=end_effector, pos=np.array([x, y, z + 0.1]), quat=np.array([0, 1, 0, 0]))
    qpos[-2:] = 0.04
    seg_df.loc[len(seg_df)] = ['start', int(scene.t)]

    mm.set_to_pose(scene, cam, franka, gso_object, df, deform_csv, str(paths['images_dir']), PHOTO_INTERVAL, name, qpos, motors_dof, fingers_dof, steps=20)
    mm.descend_to_object(scene, cam, franka, gso_object, df, deform_csv, str(paths['images_dir']), PHOTO_INTERVAL, name, end_effector, x, y, z, motors_dof, fingers_dof, steps=30)
    if DATA_TYPE == "strong" or DATA_TYPE == "eval_strong":
        current_force = 30.0
    elif DATA_TYPE == "medium" or DATA_TYPE == "eval_medium":
        current_force = 10.0
    else:
        current_force = 3.0
    seg_df.loc[len(seg_df)] = ['grasp', int(scene.t)]
    mm.grasp_object(scene, cam, franka, gso_object, df, deform_csv, str(paths['images_dir']), PHOTO_INTERVAL, name, end_effector, x, y, z, motors_dof, fingers_dof, grasp=True, grip_force=-current_force, steps=200)

    seg_df.loc[len(seg_df)] = ['lift', int(scene.t)]
    for i in range(200):
        step_no += 1; curr_z = z + (i * 0.00075)
        if not mm.lift_object(scene, cam, franka, gso_object, df, deform_csv, str(paths['images_dir']), PHOTO_INTERVAL, name, end_effector, x, y, curr_z, motors_dof, fingers_dof, grip_force=-current_force, steps=1):
            break
        if i % 2 == 0: current_force = adjust_force_with_pd_control(current_force, deform_csv, target_vel)

    if MATERIAL_TYPE == 'Elastic':
        particle_positions_np = gso_object.get_state().pos.detach().cpu().numpy()[0]
        pickup_status = 'picked up' if np.min(particle_positions_np, axis=0)[2] > 0.01 else 'not_picked_up'
    elif MATERIAL_TYPE == 'Rigid':
        low, hi = gso_object.get_AABB()
        pickup_status = 'picked up' if hi[2] > 0.01 else 'not_picked_up' # TODO: it can be improved with contact info

    seg_df.loc[len(seg_df)] = [pickup_status, int(scene.t)]

    if pickup_status == 'not_picked_up':
        final_make_step(
            scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
            photo_path=str(paths['images_dir']), photo_interval=PHOTO_INTERVAL, gso_object=gso_object, name=name
        )
        return pickup_status
    r = random.random()
    if r < 0.4:
        seg_df.loc[len(seg_df)] = ['wiggle', int(scene.t)]
        print(type(gso_object))
        success = mm.wiggle_rotation(
            scene, cam, franka, gso_object, df, deform_csv, str(paths['images_dir']), PHOTO_INTERVAL, name,
            end_effector, motors_dof, fingers_dof, grip_force=-current_force
        )
    elif r < 0.4:
        seg_df.loc[len(seg_df)] = ['shake', int(scene.t)]
        success = mm.shake_in_place(
            scene, cam, franka, gso_object, df, deform_csv, str(paths['images_dir']), PHOTO_INTERVAL, name,
            end_effector, motors_dof, fingers_dof, grip_force=-current_force, amplitude=0.035, steps_per_half=30
        )

    actions = []
    angle_choices, joint_indices = [-90, -60, -45, 45, 60, 90], [1, 7]
    STEPS_PER_DEGREE = 6
    for joint_idx in joint_indices:
        chosen_angle = random.choice(angle_choices)
        num_steps = int(abs(chosen_angle) * STEPS_PER_DEGREE)
        actions.append({"name": f"Rotating Joint {joint_idx}", "angle": chosen_angle, "steps": num_steps, "joint_index": joint_idx})
    random.shuffle(actions)

    for i, action in enumerate(actions):
        seg_df.loc[len(seg_df)] = [f'rotate', int(scene.t)]
        print(f"Executing action: {action['name']} by {action['angle']} degrees...")
        mm.rotate_single_joint_by_angle(scene, cam, df, deform_csv, str(paths['images_dir']), PHOTO_INTERVAL, name, franka, motors_dof, fingers_dof, gso_object, gripper_force=-current_force, angle_degrees=action['angle'], joint_index=action['joint_index'], steps=action['steps'])
        step_no += action['steps']
        seg_df.loc[len(seg_df)] = [f'stop moving', int(scene.t)]

        for _ in range(600 - action['steps']):
            franka.control_dofs_force(np.array([-current_force, -current_force]), fingers_dof)
            make_step(
                scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
                photo_path=str(paths['images_dir']), photo_interval=PHOTO_INTERVAL, gso_object=gso_object, name=name,
                gripper_force=-current_force
            )
            step_no += 1

    seg_df.loc[len(seg_df)] = ['rotate', int(scene.t)]
    mm.move_to_place_xy(scene, cam, franka, gso_object, df, deform_csv, str(paths['images_dir']), PHOTO_INTERVAL, name, end_effector, x, y, motors_dof, fingers_dof, grip_force=-current_force)
    seg_df.loc[len(seg_df)] = ['descend', int(scene.t)]
    mm.descend_to_place(scene, cam, franka, gso_object, df, deform_csv, str(paths['images_dir']), PHOTO_INTERVAL, name, end_effector, x, y, upper_obj_bound[2] + offset, motors_dof, fingers_dof, grip_force=-current_force)
    seg_df.loc[len(seg_df)] = ['place', int(scene.t)]
    mm.release_object(scene, cam, franka, gso_object, df, deform_csv, str(paths['images_dir']), PHOTO_INTERVAL, name, fingers_dof, grip_force=-current_force)
    # seg_df.loc[len(seg_df)] = ['pause', int(scene.t)]
    # for _ in range(100):
    #     franka.control_dofs_force(np.array([-current_force, -current_force]), fingers_dof)
    #     make_step(
    #         scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
    #         photo_path=str(paths['images_dir']), photo_interval=PHOTO_INTERVAL, gso_object=gso_object, name=name,
    #         gripper_force=-current_force
    #     )
    #cam.stop_recording(fps=10)
    final_make_step(
        scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
        photo_path=str(paths['images_dir']), photo_interval=PHOTO_INTERVAL, gso_object=gso_object, name=name,
    )
    return pickup_status

def run_new_movement_test(scene, cam, franka, gso_object, df, deform_csv, seg_df, paths, target_choice):
    """
    Rigid-only sequence:
      1) Approach + descend + grasp (force)     [reuses existing mm primitives]
      2) Lift to safe Z                         [reuses existing mm primitives]
      3) Throw toward a planar zone             [new mm.throw_to_zone primitive]
    """
    name, step_no = paths['object_name'], 0
    motors_dof, fingers_dof = np.arange(7), np.arange(7, 9)
    franka.set_dofs_kp(np.array([4500,4500,3500,3500,2000,2000,2000,100,100]))
    franka.set_dofs_kv(np.array([450,450,350,350,200,200,200,10,10]))
    end_effector = franka.get_link("hand")

    vel_limits = {'soft': 0.0002, 'medium': 0.0006, 'hard': 0.0012}
    target_vel = vel_limits.get(target_choice, 0.0002)

    # Rigid bounds
    aabb_min, aabb_max = gso_object.get_AABB().cpu().numpy()
    upper_obj_bound = aabb_max
    height = aabb_max - aabb_min
    #cam.start_recording()

    # === Approach ===
    x, y, z = 0.45, 0.45, upper_obj_bound[2] + 0.08
    qpos = franka.inverse_kinematics(link=end_effector, pos=np.array([x, y, z + 0.1]), quat=np.array([0, 1, 0, 0]))
    qpos[-2:] = 0.04
    seg_df.loc[len(seg_df)] = ['start', int(scene.t)]

    mm.set_to_pose(scene, cam, franka, gso_object, df, deform_csv, str(paths['images_dir']),
                   PHOTO_INTERVAL, name, qpos, motors_dof, fingers_dof, steps=20)
    mm.descend_to_object(scene, cam, franka, gso_object, df, deform_csv, str(paths['images_dir']),
                         PHOTO_INTERVAL, name, end_effector, x, y, z, motors_dof, fingers_dof, steps=30)

    # === Grasp (force-controlled with PD adjustment) ===
    current_force = 3.0
    step_no += 50
    seg_df.loc[len(seg_df)] = ['grasp', int(scene.t)]
    for i in range(200):
        step_no += 1
        if not mm.grasp_object(scene, cam, franka, gso_object, df, deform_csv, str(paths['images_dir']),
                               PHOTO_INTERVAL, name, end_effector, x, y, z,
                               motors_dof, fingers_dof, grasp=True, grip_force=-current_force, steps=1):
            break
        if i % 2 == 0:
            current_force = adjust_force_with_pd_control(current_force, deform_csv, target_vel)

    # === Lift (position + force) ===
    seg_df.loc[len(seg_df)] = ['lift', int(scene.t)]
    for i in range(200):
        step_no += 1
        curr_z = z + (i * 0.00075)
        if not mm.lift_object(scene, cam, franka, gso_object, df, deform_csv, str(paths['images_dir']),
                              PHOTO_INTERVAL, name, end_effector, x, y, curr_z,
                              motors_dof, fingers_dof, grip_force=-current_force, steps=1):
            break
        if i % 2 == 0:
            current_force = adjust_force_with_pd_control(current_force, deform_csv, target_vel)

    # === Pickup check (reuse your logic) ===
    low, hi = gso_object.get_AABB()
    pickup_status = 'picked up' if hi[2] > 0.01 else 'not_picked_up'
    seg_df.loc[len(seg_df)] = [pickup_status, int(scene.t)]
    if pickup_status == 'not_picked_up':
        final_make_step(scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
                        photo_path=str(paths['images_dir']), photo_interval=PHOTO_INTERVAL,
                        gso_object=gso_object, name=name)
        #cam.stop_recording(fps=10)
        return pickup_status

    # === NEW MOVEMENT TEST (PUT IN BOX)===
    # Choose a farther zone than placement to ensure a true ballistic test
    #ZONE = (0.55, 0.80, 0.35, 0.60)
    # seg_df.loc[len(seg_df)] = ['drop_in_box start', int(scene.t)]
    # # Example zone center
    # print(RUNNING_DROP_IN_BOX)
    # (cx, cy) = RUNNING_DROP_IN_BOX

    # print("YASSS", "x", cx, "y", cy)
    # success = mm.drop_in_box(
    #     scene, cam, franka, gso_object, df, deform_csv, str(paths['images_dir']), PHOTO_INTERVAL, name,
    #     end_effector, cx, cy, 0.0,  # z ignored; current EEF z is used
    #     motors_dof, fingers_dof, grip_force=(-current_force)-2, quat=np.array([0,1,0,0])
    # )

    # === NEW MOVEMENT TEST (SHAKE WHILE HOLDING Z-axiz)===

    # seg_df.loc[len(seg_df)] = ['shake start', int(scene.t)]

    # success = mm.shake_in_place(
    #     scene, cam, franka, gso_object, df, deform_csv, str(paths['images_dir']), PHOTO_INTERVAL, name,
    #     end_effector, motors_dof, fingers_dof, grip_force=-current_force, amplitude=0.035, steps_per_half=30
    # )

    # === NEW MOVEMENT TEST (WIGGLE WHILE HOLDING Y-axis)===
    seg_df.loc[len(seg_df)] = ['wiggle start', int(scene.t)]
    print(type(gso_object))
    success = mm.wiggle_rotation(
        scene, cam, franka, gso_object, df, deform_csv, str(paths['images_dir']), PHOTO_INTERVAL, name,
        end_effector, motors_dof, fingers_dof, grip_force=-current_force
    )

    # === NEW MOVEMENT: Push to Target ===
    '''seg_df.loc[len(seg_df)] = ['push start', int(scene.t)]

    # Random target within bounds
    target_xy = (
        np.random.uniform(0.55, 0.75),  # X-range
        np.random.uniform(0.35, 0.55)   # Y-range
    )

    # Random push direction (normalized 2D vector)
    theta = np.random.uniform(-np.pi, np.pi)
    push_vector = np.array([np.cos(theta), np.sin(theta)])

    success = mm.push_object_to_xy(
        scene, cam, franka, gso_object, df, deform_csv, str(paths['images_dir']), PHOTO_INTERVAL, name,
        end_effector, motors_dof, fingers_dof, target_xy, push_vector
    )
    seg_df.loc[len(seg_df)] = ['push end', int(scene.t)]
    # Wind-down for clean segmentation
    for _ in range(40):
        make_step(scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
                  photo_path=str(paths['images_dir']), photo_interval=PHOTO_INTERVAL,
                  gso_object=gso_object, name=name)
        step_no += 1

    seg_df.loc[len(seg_df)] = ['drop_in_box end', int(scene.t)]
    '''
    #cam.stop_recording(fps=10)
    final_make_step(scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
                    photo_path=str(paths['images_dir']), photo_interval=PHOTO_INTERVAL,
                    gso_object=gso_object, name=name)
    return 'movement_success' if success else 'movement_failed'

# -------------------------- GENERATE PLOTS (not used) -------------------------- ##

def generate_plots(df, deform_csv, paths, target_choice):
    """Generates and saves the plots for the simulation results."""
    # This function encapsulates all matplotlib plotting logic.
    name = paths['object_name']
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Deformation plot
    # axs[0].plot(deform_csv.iloc[:, 0], deform_csv.iloc[:, 1], marker='.', color='tab:blue', linewidth=0.5)
    # axs[0].set_xlabel('Time Step')
    # axs[0].set_ylabel('Deformation Metric')
    # axs[0].set_ylim(0, 0.6)
    # axs[0].set_title(f'Object: {name} | Target: {target_choice}')
    # axs[0].grid(True)
    axs[0].axis('off')
    # Force components plot
    force_columns = ['left_fx', 'left_fy', 'left_fz', 'right_fx', 'right_fy', 'right_fz']
    for col in force_columns:
        axs[1].plot(df['step'], df[col], marker='.', label=col)
    #axs[1].plot(deform_csv.iloc[:, 0], deform_csv.iloc[:, 2], marker='.', linestyle='-', color='black', label='grip_force', linewidth=0.5)
    # axs[1].set_ylim(-30, 25)
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Force (N)')
    axs[1].set_title('Force Components Over Time')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(paths['plot'], dpi=300, bbox_inches='tight')
    print(f"Saved plot -> {paths['plot']}")
    #plt.show()  # Show the plot for immediate feedback
    plt.close(fig) # Close the figure to free memory

def main(object_name: str, target_choice: str = 'soft'):
    print(f"🚀 Starting simulation for '{object_name}' with target '{target_choice}'...")
    try:
        paths = setup_paths(object_name, target_choice)
    except FileNotFoundError as e:
        print(f"❌ Aborting: {e}"); return
    print(paths)
    force_df = pd.DataFrame(columns=["step", "left_fx", "left_fy", "left_fz", "left_tx", "left_ty", "left_tz", "right_fx", "right_fy", "right_fz", "right_tx", "right_ty", "right_tz", "dof_0", "dof_1", "dof_2", "dof_3", "dof_4", "dof_5", "dof_6", "dof_7", "dof_8", "eef_x", "eef_y", "eef_z", "left_finger_x", "left_finger_y", "left_finger_z", "right_finger_x", "right_finger_y", "right_finger_z", "control_left_finger", "control_right_finger", "obj_COM_x", "obj_COM_y", "obj_COM_z", "obj_mass", "obj_min_x", "obj_min_y", "obj_min_z", "obj_max_x", "obj_max_y", "obj_max_z", "obj_left_finger", "obj_right_finger", "obj_plane"])
    deform_df = pd.DataFrame(columns=["step", "deformations", "grip_force"])
    segment_df = pd.DataFrame(columns=['action', 'step'])
    scene, cam, franka, gso_object = create_scene(str(paths['input_obj']))
    pickup_status = run_rotation(scene, cam, franka, gso_object, force_df, deform_df, segment_df, paths, target_choice)
    # pickup_status = run_new_movement_test(scene, cam, franka, gso_object, force_df, deform_df, segment_df, paths, target_choice)
    print(f"💾 Saving results to {paths['output_dir']}")
    force_df.to_csv(paths['force_data'], index=False)
    deform_df.to_csv(paths['deformation_data'], index=False)
    segment_df.to_csv(paths['segmentation_data'], index=False)
    generate_plots(df=force_df, deform_csv=deform_df, paths=paths, target_choice=target_choice)
    print(f"✅ Finished simulation for '{object_name}'. Status: {pickup_status}")



def get_tasks_to_run():
    #if 1:
        #return [('Twinlab_100_Whey_Protein_Fuel_Chocolate', 'none'), ('ReadytoUse_Rolled_Fondant_Pure_White_24_oz_box', 'none'), ('Reebok_FUELTRAIN', 'none')]
        # return [('001_chips_can', 'none')]
        # return [('026_sponge', 'none')]
        #return [('002_master_chef_can', 'none')]
        # return [('bottle', 'none')]
        # return [('cube', 'none')]
        #return [('000a3d9fa4ff4c888e71e698694eb0b0', 'none')]

    tasks = []
    raw_data_dir = DATA_ROOT / DATA_TYPE
    if "eval" in DATA_TYPE:
        #objects_dir = DATA_ROOT / "objects"
        objects_dir = DATA_ROOT / "all_files"

    else:
        #objects_dir = DATA_ROOT / "objects/gso"
        objects_dir = DATA_ROOT / "all_files"

    if not objects_dir.exists():
        print(f"❌ Error: Input directory '{objects_dir}' not found."); return []
    object_names = [d.name for d in objects_dir.iterdir() if d.is_dir()]
    print(f"🔍 Found {len(object_names)} objects in '{objects_dir}'.")
    for name in object_names:
        for target in TARGET_CHOICES:
            if not (raw_data_dir / "csv" / name / MATERIAL_TYPE / target / f"{name}_{MATERIAL_TYPE}_{target}.csv").exists():
                print(f"  - Queueing '{name}' with target '{target}' (no previous runs).")
                tasks.append((name, target))
            else:
                print(f"  - Skipping '{name}' with target '{target}' (already processed).")
    return tasks


if __name__ == "__main__":
    tasks_to_run = get_tasks_to_run()
    if not tasks_to_run:
        print("🎉 No new simulations to run."); exit()
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
    for p in processes: p.join()
    print("\n\nAll simulations completed.")
