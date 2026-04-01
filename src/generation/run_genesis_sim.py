import random
import time
from pathlib import Path
from multiprocessing import Process

import pandas as pd
import numpy as np
import torch
import genesis as gs
import matplotlib.pyplot as plt

import master_movement as mm
from make_step import make_step, final_make_step


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

PHOTO_INTERVAL = 80
MATERIAL_TYPE = "Rigid"
TARGET_CHOICES = ["none"]

MAX_PARALLEL_PROCESSES = 8
DATASET = "eval_03312026"
DATASET_TYPE = "ycb"  # Options: "ycb", "gso"
PLACE_TARGET_X_RANGE = (0.20, 0.70)
PLACE_TARGET_Y_RANGE = (0.20, 0.70)
MIN_PLACE_DISTANCE_FROM_SPAWN = 0.14
TARGET_TILE_SIZE = 0.20
TARGET_TILE_THICKNESS = 0.004
TARGET_TILE_COLOR = (0.10, 0.35, 0.90, 1.0)


## -------------------------- PATH SETUP -------------------------- ##
def setup_paths(object_name: str, target_choice: str) -> dict:
    input_obj_path = DATA_ROOT / "objects" / DATASET_TYPE / object_name / "model.obj"
    if DATASET_TYPE == "hugging_face":
        input_obj_path = DATA_ROOT / "objects" / DATASET_TYPE / object_name / f"{object_name}.glb"
    if not input_obj_path.exists():
        raise FileNotFoundError(f"Input file not found at: {input_obj_path}")

    output_dir = DATA_ROOT / DATASET / "csv" / object_name / MATERIAL_TYPE / target_choice
    image_root = DATA_ROOT / DATASET / "images" / object_name / MATERIAL_TYPE / target_choice
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
def sample_place_target(source_xy):
    source_xy = np.asarray(source_xy, dtype=float)
    for _ in range(100):
        tx = random.uniform(*PLACE_TARGET_X_RANGE)
        ty = random.uniform(*PLACE_TARGET_Y_RANGE)
        if np.linalg.norm(np.array([tx, ty]) - source_xy) >= MIN_PLACE_DISTANCE_FROM_SPAWN:
            return float(tx), float(ty)
    return float(PLACE_TARGET_X_RANGE[1]), float(PLACE_TARGET_Y_RANGE[1])


def create_scene(obj_path: str):
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
    cam = scene.add_camera(res=(1080, 1080), pos=(-1.5, 1.5, 0.25), lookat=(0.45, 0.45, 0.4), fov=30)
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(coup_friction=0.0))
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        material=gs.materials.Rigid(coup_friction=3.0, friction=1.0),
    )

    rho = 200.0 * random.uniform(1.0, 5.0)
    gso_object = scene.add_entity(
        material=gs.materials.Rigid(rho=rho),
        morph=gs.morphs.Mesh(file=obj_path, scale=object_scale, pos=object_spawn_pos, euler=object_euler),
        surface=gs.surfaces.Default(color=color),
    )

    target_xy = sample_place_target(drop_center)
    scene.add_entity(
        material=gs.materials.Rigid(rho=800, friction=1.0, coup_friction=1.0),
        morph=gs.morphs.Box(
            size=(TARGET_TILE_SIZE, TARGET_TILE_SIZE, TARGET_TILE_THICKNESS),
            pos=(target_xy[0], target_xy[1], TARGET_TILE_THICKNESS * 0.5),
            fixed=True,
        ),
        surface=gs.surfaces.Default(color=TARGET_TILE_COLOR),
    )

    scene.build()
    return scene, cam, franka, gso_object, target_xy


def run_rotation(scene, cam, franka, gso_object, df, deform_csv, seg_df, paths, place_target_xy):
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
    obj_center = 0.5 * (lower_obj_bound + upper_obj_bound)
    x, y = obj_center[0], obj_center[1]
    z = upper_obj_bound[2] + offset
    place_x, place_y = place_target_xy

    qpos = franka.inverse_kinematics(link=end_effector, pos=np.array([x, y, z + 0.1]), quat=np.array([0, 1, 0, 0]))
    qpos[-2:] = 0.04

    seg_df.loc[len(seg_df)] = [f"target_{place_x:.3f}_{place_y:.3f}", int(scene.t)]
    seg_df.loc[len(seg_df)] = ["grasp", int(scene.t)]
    mm.set_to_pose(
        scene,
        cam,
        franka,
        gso_object,
        df,
        deform_csv,
        str(paths["images_dir"]),
        PHOTO_INTERVAL,
        name,
        qpos,
        motors_dof,
        fingers_dof,
        steps=20,
    )
    mm.descend_to_object(
        scene,
        cam,
        franka,
        gso_object,
        df,
        deform_csv,
        str(paths["images_dir"]),
        PHOTO_INTERVAL,
        name,
        end_effector,
        x,
        y,
        z,
        motors_dof,
        fingers_dof,
        steps=30,
    )

    current_force = 10.0
    mm.grasp_object_position(
        scene,
        cam,
        franka,
        gso_object,
        df,
        deform_csv,
        str(paths["images_dir"]),
        PHOTO_INTERVAL,
        name,
        end_effector,
        x,
        y,
        z,
        motors_dof,
        fingers_dof,
        grip_force=-current_force,
        steps=200,
    )

    seg_df.loc[len(seg_df)] = ["hold", int(scene.t)]
    seg_df.loc[len(seg_df)] = ["lift", int(scene.t)]
    for i in range(200):
        curr_z = z + (i * 0.00075)
        if not mm.lift_object(
            scene,
            cam,
            franka,
            gso_object,
            df,
            deform_csv,
            str(paths["images_dir"]),
            PHOTO_INTERVAL,
            name,
            end_effector,
            x,
            y,
            curr_z,
            motors_dof,
            fingers_dof,
            grip_force=-current_force,
            steps=1,
        ):
            break

    seg_df.loc[len(seg_df)] = ["move_to_target", int(scene.t)]
    mm.move_to_place_xy(
        scene,
        cam,
        franka,
        gso_object,
        df,
        deform_csv,
        str(paths["images_dir"]),
        PHOTO_INTERVAL,
        name,
        end_effector,
        place_x,
        place_y,
        motors_dof,
        fingers_dof,
        grip_force=-current_force,
    )

    seg_df.loc[len(seg_df)] = ["descend", int(scene.t)]
    mm.descend_to_place_cautiously(
        scene,
        cam,
        franka,
        gso_object,
        df,
        deform_csv,
        str(paths["images_dir"]),
        PHOTO_INTERVAL,
        name,
        end_effector,
        place_x,
        place_y,
        upper_obj_bound[2] + offset,
        motors_dof,
        fingers_dof,
        grip_force=-current_force,
    )
    seg_df.loc[len(seg_df)] = ["place", int(scene.t)]
    mm.release_object(
        scene,
        cam,
        franka,
        gso_object,
        df,
        deform_csv,
        str(paths["images_dir"]),
        PHOTO_INTERVAL,
        name,
        fingers_dof,
        grip_force=-current_force,
    )

    for _ in range(100):
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
        )

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


def main(object_name: str, target_choice: str = "none"):
    print(f"🚀 Starting simulation for '{object_name}' with target '{target_choice}'...")
    try:
        paths = setup_paths(object_name, target_choice)
    except FileNotFoundError as e:
        print(f"❌ Aborting: {e}")
        return

    print(paths)
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
        ]
    )
    deform_df = pd.DataFrame(columns=["step", "deformations", "grip_force"])
    segment_df = pd.DataFrame(columns=["action", "step"])

    scene, cam, franka, gso_object, place_target_xy = create_scene(str(paths["input_obj"]))
    run_rotation(scene, cam, franka, gso_object, force_df, deform_df, segment_df, paths, place_target_xy)

    print(f"💾 Saving results to {paths['output_dir']}")
    force_df.to_csv(paths["force_data"], index=False)
    deform_df.to_csv(paths["deformation_data"], index=False)
    segment_df.to_csv(paths["segmentation_data"], index=False)
    generate_plots(df=force_df, paths=paths)
    print(f"✅ Finished simulation for '{object_name}'.")


def get_tasks_to_run():
    if 1:
        return [("001_chips_can", "none")]

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
