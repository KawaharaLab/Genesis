import random
import time
from pathlib import Path
from datetime import datetime
from multiprocessing import Process

import pandas as pd
import numpy as np
import torch

import genesis as gs

# Assuming these are your custom modules within the src/ directory
import master_movement as mm
from make_step import make_step, final_make_step


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

PHOTO_INTERVAL = 10
MATERIAL_TYPE = "Rigid"
# MATERIAL_TYPE = "Elastic"
# TARGET_CHOICES = ['soft', 'medium', 'hard']
TARGET_CHOICES = ['soft']  # For simplicity, only using 'soft' in this example
MAX_PARALLEL_PROCESSES = 8


def setup_paths(object_name: str, target_choice: str) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    simulation_id = f"{target_choice}_{MATERIAL_TYPE.lower()}_{timestamp}"

    input_obj_path = DATA_ROOT / "objects" / object_name / "model.obj"
    if not input_obj_path.exists():
        raise FileNotFoundError(f"Input file not found at: {input_obj_path}")

    output_dir = DATA_ROOT / "raw" / "csv" / object_name / MATERIAL_TYPE / target_choice
    images_dir = DATA_ROOT / "raw" / "images" / object_name / MATERIAL_TYPE / target_choice

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    return {
        "input_obj": input_obj_path,
        "output_dir": output_dir,
        "images_dir": images_dir,
        "force_data": output_dir / f"{object_name}_{MATERIAL_TYPE}_{target_choice}.csv",
        "deformation_data": output_dir / f"{object_name}_{MATERIAL_TYPE}_deform_{target_choice}.csv",
        "segmentation_data": output_dir / f"{object_name}_{MATERIAL_TYPE}_steps_{target_choice}.csv",
        "object_name": object_name,
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


def set_grasp(obj_path):
    GRIPPER_MIN_WIDTH, GRIPPER_MAX_WIDTH = 0.002, 0.075
    bbox = get_obj_bounding_box(obj_path)
    scale = 1.0
    if GRIPPER_MIN_WIDTH < bbox[0] < GRIPPER_MAX_WIDTH:
        euler = (0, 0, 90)
    elif GRIPPER_MIN_WIDTH < bbox[1] < GRIPPER_MAX_WIDTH:
        euler = (0, 0, 0)
    else:
        scale = (0.080 / bbox[0]) if bbox[0] < bbox[1] else (0.080 / bbox[1])
        euler = (0, 0, 90) if bbox[0] < bbox[1] else (0, 0, 0)
    return scale, euler


def adjust_force_with_pd_control(current_force, deform_csv, target_vel):
    if len(deform_csv) < 2: return current_force
    deform_velocity = deform_csv.iloc[-1, 1] - deform_csv.iloc[-2, 1]
    if deform_velocity > 1.2 * target_vel:
        current_force -= 0.1
    elif deform_velocity < 0.8 * target_vel:
        current_force += 0.1
    return max(0.1, current_force)


def create_scene(obj_path: str):
    if torch.cuda.is_available():
        gs.init(backend=gs.gpu)
    else:
        gs.init(backend=gs.cpu)
    object_scale, object_euler = set_grasp(obj_path)
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
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"), material=gs.materials.Rigid(coup_friction=3.0))
    if MATERIAL_TYPE == 'Elastic':
        gso_object = scene.add_entity(
            material=gs.materials.MPM.Elastic(),
            morph=gs.morphs.Mesh(file=obj_path, scale=object_scale, pos=(0.45, 0.45, 0.001), euler=object_euler),
            surface=gs.surfaces.Default(color=color)
        )
    elif MATERIAL_TYPE == 'Rigid':
        gso_object = scene.add_entity(
            material=gs.materials.Rigid(),
            morph=gs.morphs.Mesh(file=obj_path, scale=object_scale, pos=(0.45, 0.45, 0.001), euler=object_euler),
            surface=gs.surfaces.Default(color=color)
        )
    # material=gs.materials.MPM.Elastic(E=1.5e6, nu=0.45, rho=1100.0, sampler="pbs", model="corotation") : Self-made Elas
    scene.build()
    return scene, cam, franka, gso_object

def run_rotation(scene, cam, franka, gso_object, df, deform_csv, seg_df, paths, target_choice):
    name, step_no = paths['object_name'], 0
    motors_dof, fingers_dof = np.arange(7), np.arange(7, 9)
    franka.set_dofs_kp(np.array([4500,4500,3500,3500,2000,2000,2000,100,100]))
    franka.set_dofs_kv(np.array([450,450,350,350,200,200,200,10,10]))
    end_effector = franka.get_link("hand")
    vel_limits = {'soft': 0.0002, 'medium': 0.0006, 'hard': 0.0012}
    target_vel = vel_limits.get(target_choice, 0.0002)
    if MATERIAL_TYPE == 'Elastic':
        particle_positions_np = gso_object.get_state().pos.detach().cpu().numpy()[0]
        upper_obj_bound = np.max(particle_positions_np, axis=0)
    elif MATERIAL_TYPE == 'Rigid':
        aabb_min, upper_obj_bound = gso_object.get_AABB().cpu().numpy()
    cam.start_recording()
    x, y, z = 0.45, 0.45, upper_obj_bound[2] + 0.07
    qpos = franka.inverse_kinematics(link=end_effector, pos=np.array([x,y,z+0.1]), quat=np.array([0,1,0,0]))
    qpos[-2:] = 0.04
    seg_df.loc[len(seg_df)] = ['start', int(scene.t)]

    mm.set_to_pose(scene, cam, franka, gso_object, df, deform_csv, str(paths['images_dir']), PHOTO_INTERVAL, name, qpos, motors_dof, fingers_dof, steps=20)
    mm.descend_to_object(scene, cam, franka, gso_object, df, deform_csv, str(paths['images_dir']), PHOTO_INTERVAL, name, end_effector, x, y, z, motors_dof, fingers_dof, steps=30)
    
    current_force = 3.0
    step_no += 50
    seg_df.loc[len(seg_df)] = ['grasp', int(scene.t)]
    for i in range(200):
        step_no += 1
        if not mm.grasp_object(scene, cam, franka, gso_object, df, deform_csv, str(paths['images_dir']), PHOTO_INTERVAL, name, end_effector, x, y, z, motors_dof, fingers_dof, grasp=True, grip_force=-current_force, steps=1):
            break
        if i % 2 == 0: current_force = adjust_force_with_pd_control(current_force, deform_csv, target_vel)

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
    mm.descend_to_place(scene, cam, franka, gso_object, df, deform_csv, str(paths['images_dir']), PHOTO_INTERVAL, name, end_effector, x, y, upper_obj_bound[2] + 0.07, motors_dof, fingers_dof, grip_force=-current_force)
    seg_df.loc[len(seg_df)] = ['place', int(scene.t)]
    mm.release_object(scene, cam, franka, gso_object, df, deform_csv, str(paths['images_dir']), PHOTO_INTERVAL, name, fingers_dof, grip_force=-current_force)

    cam.stop_recording(fps=10)
    final_make_step(
        scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
        photo_path=str(paths['images_dir']), photo_interval=PHOTO_INTERVAL, gso_object=gso_object, name=name,
    )
    return pickup_status


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
    print(f"💾 Saving results to {paths['output_dir']}")
    force_df.to_csv(paths['force_data'], index=False)
    deform_df.to_csv(paths['deformation_data'], index=False)
    segment_df.to_csv(paths['segmentation_data'], index=False)
    print(f"✅ Finished simulation for '{object_name}'. Status: {pickup_status}")



def get_tasks_to_run():
    if 1:
        # return [('3D_Dollhouse_Refrigerator', 'soft')]
        return [('026_sponge', 'none')]
        # return [('002_master_chef_can', 'none')]

    tasks, objects_dir, raw_data_dir = [], DATA_ROOT / "objects", DATA_ROOT / "raw"
    if not objects_dir.exists():
        print(f"❌ Error: Input directory '{objects_dir}' not found."); return []
    object_names = [d.name for d in objects_dir.iterdir() if d.is_dir()]
    print(f"🔍 Found {len(object_names)} objects in '{objects_dir}'.")
    for name in object_names:
        if MATERIAL_TYPE == 'Elastic':
            for target in TARGET_CHOICES:
                if not (raw_data_dir / name).exists():
                    print(f"  - Queueing '{name}' with target '{target}' (no previous runs).")
                    tasks.append((name, target))
        elif MATERIAL_TYPE == 'Rigid':
            if not (raw_data_dir / name).exists():
                print(f"  - Queueing '{name}' (no previous runs).")
                tasks.append((name, 'none'))
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
