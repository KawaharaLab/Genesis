import random
import sys
import time
from pathlib import Path
from datetime import datetime
from multiprocessing import Process

import pandas as pd
import numpy as np
import torch
#This is just for Nicks computer to work
#sys.path.insert(0,"/Users/nick/Desktop/Forked_Genesis/Genesis")
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

RUNNING_DROP_IN_BOX = True
## -------------------------- PATH SETUP -------------------------- ##

def setup_paths(object_name: str, target_choice: str) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    simulation_id = f"{target_choice}_{MATERIAL_TYPE.lower()}_{timestamp}"

    input_obj_path = DATA_ROOT / "gso_obj" / object_name / "model.obj"
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


## -------------------------- SIMULATION CORE -------------------------- ##
def sample_drop_box_bounds(cx_range=(-0.55, 0.75),cy_range=(0.35, 0.85),inner_w=0.25,inner_h=0.25):
    cx = random.uniform(*cx_range)
    cy = random.uniform(*cy_range)
    x_min, x_max = cx - inner_w/2.0, cx + inner_w/2.0
    y_min, y_max = cy - inner_h/2.0, cy + inner_h/2.0
    return (x_min, x_max, y_min, y_max), (cx, cy)


def create_scene(obj_path: str):
    global RUNNING_DROP_IN_BOX
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
    x, y, z = 0.45, 0.45, upper_obj_bound[2] + 0.08
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

    cam.start_recording()

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
        cam.stop_recording(fps=10)
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
    cam.stop_recording(fps=10)
    final_make_step(scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
                    photo_path=str(paths['images_dir']), photo_interval=PHOTO_INTERVAL,
                    gso_object=gso_object, name=name)
    return 'movement_success' if success else 'movement_failed'



## -------------------------- GENERATE PLOTS (not used) -------------------------- ##

# def generate_plots(df, deform_csv, paths, target_choice):
#     """Generates and saves the plots for the simulation results."""
#     # This function encapsulates all matplotlib plotting logic.
#     name = paths['object_name']
#     fig, axs = plt.subplots(1, 2, figsize=(16, 6))

#     # Deformation plot
#     axs[0].plot(deform_csv.iloc[:, 0], deform_csv.iloc[:, 1], marker='.', color='tab:blue', linewidth=0.5)
#     axs[0].set_xlabel('Time Step')
#     axs[0].set_ylabel('Deformation Metric')
#     axs[0].set_ylim(0, 0.6)
#     axs[0].set_title(f'Object: {name} | Target: {target_choice}')
#     axs[0].grid(True)

#     # Force components plot
#     force_columns = ['left_fx', 'left_fy', 'left_fz', 'right_fx', 'right_fy', 'right_fz']
#     for col in force_columns:
#         axs[1].plot(df['step'], df[col], marker='.', label=col)
#     axs[1].plot(deform_csv.iloc[:, 0], deform_csv.iloc[:, 2], marker='.', linestyle='-', color='black', label='grip_force', linewidth=0.5)
#     axs[1].set_ylim(-30, 25)
#     axs[1].set_xlabel('Time Step')
#     axs[1].set_ylabel('Force (N)')
#     axs[1].set_title('Force Components Over Time')
#     axs[1].grid(True)
#     axs[1].legend()

#     plt.tight_layout()
#     plt.savefig(paths['plot'], dpi=300, bbox_inches='tight')
#     print(f"Saved plot -> {paths['plot']}")
#     plt.show()  # Show the plot for immediate feedback
#     plt.close(fig) # Close the figure to free memory


## -------------------------- MAIN ORCHESTRATION -------------------------- ##

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
    #pickup_status = run_rotation(scene, cam, franka, gso_object, force_df, deform_df, segment_df, paths, target_choice)
    pickup_status = run_new_movement_test(scene, cam, franka, gso_object, force_df, deform_df, segment_df, paths, target_choice)
    print(f"💾 Saving results to {paths['output_dir']}")
    force_df.to_csv(paths['force_data'], index=False)
    deform_df.to_csv(paths['deformation_data'], index=False)
    segment_df.to_csv(paths['segmentation_data'], index=False)
    print(f"✅ Finished simulation for '{object_name}'. Status: {pickup_status}")



def get_tasks_to_run():
    if 1:
        return [('Twinlab_100_Whey_Protein_Fuel_Chocolate', 'none'), ('ReadytoUse_Rolled_Fondant_Pure_White_24_oz_box', 'none'), ('Reebok_FUELTRAIN', 'none')]
        #return [('026_sponge', 'none')]
        # return [('002_master_chef_can', 'hard')]

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
