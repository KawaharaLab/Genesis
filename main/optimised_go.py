import os
import sys
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process
import time

import genesis as gs
import master_movement as mm
from make_step import make_step

# --- Configuration Section ---
# Grouped constants for easier management.
BASE_PATH = "/Users/no.166/Documents/Azka\'s Workspace/Genesis"
PHOTO_INTERVAL = 250

MATERIAL_TYPE = "Elastic"
# OBJECT_SCALE = 0.62

COLOR = random.choice([
             (255, 0, 0),
             (0, 255, 0),
             (0, 0, 255),
             (255, 255, 0),
             (0, 255, 255),
            (255, 0, 255),
        ])
def setup_paths(name, target_choice):
    """Creates all necessary directories and returns a dictionary of paths."""
    # This function consolidates all path and directory creation logic.

    # Define base paths for data types
    base_data_path = BASE_PATH + '/main/data/picked_up'
    photo_base = os.path.join(base_data_path, 'photos', name, MATERIAL_TYPE, target_choice)
    csv_base = os.path.join(base_data_path, 'csv', name, MATERIAL_TYPE, target_choice)
    # video_base = os.path.join(base_data_path, 'videos', name)

    # Create directories
    os.makedirs(photo_base, exist_ok=True)
    os.makedirs(csv_base, exist_ok=True)
    # os.makedirs(video_base, exist_ok=True)

    # Return a dictionary of all generated paths
    return {
        "photo": photo_base,
        "csv": os.path.join(csv_base, f'{name}_{MATERIAL_TYPE}_{target_choice}.csv'),
        "deform_csv": os.path.join(csv_base, f'{name}_{MATERIAL_TYPE}_deform_{target_choice}.csv'),
        "plot": os.path.join(csv_base, f'{name}_{MATERIAL_TYPE}_{target_choice}.png'),
        "name": name,
    }

def get_obj_bounding_box(obj_path):
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')

    with open(obj_path, 'r') as file:
        for line in file:
            if line.startswith('v '):  # vertex line
                _, x, y, z = line.strip().split()
                x, y, z = float(x), float(y), float(z)
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                min_z = min(min_z, z)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
                max_z = max(max_z, z)

    return [max_x-min_x, max_y-min_y, max_z-min_z]

def set_grasp(obj_path):
    """Sets the gripper constraints based on the object bounding box."""
    # --- Gripper Constraints (based on Panda robot) ---
    # We use slightly more conservative values for stability.
    GRIPPER_MIN_WIDTH = 0.002  # 2 mm
    GRIPPER_MAX_WIDTH = 0.075  # 75 mm (Slightly less than the 80mm absolute max)

    bbox = get_obj_bounding_box(obj_path)
    scale = 1.0  # Default scale for the object
    if GRIPPER_MIN_WIDTH < bbox[0] < GRIPPER_MAX_WIDTH:
        euler = (0, 0, 90)  # Pick up along the x-axis
    elif GRIPPER_MIN_WIDTH < bbox[1] < GRIPPER_MAX_WIDTH:
        euler = (0, 0, 0)   # Pick up along the y-axis
    else:
        if bbox[0] < bbox[1]:
            scale = 0.080/bbox[0]
            euler = (0, 0, 90)  # Pick up along the x-axis
        else:
            scale = 0.080/bbox[1]
            euler = (0, 0, 0)   # Pick up along the y-axis

    return scale, euler


def create_scene(obj_path):
    """Initializes and returns the simulation scene, camera, robot, and object."""
    # This function encapsulates the entire Genesis simulation setup.
    gs.init(backend=gs.cpu)
    OBJECT_SCALE, OBJECT_EULER = set_grasp(obj_path)
    # OBJECT_SCALE = 1.0  # Default scale for the object
    # OBJECT_EULER = (0, 0, 90)  # Default orientation for the object

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1e-3, substeps=15),
        viewer_options=gs.options.ViewerOptions(camera_pos=(3, -1, 1.5), camera_lookat=(0.0, 0.0, 0.0), camera_fov=30),
        show_viewer=False,
        vis_options=gs.options.VisOptions(visualize_mpm_boundary=True),
        mpm_options=gs.options.MPMOptions(lower_bound=(0.0, -0.1, -0.05), upper_bound=(0.75, 1.0, 1.0), grid_density=128)
    )

    cam = scene.add_camera(res=(1280, 720), pos=(-1.5, 1.5, 0.25), lookat=(0.45, 0.45, 0.4), fov=30)

    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid())

    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"), material=gs.materials.Rigid(coup_friction=3.0))
    gso_object = scene.add_entity(
        # material=gs.materials.MPM.Elastic(E=1.5e6, nu=0.45, rho=1100.0, sampler="pbs", model="corotation"),
        material=gs.materials.MPM.Elastic(),
        morph=gs.morphs.Mesh(file=obj_path, scale=OBJECT_SCALE, pos=(0.45, 0.45, 0), euler=OBJECT_EULER),

        surface = gs.surfaces.Default(color = COLOR)

    )

    scene.build()
    return scene, cam, franka, gso_object

def adjust_force_with_pd_control(current_force, deform_csv, target_vel):
    """
    Adjusts force based on deformation velocity.
    This function eliminates the duplicated logic from the original script.
    """
    deform_velocity = deform_csv.iloc[-1, 1] - deform_csv.iloc[-2, 1]

    if deform_velocity > 1.2 * target_vel:
        current_force -= 0.2
    elif deform_velocity < 0.8 * target_vel:
        current_force += 0.2

    return current_force

# def run_pd_control_sequence(scene, cam, franka, gso_object, df, deform_csv, paths, target_choice):
#     """Runs the entire PD control motion sequence."""
#     # This function contains the main robot motion logic.
#     name = paths['name']

#     # Setup robot DOFs
#     motors_dof = np.arange(7)
#     fingers_dof = np.arange(7, 9)
#     franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
#     franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
#     franka.set_dofs_force_range(np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]), np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]))
#     end_effector = franka.get_link("hand")

#     # Determine object height and pre-grasp pose
#     particle_positions_np = gso_object.get_state().pos.detach().cpu().numpy()
#     upper_obj_bound = np.max(particle_positions_np[0], axis=0)
#     x, y, z = 0.45, 0.45, upper_obj_bound[2] + 0.06
#     qpos = franka.inverse_kinematics(link=end_effector, pos=np.array([x, y, z]), quat=np.array([0, 1, 0, 0]))
#     qpos[-2:] = 0.04

#     # Velocity limits
#     vel_limits = {'soft': 0.0002, 'medium': 0.0006, 'hard': 0.0011}
#     target_vel = vel_limits.get(target_choice, 0.0002) # Default to soft
#     print(f"Target velocity set to: {target_vel}")

#     # franka.set_dofs_position(qpos[:-2], motors_dof)
#     # franka.set_dofs_position(qpos[-2:], fingers_dof)

#     # Start motion sequence
#     # cam.start_recording()

#     # 0-50: Move to pre-grasp
#     print("Moving to pre-grasp pose...")
#     mm.move_to_pose(scene, cam, franka, gso_object, df, deform_csv, paths['photo'], PHOTO_INTERVAL, name, qpos, motors_dof, fingers_dof, steps=50)

#     # 50-100: Descend
#     print("######################### Descending to object... #########################")
#     mm.descend_to_object(scene, cam, franka, gso_object, df, deform_csv, paths['photo'], PHOTO_INTERVAL, name, end_effector, x, y, z, motors_dof, fingers_dof, steps=50)

#     current_force = 3.0

#     # 100-300: Grasp object
#     print("######################### Grasping object with PD control... #########################")
#     for i in range(350):
#         if mm.grasp_object(scene, cam, franka, gso_object, df, deform_csv, paths['photo'], PHOTO_INTERVAL, name, end_effector, x, y, z, motors_dof, fingers_dof, grasp=True, grip_force=-current_force, steps=1) == False:
#             current_force = 0.0
#             qpos[-2:] = 0.04
#             franka.set_dofs_position(qpos[-2:], fingers_dof)
#             print('reset worked !')
#             break
#         if i % 2 == 0:
#             current_force = adjust_force_with_pd_control(current_force, deform_csv, target_vel)


#     # 300-500: Lift object
#     print("######################### Lifting object with PD control... #########################")
#     for i in range(200):
#         curr_z = z + (i * 0.00075)
#         if mm.lift_object(scene, cam, franka, gso_object, df, deform_csv, paths['photo'], PHOTO_INTERVAL, name, end_effector, x, y, curr_z, motors_dof, fingers_dof, grip_force=-current_force, steps=1) == False:
#             current_force = 0.0
#             qpos[-2:] = 0.04
#             franka.set_dofs_position(qpos[-2:], fingers_dof)
#             break
#         if i % 2 == 0:
#             current_force = adjust_force_with_pd_control(current_force, deform_csv, target_vel)


#     # 500-600: Drop object
#     print("######################### Dropping object... #########################")
#     mm.grasp_object(scene, cam, franka, gso_object, df, deform_csv, paths['photo'], PHOTO_INTERVAL, name, end_effector, x, y, z, motors_dof, fingers_dof, grasp=False, grip_force=-current_force, steps=100)


#     # 600-700: Complete motion
#     print("######################### Completing motion... #########################")
#     for _ in range(100):
#         make_step(scene, cam, franka, df, paths['photo'], PHOTO_INTERVAL, gso_object, deform_csv, name, gripper_force=0.0)


#     # cam.stop_recording()
#     #print(f"Saved video -> {paths['video']}")

def run_rotation(scene, cam, franka, gso_object, df, deform_csv, paths, target_choice):
    """Runs the entire rotation motion sequence."""
    # This function contains the main robot motion logic.
    name = paths['name']

    # Setup robot DOFs
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
    franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
    franka.set_dofs_force_range(np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]), np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]))
    end_effector = franka.get_link("hand")

    # Determine object height and pre-grasp pose
    particle_positions_np = gso_object.get_state().pos.detach().cpu().numpy()
    upper_obj_bound = np.max(particle_positions_np[0], axis=0)
    x, y, z = 0.45, 0.45, upper_obj_bound[2] + 0.05
    qpos = franka.inverse_kinematics(link=end_effector, pos=np.array([x, y, z]), quat=np.array([0, 1, 0, 0]))
    qpos[-2:] = 0.04

    # Velocity limits
    vel_limits = {'soft': 0.0002, 'medium': 0.0006, 'hard': 0.0012}
    target_vel = vel_limits.get(target_choice, 0.0002) # Default to soft

    # Start motion sequence
    # cam.start_recording()

    # 0-20: Move to pre-grasp
    # print("Moving to pre-grasp pose...")
    mm.move_to_pose(scene, cam, franka, gso_object, df, deform_csv, paths['photo'], PHOTO_INTERVAL, name, qpos, motors_dof, fingers_dof, steps=20)

    # 20-50: Descend
    print("######################### Descending to object... #########################")
    mm.descend_to_object(scene, cam, franka, gso_object, df, deform_csv, paths['photo'], PHOTO_INTERVAL, name, end_effector, x, y, z, motors_dof, fingers_dof, steps=30)

    current_force = 3.0

    # 50-400: Grasp object
    print("######################### Grasping object with PD control... #########################")
    for i in range(350):
        if mm.grasp_object(scene, cam, franka, gso_object, df, deform_csv, paths['photo'], PHOTO_INTERVAL, name, end_effector, x, y, z, motors_dof, fingers_dof, grasp=True, grip_force=-current_force, steps=1) == False:
            current_force = 0.0
            qpos[-2:] = 0.04
            franka.set_dofs_position(qpos[-2:], fingers_dof)
            break
        if i % 2.5 == 0:
            current_force = adjust_force_with_pd_control(current_force, deform_csv, target_vel)


    # 400-600: Lift object
    print("######################### Lifting object with PD control... #########################")
    for i in range(200):
        curr_z = z + (i * 0.00075)
        if mm.lift_object(scene, cam, franka, gso_object, df, deform_csv, paths['photo'], PHOTO_INTERVAL, name, end_effector, x, y, curr_z, motors_dof, fingers_dof, grip_force=-current_force, steps=1) == False:
            current_force = 0.0
            qpos[-2:] = 0.04
            franka.set_dofs_position(qpos[-2:], fingers_dof)
            break
        if i % 2.5 == 0:
            current_force = adjust_force_with_pd_control(current_force, deform_csv, target_vel)

    lower_obj_bound = np.min(particle_positions_np[0], axis=0)
    if lower_obj_bound[2] > 0:
        print('Object picked up, continuing' + f' {lower_obj_bound[2]} > 0')
    else:
        print("Object not picked up, exiting" + f' {lower_obj_bound[2]} <= 0')
        
        import shutil
        
        # Define source and destination paths
        not_picked_up_photo_path = f'{BASE_PATH}/main/data/not_picked_up/photos/{name}/{MATERIAL_TYPE}/'
        not_picked_up_csv_dir = f'{BASE_PATH}/main/data/not_picked_up/csv/{name}/{MATERIAL_TYPE}/'
            

        if os.path.exists(not_picked_up_photo_path):
            if os.path.isdir(not_picked_up_photo_path):
                shutil.rmtree(not_picked_up_photo_path)
            else:
                os.remove(not_picked_up_photo_path)

        # Create destination directories
        os.makedirs(not_picked_up_photo_path, exist_ok=True)
        os.makedirs(not_picked_up_csv_dir, exist_ok=True)

        # Move photo directory (with safety check)
        if os.path.exists(paths['photo']):
            shutil.move(paths['photo'], not_picked_up_photo_path)
        else:
            print(f"Warning: source photo path {paths['photo']} does not exist.")

        exit()



    # 600-800: Rotate robot by angle
    print("######################### Rotating robot by angle... #########################")

    actions = []
    angle_choices = [-90, -60, -45, 45, 60, 90]
    # The values for "n" that differ between your original actions
    n_values = [1, 7]

    # Define a relationship between angle and steps
    # Example: 10 steps for every degree of rotation. Adjust this multiplier as needed.
    STEPS_PER_DEGREE = 6

    for n_val in n_values:
        # 1. Choose a random angle for this specific action
        chosen_angle = random.choice(angle_choices)

        # 2. Calculate the number of steps based on the absolute value of the angle
        num_steps = int(abs(chosen_angle) * STEPS_PER_DEGREE)

        # 3. Append the complete action dictionary to the list
        actions.append({
            "name": "Rotating Robot with Quaternion",
            "func": mm.rotate_robot_by_angle_quat,
            "args": {
                "scene": scene,
                "cam": cam,
                "df": df,
                "photo_path": paths['photo'],
                "photo_interval": PHOTO_INTERVAL,
                "name": name,
                "franka": franka,
                "motors_dof": motors_dof,
                "fingers_dof": fingers_dof,
                "end_effector": end_effector,
                "gso_object": gso_object,
                "deform_csv": deform_csv,
                "angle_degrees": chosen_angle,
                "z_obj": z,
                "gripper_force": -current_force,
                "steps": num_steps, # The new dynamic step count
                "n": n_val
            }
        })

    random.shuffle(actions)  # Randomize the order

    for action in actions:
        # The print statement now shows the corresponding step count
        print(f"{action['name']} with angle {action['args']['angle_degrees']} degrees and {action['args']['steps']} steps...")
        action["func"](**action["args"])
        for _ in range(100):
            franka.control_dofs_force(np.array([-current_force, -current_force]), fingers_dof)
            make_step(scene, cam, franka, df, paths['photo'], PHOTO_INTERVAL, gso_object, deform_csv, name, gripper_force=-current_force)


    # 770-970: Complete motion
    print("######################### Completing motion... #########################")
    for _ in range(150):
        make_step(scene, cam, franka, df, paths['photo'], PHOTO_INTERVAL, gso_object, deform_csv, name)

    # cam.stop_recording()
    

def generate_plots(df, deform_csv, paths, target_choice):
    """Generates and saves the plots for the simulation results."""
    # This function encapsulates all matplotlib plotting logic.
    name = paths['name']
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Deformation plot
    axs[0].plot(deform_csv.iloc[:, 0], deform_csv.iloc[:, 1], marker='.', color='tab:blue', linewidth=0.5)
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Deformation Metric')
    axs[0].set_ylim(0, 0.6)
    axs[0].set_title(f'Object: {name} | Target: {target_choice}')
    axs[0].grid(True)

    # Force components plot
    force_columns = ['left_fx', 'left_fy', 'left_fz', 'right_fx', 'right_fy', 'right_fz']
    for col in force_columns:
        axs[1].plot(df['step'], df[col], marker='.', label=col)
    axs[1].plot(deform_csv.iloc[:, 0], deform_csv.iloc[:, 2], marker='.', linestyle='-', color='black', label='grip_force', linewidth=0.5)
    axs[1].set_ylim(-30, 25)
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Force (N)')
    axs[1].set_title('Force Components Over Time')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(paths['plot'], dpi=300, bbox_inches='tight')
    print(f"Saved plot -> {paths['plot']}")
    # plt.show()  # Show the plot for immediate feedback
    # plt.close(fig) # Close the figure to free memory

def main(obj_path, target_choice='soft'):
    """
    Main function to run a single simulation instance.
    This function is now a high-level coordinator.
    """
    name = os.path.basename(os.path.dirname(obj_path))
    paths = setup_paths(name, target_choice)

    # Create DataFrames for data logging
    df = pd.DataFrame(columns=["step", "left_fx", "left_fy", "left_fz", "left_tx", "left_ty", "left_tz",
                               "right_fx", "right_fy", "right_fz", "right_tx", "right_ty", "right_tz",
                               "dof_0", "dof_1", "dof_2", "dof_3", "dof_4", "dof_5", "dof_6", "dof_7", "dof_8",])
    deform_csv = pd.DataFrame(columns=["step", "deformations", "grip_force"])

    # Create the simulation environment
    scene, cam, franka, gso_object = create_scene(obj_path)

    # Run the primary motion sequence
    # run_pd_control_sequence(scene, cam, franka, gso_object, df, deform_csv, paths, target_choice)

    # Run secondary motion sequence with rotation
    run_rotation(scene, cam, franka, gso_object, df, deform_csv, paths, target_choice)

    # Save data and generate plots
    df.to_csv(paths['csv'], index=False)
    print(f"Saved data -> {paths['csv']}")
    deform_csv.to_csv(paths['deform_csv'], index=False)
    print(f"Saved deform data -> {paths['deform_csv']}")


# def get_incomplete_objects(base_path, names, material="Elastic", targets=['soft', 'medium', 'hard']):
#     incomplete = []

#     for name in names:
#         obj_dir = os.path.join(base_path,'main','data','picked_up','csv', name, material)

#         # If the base object directory doesn't exist, it's incomplete
#         if not os.path.isdir(obj_dir):
#             for t in targets:
#                 incomplete.append((name, t))
#                 print(f'{name}, {t} is not done')
#             continue

#         # Check each target
#         for target in targets:
#             target_path = os.path.join(obj_dir, target)
#             if not os.path.isdir(target_path):
#                 incomplete.append((name, target))
#                 print(f'{name}, {t} is not done')

#     return incomplete


def get_incomplete_objects(base_path, names, material="Elastic", targets=['soft', 'medium', 'hard']):
    """
    Checks and prints the completion status for all objects and targets.

    An item is "incomplete" if its target directory is non-existent or empty.
    An item is "completed" if its target directory exists and is not empty.

    Args:
        base_path (str): The base directory path.
        names (list): A list of object names to check.
        material (str, optional): The material type. Defaults to "Elastic".
        targets (list, optional): A list of target conditions. Defaults to ['soft', 'medium', 'hard'].

    Returns:
        list: A list of tuples, where each tuple contains the name and target
              of an incomplete object.
    """
    incomplete = []

    for name in names:
        obj_dir = os.path.join(base_path, 'main', 'data', 'picked_up', 'csv', name, material)

        # If the base object directory doesn't exist, all its targets are incomplete.
        if not os.path.isdir(obj_dir):
            for t in targets:
                incomplete.append((name, t))
                print(f'❌ {name}, {t} is not done')
            continue

        # Check each target subdirectory.
        for target in targets:
            target_path = os.path.join(obj_dir, target)
            
            # Check if the directory is missing or empty.
            if not os.path.isdir(target_path) or not os.listdir(target_path):
                incomplete.append((name, target))
                print(f'❌ {name}, {target} is not done')
            else:
                # Otherwise, it's completed.
                print(f'✅ {name}, {target} is completed')

    return incomplete


if __name__ == "__main__":
    folder_path = os.path.join(BASE_PATH, "data", "mujoco_scanned_objects", "models")
    all_files = os.listdir(folder_path)
    # selected_files = random.sample(all_files, 1)
    selected_files = "11pro_SL_TRX_FG"
    
    # The 'force' argument seems unused in the PD control logic, so it's set to a placeholder.
    # If it were used, it would be passed to main().
    processes = []
    
    for obj_name in selected_files:
        obj_path = os.path.join(folder_path, obj_name, "model.obj")
        print(f"Processing object: {obj_path}")
        for target_choice in ['hard', 'medium', 'soft']:
            print(f"--- Starting process for {obj_name} with target: {target_choice} ---")
            p = Process(target=main, args=(obj_path, target_choice))
            p.start()
            processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()
        
    print("All simulations completed.")


# if __name__ == "__main__":
#     folder_path = os.path.join(BASE_PATH, "data", "mujoco_scanned_objects", "models")
#     all_files = os.listdir(folder_path)

#     selected_files = get_incomplete_objects(BASE_PATH, all_files)
#     processes = []

#     for task in selected_files:
#         obj_name, target_choice = task
#         obj_path = os.path.join(folder_path, obj_name, "model.obj")
#         print(f"Processing object: {obj_path}")

#         print(f"--- Starting process for {obj_name} with target: {target_choice} ---")

#         # Wait until a slot is free
#         while len(processes) >= 8:
#             # Remove finished processes from the list
#             processes = [p for p in processes if p.is_alive()]
#             time.sleep(0.1)  # Avoid CPU overuse

#         # Start a new process
#         p = Process(target=main, args=(obj_path, target_choice))
#         p.start()
#         processes.append(p)

#     # Wait for all remaining processes to complete
#     for p in processes:
#         p.join()

#     print("All simulations completed.")
#     print("The objects processed are:", selected_files)
