import os
import sys
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process

import genesis as gs
import master_movement as mm
from make_step import make_step

# --- Configuration Section ---
# Grouped constants for easier management.
BASE_PATH = "/Users/no.166/Documents/Azka's Workspace/Genesis"
PHOTO_INTERVAL = 250

MATERIAL_TYPE = "Elastic"
# OBJECT_SCALE = 0.62

def setup_paths(name, frc_arg, target_choice):
    """Creates all necessary directories and returns a dictionary of paths."""
    # This function consolidates all path and directory creation logic.
    frc_str = str(frc_arg) # Use a string representation for filenames
    
    # Define base paths for data types
    photo_base = os.path.join(BASE_PATH, 'main', 'data', 'photos', name, MATERIAL_TYPE, frc_str)
    csv_base = os.path.join(BASE_PATH, 'main', 'data', 'csv', name)
    video_base = os.path.join(BASE_PATH, 'main', 'data', 'videos', name)

    # Create directories
    os.makedirs(photo_base, exist_ok=True)
    os.makedirs(csv_base, exist_ok=True)
    os.makedirs(video_base, exist_ok=True)

    # Return a dictionary of all generated paths
    return {
        "photo": photo_base,
        "csv": os.path.join(csv_base, f'{name}_{MATERIAL_TYPE}_{frc_str}N.csv'),
        "deform_csv": os.path.join(csv_base, f'{name}_{MATERIAL_TYPE}_deform_{frc_str}N.csv'),
        "video": os.path.join(video_base, f'{name}_{MATERIAL_TYPE}_{target_choice}.mp4'),
        "plot": os.path.join(csv_base, f'{name}_{MATERIAL_TYPE}_{frc_str}N_combine_{target_choice}.png')
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
    GRIPPER_MAX_DEPTH = 0.050  # 50 mm (Finger length)
    GRIPPER_MIN_HEIGHT = 0.005 # 5 mm (Minimum contact surface)

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

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1e-3, substeps=15),
        viewer_options=gs.options.ViewerOptions(camera_pos=(3, -1, 1.5), camera_lookat=(0.0, 0.0, 0.0), camera_fov=30),
        show_viewer=False,
        vis_options=gs.options.VisOptions(visualize_mpm_boundary=True),
        mpm_options=gs.options.MPMOptions(lower_bound=(0.0, -0.1, -0.05), upper_bound=(0.75, 1.0, 1.0), grid_density=128)
    )
    
    cam = scene.add_camera(res=(1280, 720), pos=(-1.5, 1.5, 0.25), lookat=(0.45, 0.45, 0.4), fov=30)
    
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid())
    
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"), material=gs.materials.Rigid(coup_friction=1.0))
    
    gso_object = scene.add_entity(
        # material=gs.materials.MPM.Elastic(E=1.5e6, nu=0.45, rho=1100.0, sampler="pbs", model="corotation"),
        material=gs.materials.MPM.Elastic(),
        morph=gs.morphs.Mesh(file=obj_path, scale=OBJECT_SCALE, pos=(0.45, 0.45, 0), euler=OBJECT_EULER)
    )
    
    scene.build()
    return scene, cam, franka, gso_object

def adjust_force_with_pd_control(current_force, deform_csv, target_vel):
    """
    Adjusts force based on deformation velocity.
    This function eliminates the duplicated logic from the original script.
    """
    deform_velocity = deform_csv.iloc[-1, 1] - deform_csv.iloc[-2, 1]
    action = "None"
    
    if deform_velocity > 1.1 * target_vel:
        current_force -= 0.25
        action = "Reducing force"
    elif deform_velocity < 0.9 * target_vel:
        current_force += 0.25
        action = "Increasing force"
        
    return current_force, action, deform_velocity

def run_pd_control_sequence(scene, cam, franka, gso_object, df, deform_csv, grip_force_df, paths, target_choice):
    """Runs the entire PD control motion sequence."""
    # This function contains the main robot motion logic.
    name = os.path.basename(os.path.dirname(paths['csv']))
    
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
    x, y, z = 0.45, 0.45, upper_obj_bound[2] + 0.08
    qpos = franka.inverse_kinematics(link=end_effector, pos=np.array([x, y, z]), quat=np.array([0, 1, 0, 0]))
    qpos[-2:] = 0.04
    
    # Velocity limits
    vel_limits = {'soft': 0.0002, 'medium': 0.0006, 'hard': 0.0011}
    target_vel = vel_limits.get(target_choice, 0.0002) # Default to soft

    # Start motion sequence
    cam.start_recording()
    
    # 0-50: Move to pre-grasp
    print("Moving to pre-grasp pose...")
    mm.move_to_pose(scene, cam, franka, gso_object, df, deform_csv, paths['photo'], PHOTO_INTERVAL, name, qpos, motors_dof, fingers_dof, steps=50)

    # 50-100: Descend
    print("######################### Descending to object... #########################")
    mm.descend_to_object(scene, cam, franka, gso_object, df, deform_csv, paths['photo'], PHOTO_INTERVAL, name, end_effector, x, y, z, motors_dof, fingers_dof, steps=50)
    
    current_force = 3.0
    for i in range(1, 101):
        grip_force_df.loc[len(grip_force_df)] = [i, 0]

    # 100-300: Grasp object
    print("######################### Grasping object with PD control... #########################")
    for i in range(200):
        mm.grasp_object(scene, cam, franka, gso_object, df, deform_csv, paths['photo'], PHOTO_INTERVAL, name, end_effector, x, y, z, motors_dof, fingers_dof, grasp=True, grip_force=-current_force, steps=1)
        if i % 2 == 0:
            current_force, action, deform_velocity = adjust_force_with_pd_control(current_force, deform_csv, target_vel)
        
        current_deformation = deform_csv.iloc[-1, 1]
        print(f"Step: {int(deform_csv.iloc[-1, 0]):>4} | Frc: {current_force:>6.2f} | Vel: {deform_velocity:>10.8f} | Tgt Vel: {target_vel:>10.8f} | Deform: {current_deformation:>7.5f} | Adj: {action}")
        grip_force_df.loc[len(grip_force_df)] = [deform_csv.iloc[-1, 0], -current_force]

    # 300-500: Lift object
    print("######################### Lifting object with PD control... #########################")
    for i in range(200):
        curr_z = z + (i * 0.00075)
        mm.lift_object(scene, cam, franka, gso_object, df, deform_csv, paths['photo'], PHOTO_INTERVAL, name, end_effector, x, y, curr_z, motors_dof, fingers_dof, grip_force=-current_force, steps=1)
        if i % 2 == 0:
            current_force, action, deform_velocity = adjust_force_with_pd_control(current_force, deform_csv, target_vel)

        current_deformation = deform_csv.iloc[-1, 1]
        print(f"Step: {int(deform_csv.iloc[-1, 0]):>4} | Frc: {current_force:>6.2f} | Vel: {deform_velocity:>10.8f} | Tgt Vel: {target_vel:>10.8f} | Deform: {current_deformation:>7.5f} | Adj: {action}")
        grip_force_df.loc[len(grip_force_df)] = [deform_csv.iloc[-1, 0], -current_force]
        
    # 500-600: Drop object
    print("######################### Dropping object... #########################")
    mm.grasp_object(scene, cam, franka, gso_object, df, deform_csv, paths['photo'], PHOTO_INTERVAL, name, end_effector, x, y, z, motors_dof, fingers_dof, grasp=False, grip_force=-5, steps=100)
    for i in range(500, 601):
        grip_force_df.loc[len(grip_force_df)] = [i, 0]
    
    # 600-700: Complete motion
    print("######################### Completing motion... #########################")
    for _ in range(100):
        make_step(scene, cam, franka, df, paths['photo'], PHOTO_INTERVAL, gso_object, deform_csv, name)
    for i in range(600, 701):
        grip_force_df.loc[len(grip_force_df)] = [i, 0]

    cam.stop_recording(save_to_filename=paths['video'], fps=1000)
    print(f"Saved video -> {paths['video']}")

def run_rotation(scene, cam, franka, gso_object, df, deform_csv, grip_force_df, paths, target_choice):
    """Runs the entire rotation motion sequence."""
    # This function contains the main robot motion logic.
    name = os.path.basename(os.path.dirname(paths['csv']))
    
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
    x, y, z = 0.45, 0.45, upper_obj_bound[2] + 0.08
    qpos = franka.inverse_kinematics(link=end_effector, pos=np.array([x, y, z]), quat=np.array([0, 1, 0, 0]))
    qpos[-2:] = 0.04
    
    # Velocity limits
    vel_limits = {'soft': 0.0002, 'medium': 0.0006, 'hard': 0.0011}
    target_vel = vel_limits.get(target_choice, 0.0002) # Default to soft

    # Start motion sequence
    cam.start_recording()
    
    # 0-50: Move to pre-grasp
    print("Moving to pre-grasp pose...")
    mm.move_to_pose(scene, cam, franka, gso_object, df, deform_csv, paths['photo'], PHOTO_INTERVAL, name, qpos, motors_dof, fingers_dof, steps=50)

    # 50-100: Descend
    print("######################### Descending to object... #########################")
    mm.descend_to_object(scene, cam, franka, gso_object, df, deform_csv, paths['photo'], PHOTO_INTERVAL, name, end_effector, x, y, z, motors_dof, fingers_dof, steps=50)
    
    current_force = 3.0
    for i in range(1, 101):
        grip_force_df.loc[len(grip_force_df)] = [i, 0]

    # 100-300: Grasp object
    print("######################### Grasping object with PD control... #########################")
    for i in range(200):
        mm.grasp_object(scene, cam, franka, gso_object, df, deform_csv, paths['photo'], PHOTO_INTERVAL, name, end_effector, x, y, z, motors_dof, fingers_dof, grasp=True, grip_force=-current_force, steps=1)
        if i % 2 == 0:
            current_force, action, deform_velocity = adjust_force_with_pd_control(current_force, deform_csv, target_vel)
        
        current_deformation = deform_csv.iloc[-1, 1]
        print(f"Step: {int(deform_csv.iloc[-1, 0]):>4} | Frc: {current_force:>6.2f} | Vel: {deform_velocity: >10.8f} | Tgt Vel: {target_vel: >10.8f} | Deform: {current_deformation:>7.5f} | Adj: {action}")
        grip_force_df.loc[len(grip_force_df)] = [deform_csv.iloc[-1, 0], -current_force]

    # 300-500: Lift object
    print("######################### Lifting object with PD control... #########################")
    for i in range(200):
        curr_z = z + (i * 0.00075)
        mm.lift_object(scene, cam, franka, gso_object, df, deform_csv, paths['photo'], PHOTO_INTERVAL, name, end_effector, x, y, curr_z, motors_dof, fingers_dof, grip_force=-current_force, steps=1)
        if i % 2 == 0:
            current_force, action, deform_velocity = adjust_force_with_pd_control(current_force, deform_csv, target_vel)

        current_deformation = deform_csv.iloc[-1, 1]
        print(f"Step: {int(deform_csv.iloc[-1, 0]):>4} | Frc: {current_force:>6.2f} | Vel: {deform_velocity: >10.8f} | Tgt Vel: {target_vel: >10.8f} | Deform: {current_deformation:>7.5f} | Adj: {action}")
        grip_force_df.loc[len(grip_force_df)] = [deform_csv.iloc[-1, 0], -current_force]
    
    # 500-2000: Rotate robot by angle
    print("######################### Rotating robot by angle... #########################")
    ANGLE_DEGREES = 90  # Example angle for rotation
    mm.rotate_robot_by_angle(scene, cam, df, paths['photo'], PHOTO_INTERVAL, name, franka, motors_dof, fingers_dof, end_effector, gso_object, deform_csv, ANGLE_DEGREES, z_obj=z, gripper_force=-12.0, steps=1000)
    # mm.rotate_end_effector_z(scene, cam, franka, df, paths['photo'], PHOTO_INTERVAL, gso_object, deform_csv, name, end_effector, x, y, z, motors_dof, fingers_dof, ANGLE_DEGREES, steps=500, gripper_force=-5.0)


    # 2000-2100: Drop object
    print("######################### Dropping object... #########################")
    mm.grasp_object(scene, cam, franka, gso_object, df, deform_csv, paths['photo'], PHOTO_INTERVAL, name, end_effector, x, y, z, motors_dof, fingers_dof, grasp=False, grip_force=-5, steps=100)
    for i in range(500, 601):
        grip_force_df.loc[len(grip_force_df)] = [i, 0]
    
    # 2100-2200: Complete motion
    print("######################### Completing motion... #########################")
    for _ in range(100):
        make_step(scene, cam, franka, df, paths['photo'], PHOTO_INTERVAL, gso_object, deform_csv, name)
    for i in range(600, 701):
        grip_force_df.loc[len(grip_force_df)] = [i, 0]

    cam.stop_recording(save_to_filename=paths['video'], fps=1000)
    print(f"Saved video -> {paths['video']}")

def generate_plots(df, deform_csv, grip_force_df, paths, target_choice):
    """Generates and saves the plots for the simulation results."""
    # This function encapsulates all matplotlib plotting logic.
    name = os.path.basename(os.path.dirname(paths['csv']))
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    # Deformation plot
    axs[0].plot(deform_csv.iloc[:, 0], deform_csv.iloc[:, 1], marker='o', color='tab:blue', linewidth=0.5)
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Deformation Metric')
    axs[0].set_ylim(0, 0.6)
    axs[0].set_title(f'Object: {name} | Target: {target_choice}')
    axs[0].grid(True)

    # Force components plot
    force_columns = ['left_fx', 'left_fy', 'left_fz', 'right_fx', 'right_fy', 'right_fz']
    for col in force_columns:
        axs[1].plot(df['step'], df[col], marker='o', label=col)
    axs[1].plot(grip_force_df['step'], grip_force_df['grip_force'], marker='o', linestyle='-', color='black', label='grip_force', linewidth=0.5)
    axs[1].set_ylim(-30, 25)
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Force (N)')
    axs[1].set_title('Force Components Over Time')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(paths['plot'], dpi=300, bbox_inches='tight')
    print(f"Saved plot -> {paths['plot']}")
    plt.show()  # Show the plot for immediate feedback
    # plt.close(fig) # Close the figure to free memory

def main(frc_arg, obj_path, target_choice='soft'):
    """
    Main function to run a single simulation instance.
    This function is now a high-level coordinator.
    """
    name = os.path.basename(os.path.dirname(obj_path))
    paths = setup_paths(name, frc_arg, target_choice)

    # Create DataFrames for data logging
    df = pd.DataFrame(columns=["step", "left_fx", "left_fy", "left_fz", "left_tx", "left_ty", "left_tz",
                               "right_fx", "right_fy", "right_fz", "right_tx", "right_ty", "right_tz",
                               "dof_0", "dof_1", "dof_2", "dof_3", "dof_4", "dof_5", "dof_6", "dof_7", "dof_8"])
    deform_csv = pd.DataFrame(columns=["step", "deformations"])
    grip_force_df = pd.DataFrame(columns=["step", "grip_force"])

    # Create the simulation environment
    scene, cam, franka, gso_object = create_scene(obj_path)

    # Run the primary motion sequence
    # run_pd_control_sequence(scene, cam, franka, gso_object, df, deform_csv, grip_force_df, paths, target_choice)

    # Run secondary motion sequence with rotation
    run_rotation(scene, cam, franka, gso_object, df, deform_csv, grip_force_df, paths, target_choice)

    # Save data and generate plots
    df.to_csv(paths['csv'], index=False)
    print(f"Saved data -> {paths['csv']}")
    deform_csv.to_csv(paths['deform_csv'], index=False)
    print(f"Saved deform data -> {paths['deform_csv']}")
    
    generate_plots(df, deform_csv, grip_force_df, paths, target_choice)


if __name__ == "__main__":
    folder_path = os.path.join(BASE_PATH, "data", "mujoco_scanned_objects", "models")
    all_files = os.listdir(folder_path)
    selected_files = ['3D_Dollhouse_Refrigerator']
    # selected_files = random.sample(all_files, 1)
    
    # The 'force' argument seems unused in the PD control logic, so it's set to a placeholder.
    # If it were used, it would be passed to main().
    frc_values = [999] 
    processes = []
    
    for obj_name in selected_files:
        obj_path = os.path.join(folder_path, obj_name, "model.obj")
        print(f"Processing object: {obj_path}")
        for frc_arg in frc_values:
            for target_choice in ['hard']:
                print(f"--- Starting process for {obj_name} with target: {target_choice} ---")
                p = Process(target=main, args=(frc_arg, obj_path, target_choice))
                p.start()
                processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()
        
    print("All simulations completed.")
    print("The objects processed are: ", selected_files)