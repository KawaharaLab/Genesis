########################## import ##########################
import argparse
import sys
import numpy as np
import random
import genesis as gs
import pandas as pd
import os
import master_movement as mm
import matplotlib.pyplot as plt
from make_step import make_step # importing make_step from main file including all relevant functions

base_path = "/Users/no.166/Documents/Azka\'s Workspace/Genesis"

import imageio.v3 as iio
"""https://pypi.org/project/imageio/"""

####################### pre-settings #########################

photo_interval = 250
material_type = "Elastic"
frc = 0.5 #0.5, 1.5, 2.5
scl = 0.6 # scaling factor for the object
current_force = 1.0 # initial force for grasping, will be adjusted during the process

####################### object path ##########################
photo_path = "/tmp/photos/"

####################### record csv ##########################

def main(frc_arg, obj_path, target_choice='soft'):

    #obj_path = "/Users/no.166/Documents/Azka's Workspace/Genesis/data/mujoco_scanned_objects/models/adistar_boost_m/model.obj"

    name = obj_path.split('/')[-2]

    photo_path = f'{base_path}/main/data/photos/{name}/{material_type}/{frc_arg}/'
    os.makedirs(photo_path, exist_ok=True)
    os.makedirs(f'{base_path}/main/data/csv/{name}', exist_ok=True)
    csv_path = f'{base_path}/main/data/csv/{name}/{name}_{material_type}_{frc_arg}N.csv'
    deform_csv_path = f'{base_path}/main/data/csv/{name}/{name}_{material_type}_deform_{frc_arg}N.csv'
    with open(csv_path, 'w'):
        pass
    with open(deform_csv_path, 'w'):
        pass

    parser = argparse.ArgumentParser()
    os.makedirs(f'{base_path}/main/data/videos/{name}', exist_ok=True)
    parser.add_argument("-v", "--video", default=f'main/data/videos/{name}/{name}_{material_type}_{target_choice}.mp4')
    parser.add_argument("-o", "--outfile", default=f'main/data/csv/{name}/{name}_{material_type}_{target_choice}N.csv')
    args = parser.parse_args()

    # Create two DataFrames to record force/torque + DOF states and deformations
    df = pd.DataFrame(columns=["step", "left_fx", "left_fy", "left_fz", "left_tx", "left_ty", "left_tz",
                           "right_fx", "right_fy", "right_fz", "right_tx", "right_ty", "right_tz",
                           "dof_0", "dof_1", "dof_2", "dof_3", "dof_4", "dof_5", "dof_6", "dof_7", "dof_8"])
    deform_csv = pd.DataFrame(columns=["step","deformations"])

    # Test grip_force dataframe
    grip_force_df = pd.DataFrame(columns=["step", "grip_force"])

    ########################## init ##########################
    # gs.init(backend=gs.cpu, debug=True, logging_level="error")
    gs.init(backend=gs.cpu)

    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=30,
        max_FPS=60,
    )
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=15,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=30,
        ),
        show_viewer=False,
        vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=True,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(0.0, -0.1, -0.05),
            upper_bound=(0.75, 1.0, 1.0),
            grid_density=128,
        ),
    )
    ########################## camera ##########################
    cam = scene.add_camera(
        res=(1280, 720),
        pos = (-1.5, 1.5, 0.25), lookat = (0.45, 0.45, 0.4),
        fov=30,
    )
    # --------------------------------------------------------
    ########################## entities ##########################
    plane = scene.add_entity(
        # gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
        gs.morphs.Plane(),
        material=gs.materials.Rigid()
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        material=gs.materials.Rigid(coup_friction=1.0),
    )
    gso_object = scene.add_entity(
        # material=gs.materials.Rigid(),
        material=gs.materials.MPM.Elastic(
            E=1.5e6,
            nu=0.45,
            rho=1100.0,
            lam=None,
            mu=None,
            sampler="pbs",
            model="corotation",
        ), #Rubber
        #     E=2500,
        #     nu=0.499,
        #     rho=920,
        #     sampler="pbs",
        #     model="neohooken"
        # ),
        # def __init__(
        #         self,
        #         E=3e5,
        #         nu=0.2,
        #         rho=1000.0,
        #         lam=None,
        #         mu=None,
        #         sampler="pbs",
        #         model="corotation",
        #     ):

        # material=gs.materials.MPM.ElastoPlastic( #PET
        #     E=2.45e6,
        #     nu=0.4,
        #     rho=1400,
        #     use_von_mises=True,
        #     von_mises_yield_stress=18000,
        # ),
        # material=gs.materials.MPM.ElastoPlastic( #PP
        #     E=2.0e6,
        #     nu=0.42,
        #     rho=900,
        #     use_von_mises=True,
        #     von_mises_yield_stress=33000,
        # ),
        #material=gs.materials.MPM.ElastoPlastic( #aluminum
        #    E=69e6,
        #    nu=0.33,
        #    rho=2700,
        #    use_von_mises=True,
        #    von_mises_yield_stress=95000,
        #),
        #material=gs.materials.MPM.ElastoPlastic( #steel
        #    E=180e6,
        #    nu=0.25,
        #    rho=7860,
        #    use_von_mises=True,
        #    von_mises_yield_stress=502000,
        #),
        morph=gs.morphs.Mesh(
            file= obj_path,
            scale=scl, #record
            pos=(0.45, 0.45, 0),
            euler=(0, 0, 90), #record
        ),
    )
    
    ########################## build ##########################
    scene.build()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    # Optional: set control gains
    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )
    end_effector = franka.get_link("hand")

    # ------ previous method of getting top of object (Rigid) ------
    # print("BBBB",gso_object.get_AABB())
    # lower_obj_bound, upper_obj_bound = gso_object.get_AABB()
    # print("AAAA",upper_obj_bound[2].item())

    # ------ new method of getting top of object (MPM) ------
    state = gso_object.get_state()
    particle_positions_np = state.pos.detach().cpu().numpy()
    # lower_obj_bound = np.min(particle_positions_np[0], axis=0) # not needed for grasping
    upper_obj_bound = np.max(particle_positions_np[0], axis=0)

    # move to pre-grasp pose
    x = 0.45
    y = 0.45
    z = upper_obj_bound[2] + 0.08


    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([x, y, z]),
        quat=np.array([0, 1, 0, 0]),
    )
    qpos[-2:] = 0.04
    franka.set_dofs_position(qpos[:-2], motors_dof)
    franka.set_dofs_position(qpos[-2:], fingers_dof)
    cam.start_recording()

    #################### call functions for motion #####################
    set = 'PD_control_1' # set to False to run the full 1000 steps
    if set == 'quicklook':
        # 1-100 steps
        print("Moving to pre-grasp pose...")
        mm.move_to_pose(
        scene=scene,
        cam=cam,
        franka=franka,
        gso_object=gso_object,
        df=df,
        deform_csv=deform_csv,
        photo_path=photo_path,
        photo_interval=photo_interval,
        name=name,
        qpos=qpos,
        motors_dof=motors_dof,
        fingers_dof=fingers_dof,
        steps=25
        )
        # 100-125 steps
        # buffer time to complete motion
        print("Completing motion...")
        for _ in range(25):
            make_step(scene, cam, franka, df, photo_path, photo_interval, gso_object, deform_csv, name)
    elif set == 'PD_control_1':
        # velocity limits for Elastic
        soft_vel_limit = 0.0002
        medium_vel_limit = 0.0006
        hard_vel_limit = 0.0011

        # velocity limits for ElastoPlastic
        # soft_vel_limit = 0.0010
        # medium_vel_limit = 0.0030
        # hard_vel_limit = 0.0060

        if target_choice == 'soft':
            target_vel = soft_vel_limit
        elif target_choice == 'medium':
            target_vel = medium_vel_limit
        elif target_choice == 'hard':
            target_vel = hard_vel_limit

        # 0-50 steps
        print("Moving to pre-grasp pose...")
        mm.move_to_pose(
        scene=scene,
        cam=cam,
        franka=franka,
        gso_object=gso_object,
        df=df,
        deform_csv=deform_csv,
        photo_path=photo_path,
        photo_interval=photo_interval,
        name=name,
        qpos=qpos,
        motors_dof=motors_dof,
        fingers_dof=fingers_dof,
        steps=50
        )

        # 50-100 steps
        print("######################### Descending to object... #########################")
        mm.descend_to_object(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                        end_effector, x, y, z, motors_dof, fingers_dof, steps=50)
        
        current_force = 3.0 # initial force for grasping, will be adjusted during the process
        # record initial grip force for first 100 steps
        for i in range(1, 101):
            grip_force_df.loc[len(grip_force_df)] = [i, 0]

        # 100-300 steps
        print("######################### Grasping object with PD control... #########################")
        for i in range(200):
            # curr_force = current_force
            mm.grasp_object(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                    end_effector, x, y, z, motors_dof, fingers_dof, grasp=True, grip_force =-current_force, steps=1)
            current_deformation = deform_csv.iloc[-1, 1]  # get the last deformation value
            deform_velocity = deform_csv.iloc[-1, 1] - deform_csv.iloc[-2,1]  # get the last two deformation values
            
            if i % 2 == 0:  # every few steps, adjust the force
                if deform_velocity > 1.1*target_vel:
                    # print(f"At step {deform_csv.iloc[-1,0]}. Velocity {deform_velocity} exceeded target {target_vel}. Reducing force.")
                    current_force -= 0.25
                    action = "Reducing force"
                elif deform_velocity < 0.9*target_vel:
                    # print(f"At step {deform_csv.iloc[-1,0]}. Velocity {deform_velocity} is below target {target_vel}. Increasing force.")
                    current_force += 0.25
                    action = "Increasing force"
                else:
                    # print(f"At step {deform_csv.iloc[-1,0]}. Velocity {deform_velocity} is within target {target_vel}. Constant force.")
                    action = "None"
                    pass
            # record current grip force
            print(
                f"At step {int(deform_csv.iloc[-1, 0]):>4}. "
                f"Force: {current_force:>6.2f} | "
                f"Deformation velocity: {deform_velocity:>14.8f} | "
                f"Target velocity: {target_vel:>14.8f} | "
                f"Deformation: {current_deformation:>7.5f} | "
                f"Adjustment: {action}"
            )
            grip_force_df.loc[len(grip_force_df)] = [deform_csv.iloc[-1, 0], -current_force]

        # 300-500 steps
        print("######################### Lifting object with PD control... #########################")
        for i in range(200):
            curr_z = upper_obj_bound[2] + 0.08 + (i * 0.00075)  # increment z position
            mm.lift_object(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                            end_effector, x, y, curr_z, motors_dof, fingers_dof, grip_force=-current_force, steps=1)
            current_deformation = deform_csv.iloc[-1, 1]  # get the last deformation value
            deform_velocity = deform_csv.iloc[-1, 1] - deform_csv.iloc[-2,1]  # get the last two deformation values
            if i % 2 == 0:  # every few steps, adjust the force
                if deform_velocity > 1.1*target_vel:
                    # print(f"At step {deform_csv.iloc[-1,0]}. Velocity {deform_velocity} exceeded target {target_vel}. Reducing force.")
                    current_force -= 0.25
                    action = "Reducing force"
                elif deform_velocity < 0.9*target_vel:
                    # print(f"At step {deform_csv.iloc[-1,0]}. Velocity {deform_velocity} is below target {target_vel}. Increasing force.")
                    current_force += 0.25
                    action = "Increasing force" 
                else:
                    # print(f"At step {deform_csv.iloc[-1,0]}. Velocity {deform_velocity} is within target {target_vel}. Constant force.")
                    action = "None"
                    pass
            # record current grip force
            print(
                f"At step {int(deform_csv.iloc[-1, 0]):>4}. "
                f"Force: {current_force:>6.2f} | "
                f"Deformation velocity: {deform_velocity:>14.8f} | "
                f"Target velocity: {target_vel:>14.8f} | "
                f"Deformation: {current_deformation:>7.5f} | "
                f"Adjustment: {action}"
            )
            grip_force_df.loc[len(grip_force_df)] = [deform_csv.iloc[-1, 0], -current_force]
        
        # 500-600 steps
        print("######################### Dropping object... #########################")
        mm.grasp_object(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                    end_effector, x, y, z, motors_dof, fingers_dof, grasp=False, grip_force=-5, steps=100)
        for i in range(500, 601):
            grip_force_df.loc[len(grip_force_df)] = [i, 0]
        
        # 600-700 steps
        print("######################### Completing motion... #########################")
        for _ in range(100):
            make_step(scene, cam, franka, df, photo_path, photo_interval, gso_object, deform_csv, name)
        for i in range(600, 701):
            grip_force_df.loc[len(grip_force_df)] = [i, 0]



    elif set == 'full':
        # 1-100 steps
        print("Moving to pre-grasp pose...")
        mm.move_to_pose(
            scene=scene,
            cam=cam,
            franka=franka,
            gso_object=gso_object,
            df=df,
            deform_csv=deform_csv,
            photo_path=photo_path,
            photo_interval=photo_interval,
            name=name,
            qpos=qpos,
            motors_dof=motors_dof,
            fingers_dof=fingers_dof,
            steps=100
        )

        # 100-300 steps
        print("Descending to object...")
        mm.descend_to_object(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                        end_effector, x, y, z, motors_dof, fingers_dof)

        # 300-500 steps
        print("Grasping object...")
        mm.grasp_object(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                    end_effector, x, y, z, motors_dof, fingers_dof, grasp=True, grip_force =-frc_arg)

        # 500-700 steps
        print("Lifting object...")
        mm.lift_object(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                    end_effector, x, y, z, motors_dof, fingers_dof, grip_force=-frc_arg)
        
        # 700-900 steps
        # added drop_object function call
        print("Dropping object...")
        mm.grasp_object(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                    end_effector, x, y, z, motors_dof, fingers_dof, grasp=False, grip_force=-frc_arg)

        # mm.rotate_robot_by_angle(
        #     scene=scene,
        #     franka=franka,
        #     cam=cam,
        #     df=df,
        #     photo_path=photo_path,
        #     photo_interval=photo_interval,
        #     name=name,
        #     motors_dof=motors_dof,
        #     fingers_dof=fingers_dof,
        #     end_effector=end_effector,
        #     pos=(-1, -1),
        #     angle_degrees=45,
        #     z_offset=z,
        #     steps=1000
        # )

        # 900-1000 steps: TOTAL = 1400 steps
        # buffer time to complete motion
        print("Completing motion...")
        for _ in range(200):
            make_step(scene, cam, franka, df, photo_path, photo_interval, gso_object, deform_csv, name)

    # ---- 追加: 録画終了・保存 -------------------------------
    cam.stop_recording(save_to_filename=args.video, fps=1000)
    print(f"saved -> {args.video}")
    df.to_csv(csv_path, index=False)
    print(f"saved -> {csv_path}")
    deform_csv.to_csv(deform_csv_path, index=False)
    print(f"saved -> {deform_csv_path}")

    # --- SUBPLOTS SETUP ---------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

    # --- FIRST SUBPLOT: Deformation Over Time -----------------------------------
    axs[0].plot(deform_csv.iloc[:, 0], deform_csv.iloc[:, 1], marker='o', color='tab:blue', linewidth=0.5)
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Deformation Metric')
    axs[0].set_ylim(0, 0.6)
    axs[0].set_title(f'Object: {name} | Target: {target_choice}')
    axs[0].grid(True)

    # --- SECOND SUBPLOT: Force Components ---------------------------------------
    force_columns = [
        'left_fx', 'left_fy', 'left_fz',
        'right_fx', 'right_fy', 'right_fz'
    ]

    for col in force_columns:
        axs[1].plot(df['step'], df[col], marker='o', label=col)

    axs[1].plot(grip_force_df['step'], grip_force_df['grip_force'], 
            marker='o', linestyle='-', color='black', label='grip_force', linewidth=0.5)
    
    # Update y-axis range
    # combined_min = min(df[force_columns].min().min(), grip_force_df['grip_force'].min())
    # combined_max = max(df[force_columns].max().max(), grip_force_df['grip_force'].max())
    # combined_min = df[force_columns].min().min()
    # combined_max = df[force_columns].max().max()
    # axs[1].set_ylim(combined_min - 5, combined_max + 5)
    axs[1].set_ylim(-30, 25)

    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Force (N)')
    axs[1].set_title('Force Components Over Time')
    axs[1].grid(True)
    axs[1].legend()

    # --- FINALIZE & SAVE --------------------------------------------------------
    plt.tight_layout()

    # Save combined figure with custom name (or adjust as needed)
    plot_path = csv_path.replace('.csv', f'_combine_{target_choice}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"plot saved -> {plot_path}")

    # Optional: show the figure
    plt.show()

from multiprocessing import Process

if __name__ == "__main__":

    folder_path = "/Users/no.166/Documents/Azka's Workspace/Genesis/data/mujoco_scanned_objects/models/"
    all_files = os.listdir(folder_path)
    selected_files = random.sample(all_files, 1)
    
    print("Available attributes in master_movement:", dir(mm))
    frc_values = [999]
    # [1,3,5,8,10,20,30,50,100]
    processes = []
    for obj in selected_files:
        # obj_path = os.path.join(folder_path, obj, "model.obj")
        obj_path = "/Users/no.166/Documents/Azka's Workspace/Genesis/data/mujoco_scanned_objects/models/Office_Depot_HP_96_Remanufactured_Ink_Cartridge_Black/model.obj"
        print(f"Processing object: {obj_path}")
        for frc_arg in frc_values:
            for target_choice in ['hard', 'medium', 'soft']:
                print(f"Running with force: {frc_arg}N and target choice: {target_choice}")
                p = Process(target=main, args=(frc_arg,obj_path, target_choice))
                p.start()
                processes.append(p)

    for p in processes:
        p.join()

