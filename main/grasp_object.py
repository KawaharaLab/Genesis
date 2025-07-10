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

def main(frc_arg, obj_path, target_choice='medium'):

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
    parser.add_argument("-v", "--video", default=f'main/data/videos/{name}/{name}_{material_type}_{frc_arg}N.mp4')
    parser.add_argument("-o", "--outfile", default=f'main/data/csv/{name}/{name}_{material_type}_{frc_arg}N.csv')
    args = parser.parse_args()

    # Create two DataFrames to record force/torque + DOF states and deformations
    df = pd.DataFrame(columns=["step", "left_fx", "left_fy", "left_fz", "left_tx", "left_ty", "left_tz",
                           "right_fx", "right_fy", "right_fz", "right_tx", "right_ty", "right_tz",
                           "dof_0", "dof_1", "dof_2", "dof_3", "dof_4", "dof_5", "dof_6", "dof_7", "dof_8"])
    deform_csv = pd.DataFrame(columns=["step","deformations"])

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
        material=gs.materials.MPM.Elastic(), #Rubber
        #     E=2500,
        #     nu=0.499,
        #     rho=920,
        #     sampler="pbs",
        #     model="neohooken"
        # ),
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
        steps=25
        )

        # 50-100 steps
        print("Descending to object...")
        mm.descend_to_object(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                        end_effector, x, y, z, motors_dof, fingers_dof, steps=50)
        
        # 100-200 steps
        print("Test grasp low force...")
        test_force_low = 0.5 # low force for testing
        mm.grasp_object(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                    end_effector, x, y, z, motors_dof, fingers_dof, grasp=True, grip_force =-test_force_low, steps=50)
        mm.grasp_object(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                    end_effector, x, y, z, motors_dof, fingers_dof, grasp=False, grip_force =-test_force_low, steps=50)
        low_test_deformation = deform_csv.iloc[-100:, 1].max()  # get the last deformation value
        print(f"Low test deformation: {low_test_deformation}")

        # 200-300 steps
        print("Test grasp higher force...")
        test_force_high = 10.0 # high force for testing
        mm.grasp_object(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                    end_effector, x, y, z, motors_dof, fingers_dof, grasp=True, grip_force =-test_force_high, steps=50)
        mm.grasp_object(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                    end_effector, x, y, z, motors_dof, fingers_dof, grasp=False, grip_force =-test_force_high, steps=50)
        high_test_deformation = deform_csv.iloc[-100:, 1].max()  # get the last deformation value
        print(f"High test deformation: {high_test_deformation}")

        soft_target_max = low_test_deformation * 1.5
        medium_target_max = (low_test_deformation + high_test_deformation) / 2.0
        hard_target_max = high_test_deformation * 0.9

        if target_choice == 'soft':
            target_deform = soft_target_max
        elif target_choice == 'medium':
            target_deform = medium_target_max
        elif target_choice == 'hard':
            target_deform = hard_target_max

        current_force = 1.0 # initial force for grasping, will be adjusted during the process

        # 300-500 steps
        for i in range(200):
            mm.grasp_object(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                    end_effector, x, y, z, motors_dof, fingers_dof, grasp=True, grip_force =-current_force, steps=1)
            current_deformation = deform_csv.iloc[-1, 1]  # get the last deformation value
            if current_deformation > target_deform:
                print(f"At step {deform_csv.iloc[-1,0]}. Deformation {current_deformation} exceeded target {target_deform}. Reducing force.")
                current_force *= 0.5
            else:
                print(f"At step {deform_csv.iloc[-1,0]}. Deformation {current_deformation} is below target {target_deform}. Increasing force.")
                current_force *= 1.2

        # 500-700 steps
        for i in range(200):
            # 500-700 steps
            print("Lifting object...")
            curr_z = upper_obj_bound[2] + 0.08 + (i * 0.00075)  # increment z position
            mm.lift_object(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                        end_effector, x, y, curr_z, motors_dof, fingers_dof, grip_force=-current_force, steps=1)
            current_deformation = deform_csv.iloc[-1, 1]  # get the last deformation value
            if current_deformation > target_deform:
                print(f"At step {deform_csv.iloc[-1,0]}. Deformation {current_deformation} exceeded target {target_deform}. Reducing force.")
                current_force *= 0.5
            else:
                print(f"At step {deform_csv.iloc[-1,0]}. Deformation {current_deformation} is below target {target_deform}. Increasing force.")
                current_force *= 1.2
        
        # 700-800 steps
        # added drop_object function call
        print("Dropping object...")
        mm.grasp_object(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                    end_effector, x, y, z, motors_dof, fingers_dof, grasp=False, grip_force=-5, steps=100)
        

        # 800-1000 steps: TOTAL = 1200 steps
        # buffer time to complete motion
        print("Completing motion...")
        for _ in range(200):
            make_step(scene, cam, franka, df, photo_path, photo_interval, gso_object, deform_csv, name)
        



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
    # --------------------------------------------------------

    # generate a graph of deformation over time
    plt.plot(deform_csv.iloc[:, 0], deform_csv.iloc[:, 1], marker='o')
    plt.xlabel('Time Step')
    plt.ylabel('Deformation Metric')
    plt.title('Deformation Over Time Steps')
    plt.grid(True)
    plt.ylim(0, 0.6)

    plot_path = deform_csv_path.replace('.csv', '.png')  # same filename but with .png
    plt.savefig(plot_path)
    print(f"plot saved -> {plot_path}")

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
        obj_path = "/Users/no.166/Documents/Azka's Workspace/Genesis/data/mujoco_scanned_objects/models/3D_Dollhouse_Refrigerator/model.obj"
        print(f"Processing object: {obj_path}")
        for frc_arg in frc_values:
            p = Process(target=main, args=(frc_arg,obj_path))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

