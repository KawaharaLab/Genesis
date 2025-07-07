import argparse
import sys
import numpy as np
sys.path.insert(0, "/Users/no.166/Documents/Azka\'s Workspace/Genesis")

import genesis as gs
import pandas as pd
import os

DEBUG = 0
scl = 1

obj_path = "/Users/no.166/Documents/Azka's Workspace/Genesis/data/mujoco_scanned_objects/models/2_of_Jenga_Classic_Game/model.obj"

photo_interval = 2
material_type = "Elastic"

def set_photo_path():
    arr = obj_path.split('/')
    return arr[-2]

name = set_photo_path()

'''if DEBUG:
    photo_path = None
else:
    photo_path = '/Users/no.166/Documents/Azka\'s Workspace/Genesis/data/photos/' + name + '/' + material_type + '/' + str(frc) + '/'
    os.makedirs('/Users/no.166/Documents/Azka\'s Workspace/Genesis/data/photos/' + name + '/' + material_type  + '/' + str(frc), exist_ok=True)
    os.makedirs(f'/Users/no.166/Documents/Azka\'s Workspace/Genesis/data/csv/{name}', exist_ok=True)
    csv_path = f'/Users/no.166/Documents/Azka\'s Workspace/Genesis/data/csv/{name}/{name}_{material_type}_{frc}.csv'
    with open(csv_path, 'w'):
        pass'''

import imageio.v3 as iio
"""https://pypi.org/project/imageio/"""

def make_step(scene, cam, franka, df, photo_path, photo_interval, gso_object, df2):
    """フランカを目標位置に移動させるステップ関数"""
    scene.step()

    # ==================== DEFORMATION CHECK FOR A SPECIFIC OBJECT ====================
    # Check if the object is an MPM entity
    print('check')
    max_deformation = 0.0

    if isinstance(gso_object, gs.engine.entities.MPMEntity):
        # 1. Get all deformation values
        all_deformation_values = scene.sim.mpm_solver.deformation_metric.to_numpy()

        # 2. Get the slice indices for your specific object
        start_idx = gso_object.particle_start
        end_idx = start_idx + gso_object.n_particles

        # 3. Slice the array
        object_deformation = all_deformation_values[:, start_idx:end_idx]

        # 4. Calculate the max deformation for just this object
        max_deformation = object_deformation.max()

        if max_deformation > 0.01:
            print(f"Object deformed at step {scene.t}! Max value: {max_deformation:.4f}")

    t = int(scene.t) - 1
    #cam.render()
    if DEBUG:
        rgb, _, _, _  = cam.render(rgb=True)
    else:
        if t % photo_interval == 0:
            for i in range(1):
                rgb, _, _, _  = cam.render(rgb=True)
                if i == 1:
                    cam.set_pose(pos = (2.1, -1.2, 0.1), lookat = (0.45, 0.45, 0.5))
                elif i == 0:
                    cam.set_pose(pos = (-1.5, 1.5, 0.25), lookat = (0.45, 0.45, 0.4))
                elif i == 2:
                    cam.set_pose(pos = (2, 2, 0.1), lookat = (0, 0, 0.1))
                if photo_path is not None:
                    filepath = photo_path + f"{name}{t:05d}Angle{i}.png"
                    iio.imwrite(filepath, rgb)
        dofs = franka.get_dofs_position()
        dofs = [x.item() for x in dofs]
        links_force_torque = franka.get_links_force_torque([9, 10]) # 手先のlocal_indexは9, 10
        links_force_torque = [x.item() for x in links_force_torque[0]] + [x.item() for x in links_force_torque[1]]
        df.loc[len(df)] = [
            scene.t,
            links_force_torque[0], links_force_torque[1], links_force_torque[2],
            links_force_torque[3], links_force_torque[4], links_force_torque[5],
            links_force_torque[6], links_force_torque[7], links_force_torque[8],
            links_force_torque[9], links_force_torque[10], links_force_torque[11],
            dofs[0], dofs[1], dofs[2], dofs[3], dofs[4], dofs[5], dofs[6], dofs[7], dofs[8]
        ]
        df2.loc[len(df2)] = [
            scene.t,
            max_deformation
        ]
    # #force


def main(frc_arg):

    name = set_photo_path()
    photo_path = f'/Users/no.166/Documents/Azka\'s Workspace/Genesis/data/photos/{name}/{material_type}/{frc_arg}/'
    os.makedirs(photo_path, exist_ok=True)
    os.makedirs(f'/Users/no.166/Documents/Azka\'s Workspace/Genesis/data/csv/{name}', exist_ok=True)
    csv_path = f'/Users/no.166/Documents/Azka\'s Workspace/Genesis/data/csv/{name}/{name}_{material_type}_{frc_arg}.csv'
    csv2_path = f'/Users/no.166/Documents/Azka\'s Workspace/Genesis/data/csv/{name}/deform_nograsp.csv'

    with open(csv_path, 'w'):
        pass
    with open(csv2_path, 'w'):
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", default="data/videos/grasp_can.mp4")
    parser.add_argument("-o", "--outfile", default="data/csv/grasp_can.csv")
    args = parser.parse_args()
    df = pd.DataFrame(columns=["step", "left_fx", "left_fy", "left_fz", "left_tx", "left_ty", "left_tz",
                           "right_fx", "right_fy", "right_fz", "right_tx", "right_ty", "right_tz",
                           "dof_0", "dof_1", "dof_2", "dof_3", "dof_4", "dof_5", "dof_6", "dof_7", "dof_8"])
    df2 = pd.DataFrame(columns=["step","deformations"])

    ########################## init ##########################
    # gs.init(backend=gs.cpu, debug=True, logging_level="error")
    gs.init(backend=gs.cpu)
    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        # camera_pos=(-1.5, 1.5, 0.25),  # X軸方向からのサイドビュー
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=30,
        max_FPS=60,
    )
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=5e-3,
            substeps=35,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            # camera_pos=(-1.5, 1.5, 0.25),  # X軸方向からのサイドビュー
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
            grid_density=64,
        ),
    )
    # ---- 追加: オフスクリーンカメラ ------------------------
    cam = scene.add_camera(
        res=(1280, 720),
        # X 軸方向からのサイドビュー、Z を 0.1（缶の中心高さ程度）にして水平に
        # pos=(2.0, 2.0, 0.1),
        pos=(-1.5, 1.5, 0.25),  # X軸方向からのサイドビュー
        lookat=(0.0, 0.0, 0.1),
        fov=30,
    )
    # --------------------------------------------------------
    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )
    gso_object = scene.add_entity(
        # material=gs.materials.Rigid(),
        material=gs.materials.MPM.Elastic(), #Rubber
        #     E=10200,
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
            pos=(0.45, 0.45, 0.001),
            euler=(0, 0, 90), #record
        ),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        material=gs.materials.Rigid(coup_friction=frc_arg),
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
    # print("BBBB",gso_object.get_AABB())

    # lower_obj_bound, upper_obj_bound = gso_object.get_AABB()
    # print("AAAA",upper_obj_bound[2].item())

    # 1. Get the current state of the object, which includes particle positions
    state = gso_object.get_state()

    # 2. Access the particle positions (this will be a PyTorch tensor)
    particle_positions_np = state.pos.detach().cpu().numpy()

    # 3. Calculate the min and max coordinates along each axis
    # The tensor shape is likely (batch_size, n_particles, 3)
    lower_obj_bound = np.min(particle_positions_np[0], axis=0)
    upper_obj_bound = np.max(particle_positions_np[0], axis=0)
    
    # move to pre-grasp pose
    x = 0.45
    y = 0.45
    z = upper_obj_bound[2] + 0.02

    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([x, y, z]),
        quat=np.array([0, 1, 0, 0]),
    )
    qpos[-2:] = 0.04
    franka.set_dofs_position(qpos[:-2], motors_dof)
    franka.set_dofs_position(qpos[-2:], fingers_dof)
    cam.start_recording()
    #=================この中を調整========================
    # reach
    for i in range (100):
        franka.set_dofs_position(qpos[:-2], motors_dof)
        franka.set_dofs_position(qpos[-2:], fingers_dof)
        make_step(scene, cam, franka, df, photo_path, photo_interval, gso_object, df2)
        #print(chips_can.get_pos())
    for i in range(200):
        #record optimized moments
        #z -= 0.0025
        print('step1')
        # qpos = franka.inverse_kinematics(
        #     link=end_effector,
        #     pos=np.array([x, y, z]),
        #     quat=np.array([0, 1, 0, 0]),
        # )
        qpos[-2:] = 0.04
        franka.control_dofs_position(qpos[:-2], motors_dof)
        # franka.control_dofs_position(qpos[-2:], fingers_dof)
        make_step(scene, cam, franka, df, photo_path, photo_interval, gso_object, df2)
        #print(chips_can.get_pos())
    # grasp
    for i in range(400):
        print('step2')
        # qpos = franka.inverse_kinematics(
        #     link=end_effector,
        #     pos=np.array([x, y, z]),
        #     quat=np.array([0, 1, 0, 0]),
        # )
        franka.control_dofs_position(qpos[:-2], motors_dof)
        # franka.control_dofs_force(np.array([-0.01*i, -0.01*i]), fingers_dof)
        make_step(scene, cam, franka, df, photo_path, photo_interval, gso_object, df2)
        #print(chips_can.get_pos())

    for i in range(200):
        z += 0.0005
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=np.array([x, y, z]),
            quat = np.array([0, 1, 0, 0]),
        )
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(np.array([-5, -5]), fingers_dof)
        make_step(scene, cam, franka, df, photo_path, photo_interval, gso_object, df2)
        #print(chips_can.get_pos())

    for i in range(101):
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(np.array([-5+0.05*i, -5+0.05*i]), fingers_dof)
        make_step(scene, cam, franka, df, photo_path, photo_interval, gso_object, df2)
        #print(chips_can.get_pos())
    # ---- 追加: 録画終了・保存 -------------------------------
    cam.stop_recording(save_to_filename=args.video, fps=1000)
    print(f"saved -> {args.video}")
    df.to_csv(csv_path, index=False)
    print(f"saved -> {csv_path}")
    df2.to_csv(csv2_path, index=False)
    print(f"saved -> {csv2_path}")

    # --------------------------------------------------------
from multiprocessing import Process

if __name__ == "__main__":
    frc_values = [2.5]
    processes = []

    main(frc_arg=2.5)

    # for frc_arg in frc_values:
    #     p = Process(target=main, args=(frc_arg,))
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()

