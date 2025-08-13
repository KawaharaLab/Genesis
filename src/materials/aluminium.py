import argparse
import genesis as gs
import pandas as pd
import torch
from . import sim
import numpy as np
import genesis.utils.geom as gu



def aluminium(object_name, object_euler, object_scale, grasp_pos, object_path, qpos_init, photo_interval, coup_friction=0.5):
    default_video_path, default_outfile_path, base_photo_name = sim.set_path(
                                                                    object_name=object_name,
                                                                    coup_friction=coup_friction,
                                                                    material_type="aluminium",
                                                                )
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", default=default_video_path)
    parser.add_argument("-o", "--outfile", default=default_outfile_path)
    args = parser.parse_args()
    df = pd.DataFrame(columns=["step", "left_fx", "left_fy", "left_fz", "left_tx", "left_ty", "left_tz",
                           "right_fx", "right_fy", "right_fz", "right_tx", "right_ty", "right_tz",
                           "dof_0", "dof_1", "dof_2", "dof_3", "dof_4", "dof_5", "dof_6", "dof_7", "dof_8"])
    ########################## init ##########################
    if torch.cuda.is_available():
        gs.init(backend=gs.gpu)
    else:
        gs.init(backend=gs.cpu)
    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=30,
    )
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=1,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, -1, 1.5),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=30,
        ),
        show_viewer=False,
        profiling_options=gs.options.ProfilingOptions(
            show_FPS= False,
        )
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )
    chips_can = scene.add_entity(
        material=gs.materials.Rigid( #Aluminium
            rho=2700,
            coup_friction=coup_friction,
            friction=coup_friction,
        ),
        morph=gs.morphs.Mesh(
            file=object_path,
            scale=object_scale, #record
            pos=(0.45, 0.45, 0.0),
            euler=object_euler, #record
        ),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        material=gs.materials.Rigid(coup_friction=coup_friction, friction=coup_friction),
    )
        # ---- 追加: オフスクリーンカメラ ------------------------
    # cam = scene.add_camera(
    #     res=(1280, 720),
    #     # X 軸方向からのサイドビュー、Z を 0.1（缶の中心高さ程度）にして水平に
    #     pos=(2.0, 2.0, 0.1),
    #     lookat=(0.0, 0.0, 0.1),
    #     fov=30,
    # )
    # --------------------------------------------------------
    cam = scene.add_camera(res=(1280, 1280))
    degree_z = 90
    theta_z = np.pi * degree_z / 180.0
    R_z = np.array([[np.cos(theta_z), np.sin(theta_z), 0], [-np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
    #z反転
    flip_z = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    R = flip_z @ R_z
    #z反転後、y軸周りにdegree[°]回転
    degree = 15
    theta = np.pi * degree / 180.0
    R_y = np.array([[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]])
    R = R_y @ R
    trans = np.array([0.2, 0, -0.5])
    cam.attach(franka.get_link("hand"), gu.trans_R_to_T(trans, R))
    ########################## build ##########################
    scene.build(n_envs=1)
    sim.control_franka(
        scene=scene,
        cam=cam,
        franka=franka,
        grasp_pos=grasp_pos,
        qpos_init=qpos_init,
        strength=30,
        df=df,
        base_photo_name=base_photo_name,
        photo_interval=photo_interval
    )

    # ---- 追加: 録画終了・保存 -------------------------------
    cam.stop_recording(save_to_filename=args.video, fps=1000/photo_interval)
    print(f"saved -> {args.video}")
    df.to_csv(args.outfile, index=False)
    print(f"saved -> {args.outfile}")
    gs.destroy()
    # --------------------------------------------------------