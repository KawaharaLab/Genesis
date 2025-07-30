import argparse
import genesis as gs
import pandas as pd
import torch
from . import sim

def steel(object_name, object_euler, object_scale, grasp_pos, object_path, qpos_init, photo_interval, coup_friction=0.5):
    default_video_path, default_outfile_path, base_photo_name = sim.set_path(
                                                                    object_name=object_name,
                                                                    coup_friction=coup_friction,
                                                                    material_type="steel",
                                                                )
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", default=default_video_path)
    parser.add_argument("-o", "--outfile", default=default_outfile_path)
    args = parser.parse_args()
    df = pd.DataFrame(columns=["step", "left_fx", "left_fy", "left_fz", "left_tx", "left_ty", "left_tz",
                           "right_fx", "right_fy", "right_fz", "right_tx", "right_ty", "right_tz",
                           "dof_0", "dof_1", "dof_2", "dof_3", "dof_4", "dof_5", "dof_6", "dof_7", "dof_8"])
    ########################## init ##########################
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     gs.init(backend=gs.gpu)
    # else:
    #     device = torch.device("cpu")
    #     gs.init(backend=gs.cpu, debug=True)
    device = torch.device("cpu")
    gs.init(backend=gs.cpu, logging_level="debug")
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
    # ---- 追加: オフスクリーンカメラ ------------------------
    cam = scene.add_camera(
        res=(1280, 720),
        # X 軸方向からのサイドビュー、Z を 0.1（缶の中心高さ程度）にして水平に
        pos=(2.0, 2.0, 0.1),
        lookat=(0.0, 0.0, 0.1),
        fov=30,
    )
    # --------------------------------------------------------
    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )
    chips_can = scene.add_entity(
        material=gs.materials.Rigid( #steel
            rho=7860,
            coup_friction=1e-2,
            friction=1e-2,
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

    ########################## build ##########################
    scene.build()
    
    sim.control_franka(
        scene, 
        cam, 
        franka, 
        grasp_pos, 
        qpos_init, 
        df, 
        base_photo_name,
        photo_interval
    )
    # ---- 追加: 録画終了・保存 -------------------------------
    cam.stop_recording(save_to_filename=args.video, fps=1000/photo_interval)
    print(f"saved -> {args.video}")
    df.to_csv(args.outfile, index=False)
    print(f"saved -> {args.outfile}")
    gs.destroy()
    # --------------------------------------------------------