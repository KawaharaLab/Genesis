import dataclasses

import jax

from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import argparse
import time
import numpy as np
import genesis as gs
import pandas as pd
import torch
import imageio.v3 as iio

import genesis.utils.geom as gu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", default="pi0_ur5e.mp4")
    parser.add_argument("-o", "--outfile", default="pi0_ur5e.csv")
    args = parser.parse_args()
    df = pd.DataFrame(columns=["step", "left_fx", "left_fy", "left_fz", "left_tx", "left_ty", "left_tz",
                           "right_fx", "right_fy", "right_fz", "right_tx", "right_ty", "right_tz"])
    ########################## init ##########################
    # if torch.cuda.is_available():
    #     gs.init(backend=gs.gpu)
    # else:
    #     gs.init(backend=gs.cpu)
    gs.init(backend=gs.gpu, logging_level="ERROR")  # CPU backend for this example

    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=30,
        max_FPS=60,
    )

    scene = gs.Scene(
        viewer_options=viewer_options,
        rigid_options=gs.options.RigidOptions(dt=0.01),
        show_viewer=False,          # ★ GUI を開かない
    )
    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )
    bottle = scene.add_entity(
        material=gs.materials.Rigid(rho=300),
        morph=gs.morphs.URDF(
            file="urdf/3763/mobility_vhacd.urdf",
            scale=0.09,
            pos=(0.65, 0.0, 0.036),
            euler=(0, 90, 0),
        ),
        # visualize_contact=True,
    )
    ur5e = scene.add_entity(
        gs.morphs.URDF(file="src/ur5e_robotiq85/ur5e_robotiq85.urdf", fixed=True),
    )
    switch = []
    # ---- 追加: カメラ ------------------------
    # cam_wrist = scene.add_camera(
    #     model="thinlens",
    #     res=(224, 224),
    #     pos=(-1.4, 0.6, 0.55),
    #     lookat=(0.65, -0.2, 0.2),
    #     fov=30,
    #     GUI=False,
    # )
    end_effector = ur5e.get_link("wrist_3_link")

    # cam = scene.add_camera(
    #     model="thinlens",
    #     res=(224, 224),
    #     pos=(2.5, 0.0, 0.5),
    #     lookat=(0.65, 0.0, 0.2),
    #     fov=30,
    #     GUI=False,
    # )
    # degree_z = 90
    # theta_z = np.pi * degree_z / 180.0
    # R_z = np.array([[np.cos(theta_z), np.sin(theta_z), 0], [-np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
    # #z反転
    # flip_z = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    # R = flip_z @ R_z
    # #z反転後、y軸周りにdegree[°]回転
    # degree = 15
    # theta = np.pi * degree / 180.0
    # R_y = np.array([[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]])
    # R = R_y @ R
    # trans = np.array([0.2, 0, -0.5])
    # cam_wrist.attach(ur5e.get_link("wrist_3_link"), gu.trans_R_to_T(trans, R))
    cam_wrist = scene.add_camera(
        model="thinlens",
        res=(224, 224),
        pos=(-1.2, 1.2, 0.55),
        lookat=(0.65, -0.2, 0.2),
        fov=30,
        GUI=False,
    )

    cam = scene.add_camera(
        model="thinlens",
        res=(224, 224),
        pos=(2.0, 0.4, 0.5),
        lookat=(0.65, 0.0, 0.2),
        fov=30,
        GUI=False,
    )
    # --------------------------------------------------------

    ########################## build ##########################
    scene.build(n_envs=1)

    motors_dof = np.arange(6)
    fingers_dof = np.arange(6, 7)

    # Optional: set control gains
    ur5e.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 100, 100, 100, 100, 100, 100]),
    )
    ur5e.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 10, 10, 10, 10, 10, 10]),
    )
    ur5e.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -100, -100, -100, -100, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 100, 100, 100, 100, 100, 100]),
    )
    qpos = ur5e.inverse_kinematics(
        link=end_effector,
        pos=np.array([[0.65, 0.0, 0.25]]),
        quat=np.array([[0, 1, 0, 0]]),
        dofs_idx_local = motors_dof,
    )
    qpos[0][6] = 0.04
    ur5e.set_dofs_position(qpos[:, :6], motors_dof)
    ur5e.set_dofs_position(qpos[:, 6:7], fingers_dof)
    cam.start_recording()
    cam_wrist.start_recording()
    #=======================================================================================================
    # config = _config.get_config("pi0_droid")
    # checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_base")
    config = _config.get_config("pi0_fast_ur5e")
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_base")

    policy = _policy_config.create_trained_policy(config, checkpoint_dir)
    for t in range (4000):
        if t % 5 == 0:
            rgb, _, _, _ = cam.render(rgb=True)
            rgb_wrist, _, _, _ = cam_wrist.render(rgb=True)
        if t % 40 == 0:
            h_step = -1
            dofs = ur5e.get_dofs_position()[0].cpu().numpy()
            arm = dofs[:6]
            print("arm", arm)
            finger = np.array(dofs[6:7])
            print("rgb", rgb.shape, "rgb_wrist", rgb_wrist.shape, arm.shape, "finger", finger.shape)
            observation = {
                "base_rgb": rgb.astype(np.uint8),
                "wrist_rgb": rgb_wrist.astype(np.uint8),
                "joints": arm,
                "gripper": finger,
                "prompt": "pick up and lift the yellow bottle",
            }
            result = policy.infer(observation)
            print("result:", result["actions"][h_step])
        if t % 5 == 0:
            h_step += 1
        # ur5e.control_dofs_position([result["actions"][h_step][:-1]], motors_dof)
        ur5e.control_dofs_position([result["actions"][h_step][:6]], motors_dof)
        finger_control = result["actions"][h_step][6:7]
        ur5e.control_dofs_position(finger_control, fingers_dof)
        scene.step()
    #================================================================================================================

    # ---- 追加: 録画終了・保存 -------------------------------
    cam.stop_recording(save_to_filename=args.video, fps=20)
    cam_wrist.stop_recording(save_to_filename="cam_wrist_ur5e.mp4", fps=20)
    print(f"saved -> {args.video}")
    df.to_csv(args.outfile, index=False)
    print(f"saved -> {args.outfile}")
    print("switch points:", switch)
    # --------------------------------------------------------
if __name__ == "__main__":
    main()
