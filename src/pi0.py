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
    parser.add_argument("-v", "--video", default="grasp_bottle_world_weaker_contact.mp4")
    parser.add_argument("-o", "--outfile", default="grasp_bottle_world_weaker_contact.csv")
    args = parser.parse_args()
    df = pd.DataFrame(columns=["step", "left_fx", "left_fy", "left_fz", "left_tx", "left_ty", "left_tz",
                           "right_fx", "right_fy", "right_fz", "right_tx", "right_ty", "right_tz"])
    ########################## init ##########################
    # if torch.cuda.is_available():
    #     gs.init(backend=gs.gpu)
    # else:
    #     gs.init(backend=gs.cpu)
    gs.init(backend=gs.cpu, logging_level="ERROR")  # CPU backend for this example

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
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    switch = []
    # ---- 追加: カメラ ------------------------
    cam = scene.add_camera(
        res=(224, 224),
        pos=(-1.4, 0.6, 0.55),
        lookat=(0.65, -0.2, 0.2),
        fov=30,
        GUI=False,
    )
    end_effector = franka.get_link("hand")

    cam_wrist = scene.add_camera(res=(224, 224))
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
    cam_wrist.attach(franka.get_link("hand"), gu.trans_R_to_T(trans, R))
    # --------------------------------------------------------

    ########################## build ##########################
    scene.build(n_envs=1)

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
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([[0.65, 0.0, 0.25]]),
        quat=np.array([[0, 1, 0, 0]]),
    )
    qpos[0][-2:] = 0.04
    franka.set_dofs_position(qpos[:, :-2], motors_dof)
    franka.set_dofs_position(qpos[:, -2:], fingers_dof)
    cam.start_recording()
    cam_wrist.start_recording()
    #=======================================================================================================
    config = _config.get_config("pi0_fast_droid")
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")

    policy = _policy_config.create_trained_policy(config, checkpoint_dir)
    for t in range (400):
        if t % 5 == 0:
            rgb, _, _, _ = cam.render(rgb=True)
            rgb_wrist, _, _, _ = cam_wrist.render(rgb=True)
        if t % 40 == 0:
            h_step = 0
            dofs = franka.get_dofs_position()[0].numpy()
            arm = dofs[:-2]
            print("arm", arm)
            finger = np.array([(dofs[-2] + dofs[-1])/0.08])
            print("rgb", rgb.shape, "rgb_wrist", rgb_wrist.shape, "arm", arm.shape)
            observation = {
                "observation/exterior_image_1_left": rgb.astype(np.uint8),
                "observation/wrist_image_left": rgb_wrist.astype(np.uint8),
                "observation/joint_position": arm,
                "observation/gripper_position": finger,
                "prompt": "pick up the yellow bottle",
            }
            result = policy.infer(observation)
            print("result:", result["actions"][h_step])

        if t % 5 == 0:
            franka.control_dofs_velocity([result["actions"][h_step][:-1]], motors_dof)
            finger_control = 0.04 * result["actions"][h_step][-1]
            franka.control_dofs_position([finger_control, finger_control], fingers_dof)
            h_step += 1
        scene.step()
    #================================================================================================================

    # ---- 追加: 録画終了・保存 -------------------------------
    cam.stop_recording(save_to_filename=args.video, fps=20)
    cam_wrist.stop_recording(save_to_filename="cam_wrist.mp4", fps=20)
    print(f"saved -> {args.video}")
    df.to_csv(args.outfile, index=False)
    print(f"saved -> {args.outfile}")
    print("switch points:", switch)
    # --------------------------------------------------------
if __name__ == "__main__":
    main()
