import dataclasses

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
OPENPI_PATHS = [
    REPO_ROOT / "genesis" / "ext" / "openpi" / "src",
    REPO_ROOT / "genesis" / "ext" / "openpi" / "packages" / "openpi-client" / "src",
]
for candidate in OPENPI_PATHS:
    if candidate.is_dir():
        sys.path.append(str(candidate))

from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config

import argparse
import time
import numpy as np
import genesis as gs
import pandas as pd
import torch
import imageio.v3 as iio

import genesis.utils.geom as gu


# Gripper joint limits used to normalize/denormalize between [0, 1].
GRIPPER_OPEN_POS = 0.0
GRIPPER_CLOSED_POS = 0.8
GRIPPER_RANGE = GRIPPER_CLOSED_POS - GRIPPER_OPEN_POS


def _wrap_angles_rad(angles: np.ndarray) -> np.ndarray:
    """Wrap angles (radians) into [-π, π]."""
    return (angles + np.pi) % (2 * np.pi) - np.pi


def _extract_action_sequence(actions: np.ndarray) -> np.ndarray:
    """Normalize policy action tensors to shape (horizon, action_dim)."""
    array = np.asarray(actions)
    if array.ndim == 3:
        if array.shape[0] == 0:
            return np.empty((0, 0), dtype=np.float32)
        array = array[0]
    elif array.ndim == 2:
        pass
    elif array.ndim == 1:
        array = array.reshape(1, -1)
    else:
        return np.empty((0, 0), dtype=np.float32)
    return np.ascontiguousarray(array, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", default="pi0_ur5e.mp4")
    parser.add_argument("--wrist-video", default="cam_wrist_ur5e.mp4")
    parser.add_argument("-o", "--outfile", default="pi0_ur5e.csv")
    parser.add_argument("--base-fov", type=float, default=60.0)
    # Wider default FOV for wrist cam to capture more context without moving it.
    parser.add_argument("--wrist-fov", type=float, default=80.0)
    parser.add_argument("--base-image", default="base_debug.png")
    parser.add_argument("--wrist-image", default="wrist_debug.png")
    parser.add_argument("--debug", action="store_true")
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
    table_size = (1.8, 1.8, 0.04)
    table_pos = (0.65, 0.0, 0.02)
    table_top_z = table_pos[2] + table_size[2] * 0.5
    table = scene.add_entity(
        material=gs.materials.Rigid(rho=800, friction=1.0),
        morph=gs.morphs.Box(
            size=table_size,
            pos=table_pos,
        ),
        surface=gs.surfaces.Default(color=(1.0, 1.0, 1.0, 1.0)),
    )
    bin_size = (0.25, 0.25, 0.12)
    bin_pos = (0.65, -0.3, table_top_z + bin_size[2] * 0.5)
    bin_container = scene.add_entity(
        material=gs.materials.Rigid(rho=300, friction=1.2),
        morph=gs.morphs.Box(
            size=bin_size,
            pos=bin_pos,
        ),
        surface=gs.surfaces.Default(color=(0.1, 0.4, 0.9, 1.0)),
    )
    cube_size = 0.06
    cube = scene.add_entity(
        material=gs.materials.Rigid(rho=500),
        morph=gs.morphs.Box(
            size=(cube_size, cube_size, cube_size),
            pos=(0.65, 0.0, table_top_z + cube_size * 0.5),
        ),
        surface=gs.surfaces.Default(color=(0.9, 0.1, 0.1, 1.0)),
    )
    ur5e = scene.add_entity(
        gs.morphs.URDF(
            file="src/vla/ur5e_robotiq85/ur5e_robotiq85.urdf",
            fixed=True,
            pos=(0.0, 0.0, table_top_z + 0.05),
        ),
    )
    switch = []
    # ---- カメラ ------------------------
    end_effector = ur5e.get_link("wrist_3_link")

    cam_wrist = scene.add_camera(
        model="thinlens",
        res=(1024, 1024),
        pos=(0.0, 0.0, 0.0),  # overwritten by attach below
        lookat=(0.0, 0.0, 0.0),
        fov=args.wrist_fov,
        GUI=False,
    )
    # Mount wrist camera near the gripper tip with a small forward offset and downward pitch.
    roll = np.deg2rad(-10)   # keep camera upright relative to tool flange
    pitch = np.deg2rad(180)  # 初期位置のグリッパから根本への軸周り
    yaw = np.deg2rad(-90)    # 鉛直上向きの軸周り
    R_roll = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    R = R_yaw @ R_pitch @ R_roll
    trans = np.array([0.14, 0.0, -0.08])  # z: 鉛直下向き
    cam_wrist.attach(end_effector, gu.trans_R_to_T(trans, R))


    # cam = scene.add_camera(
    #     model="thinlens",
    #     res=(1024, 1024),
    #     pos=(-0.6, 0.9, 0.9),
    #     lookat=(0.65, -0.3, 0.0),
    #     fov=args.base_fov,
    #     GUI=False,
    # )    
    cam = scene.add_camera(
        model="thinlens",
        res=(1024, 1024),
        pos=(1.5, 0.0, 0.4),
        lookat=(0.65, 0.0, 0.2),
        fov=args.base_fov,
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
        pos=np.array([[0.68, 0.0, table_top_z + 0.2]]),
        quat=np.array([[0, 1, 0, 0]]),
        dofs_idx_local = motors_dof,
    )
    qpos[0][6] = 0.04
    ur5e.set_dofs_position(qpos[:, :6], motors_dof)
    ur5e.set_dofs_position(qpos[:, 6:7], fingers_dof)
    if args.debug:
        # Render one frame from both cameras and exit for quick framing/fov checks.
        scene.step()
        base_rgb, _, _, _ = cam.render(rgb=True)
        wrist_rgb, _, _, _ = cam_wrist.render(rgb=True)
        iio.imwrite(args.base_image, base_rgb.astype(np.uint8))
        iio.imwrite(args.wrist_image, wrist_rgb.astype(np.uint8))
        print(f"saved debug images -> {args.base_image}, {args.wrist_image}")
        return

    cam.start_recording()
    cam_wrist.start_recording()
    #=======================================================================================================
    # config = _config.get_config("pi0_droid")
    # checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_base")
    config = _config.get_config("pi05_ur3_robotiq")
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_base")
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)
    action_stride = 5
    actions_per_inference = 16
    pending_actions = None
    pending_abs_arm_targets = None
    pending_index = 0

    for t in range (1000):
        if t % 5 == 0:
            rgb, _, _, _ = cam.render(rgb=True)
            rgb_wrist, _, _, _ = cam_wrist.render(rgb=True)
        if t % 80 == 0:
            dofs_state = ur5e.get_dofs_position()[0].cpu().numpy()
            arm = _wrap_angles_rad(dofs_state[:6])
            finger = np.array(dofs_state[6:7])
            finger_normalized = np.clip(finger, GRIPPER_OPEN_POS, GRIPPER_CLOSED_POS) / GRIPPER_RANGE
            state = np.array(arm.tolist() + finger_normalized.tolist())
            images = {
                "cam_high": rgb.astype(np.uint8),
                "cam_left_wrist": rgb_wrist.astype(np.uint8),
            }

            observation = {
                "images": images,
                "state": state.astype(np.float32),
                "prompt": "pick up the red cube and place it on the blue box",
            }
            result = policy.infer(observation)
            action_seq = _extract_action_sequence(result.get("actions"))
            if action_seq.size > 0 and action_seq.shape[1] >= 7:
                horizon = min(actions_per_inference, action_seq.shape[0])
                pending_actions = action_seq[:horizon]
                pending_abs_arm_targets = _wrap_angles_rad(dofs_state[:6][None, :] + pending_actions[:, :6])
                pending_index = 0

        if t % action_stride == 0 and pending_actions is not None and pending_abs_arm_targets is not None:
            if pending_index < len(pending_actions):
                action = pending_actions[pending_index]
                arm_target = pending_abs_arm_targets[pending_index]
                gripper_target = (
                    np.clip(action[6], 0.0, 1.0) * GRIPPER_RANGE + GRIPPER_OPEN_POS
                )
                ur5e.control_dofs_position([arm_target], motors_dof)
                ur5e.control_dofs_position([[gripper_target]], fingers_dof)
                pending_index += 1
        scene.step()
    #================================================================================================================

    # ---- 追加: 録画終了・保存 -------------------------------
    cam.stop_recording(save_to_filename=args.video, fps=20)
    cam_wrist.stop_recording(save_to_filename=args.wrist_video, fps=20)
    print(f"saved -> {args.video}")
    print(f"saved -> {args.wrist_video}")
    df.to_csv(args.outfile, index=False)
    print(f"saved -> {args.outfile}")
    print("switch points:", switch)
    # --------------------------------------------------------
if __name__ == "__main__":
    main()
