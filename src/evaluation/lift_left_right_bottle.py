import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import genesis as gs


DT = 0.01
HOLD_SECONDS = 1.0
HOLD_STEPS = int(HOLD_SECONDS / DT)
SIM_HZ = int(round(1.0 / DT))
RENDER_EVERY = 1
SIM_STEP_COUNT = 0
PANDA_XML_PATH = Path("genesis/assets/xml/franka_emika_panda/panda.xml")


def sim_step(scene, cam=None, record=False):
    global SIM_STEP_COUNT
    scene.step()
    if record and cam is not None and (SIM_STEP_COUNT % max(1, RENDER_EVERY) == 0):
        cam.render(rgb=True)
    SIM_STEP_COUNT += 1


def control_pose(franka, motors_dof, fingers_dof, qpos, gripper_opening=None):
    franka.control_dofs_position(qpos[0, :-2], motors_dof)
    if gripper_opening is None:
        franka.control_dofs_position(qpos[0, -2:], fingers_dof)
    else:
        franka.control_dofs_position(np.array([gripper_opening, gripper_opening]), fingers_dof)


def infer_finger_length_from_mjcf(xml_path: Path) -> float:
    """Return finger length (root->tip) in meters from MJCF collision boxes."""
    try:
        root = ET.parse(xml_path).getroot()
        z_max = 0.053  # fallback: known Panda fingertip collision extent
        for default_node in root.findall(".//default"):
            cls = default_node.attrib.get("class", "")
            if not cls.startswith("fingertip_pad_collision_"):
                continue
            geom = default_node.find("geom")
            if geom is None:
                continue
            pos = [float(v) for v in geom.attrib.get("pos", "0 0 0").split()]
            size = [float(v) for v in geom.attrib.get("size", "0 0 0").split()]
            if len(pos) == 3 and len(size) == 3:
                z_max = max(z_max, pos[2] + size[2])
        return float(z_max)
    except Exception:
        return 0.053


def get_aabb_min_max(obj_entity):
    aabb = obj_entity.get_AABB().cpu().numpy()
    aabb = np.asarray(aabb, dtype=float).reshape(-1, 3)
    return aabb[0], aabb[1]


def compute_grasp_heights(franka, hand_idx, left_finger_idx, obj_entity, finger_len, hover_z):
    lower, upper = get_aabb_min_max(obj_entity)
    obj_top = float(upper[2])
    obj_h = float(max(upper[2] - lower[2], 1e-4))

    links_pos = franka.get_links_pos([hand_idx, left_finger_idx])[0].cpu().numpy()
    hand_z = float(links_pos[0, 2])
    root_z = float(links_pos[1, 2])
    hand_to_root_dz = root_z - hand_z
    sign = -1.0 if hand_to_root_dz < 0.0 else 1.0
    hand_to_tip_dz = hand_to_root_dz + sign * finger_len

    root_clearance = max(0.008, 0.15 * obj_h)
    tip_target_depth = min(0.010, 0.20 * obj_h)

    z_for_tip = (obj_top - tip_target_depth) - hand_to_tip_dz
    z_for_root = (obj_top + root_clearance) - hand_to_root_dz
    grasp_hand_z = max(z_for_tip, z_for_root)
    grasp_hand_z = min(grasp_hand_z, hover_z - 0.01)

    clamp_hand_z = max(grasp_hand_z - 0.004, z_for_root)
    safe_retract_z = max(grasp_hand_z + 0.03, obj_top + 0.12)
    lift_z = max(0.30, safe_retract_z + 0.10)
    return grasp_hand_z, clamp_hand_z, safe_retract_z, lift_z


def ik_pose(franka, end_effector, pos, quat=(0, 1, 0, 0)):
    pos_vec = np.asarray(pos, dtype=float).reshape(3)
    quat_vec = np.asarray(quat, dtype=float).reshape(4)
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=pos_vec[None, :],
        quat=quat_vec[None, :],
    )
    return qpos


def move_ee(
    scene, franka, end_effector, motors_dof, fingers_dof, pos, gripper_opening, steps=120, cam=None, record=False
):
    current = franka.get_links_pos([8])[0].cpu().numpy().reshape(3)
    target = np.array(pos, dtype=float)
    for i in range(max(1, steps)):
        alpha = (i + 1) / max(1, steps)
        interp = (1.0 - alpha) * current + alpha * target
        qpos = ik_pose(franka, end_effector, interp)
        control_pose(franka, motors_dof, fingers_dof, qpos, gripper_opening=gripper_opening)
        sim_step(scene, cam=cam, record=record)


def set_gripper(scene, franka, motors_dof, fingers_dof, start_open, end_open, steps=50, cam=None, record=False):
    arm_hold = franka.get_dofs_position(motors_dof)[0].cpu().numpy()
    for i in range(max(1, steps)):
        alpha = (i + 1) / max(1, steps)
        opening = (1.0 - alpha) * start_open + alpha * end_open
        franka.control_dofs_position(arm_hold, motors_dof)
        franka.control_dofs_position(np.array([opening, opening]), fingers_dof)
        sim_step(scene, cam=cam, record=record)


def hold_pose(scene, franka, end_effector, motors_dof, fingers_dof, pos, gripper_opening, steps, cam=None, record=False):
    qpos = ik_pose(franka, end_effector, pos)
    for _ in range(max(1, steps)):
        control_pose(franka, motors_dof, fingers_dof, qpos, gripper_opening=gripper_opening)
        sim_step(scene, cam=cam, record=record)


def pick_hold_return(
    scene,
    franka,
    end_effector,
    motors_dof,
    fingers_dof,
    hand_idx,
    left_finger_idx,
    obj_entity,
    finger_len,
    xy,
    cam=None,
    record=False,
):
    x, y = xy
    hover_z = 0.24
    grasp_z, clamp_z, safe_retract_z, lift_z = compute_grasp_heights(
        franka=franka,
        hand_idx=hand_idx,
        left_finger_idx=left_finger_idx,
        obj_entity=obj_entity,
        finger_len=finger_len,
        hover_z=hover_z,
    )

    move_ee(
        scene, franka, end_effector, motors_dof, fingers_dof, (x, y, hover_z), gripper_opening=0.04, steps=150, cam=cam, record=record
    )
    move_ee(
        scene, franka, end_effector, motors_dof, fingers_dof, (x, y, grasp_z), gripper_opening=0.04, steps=100, cam=cam, record=record
    )
    move_ee(
        scene, franka, end_effector, motors_dof, fingers_dof, (x, y, clamp_z), gripper_opening=0.04, steps=35, cam=cam, record=record
    )

    set_gripper(scene, franka, motors_dof, fingers_dof, start_open=0.04, end_open=0.0, steps=60, cam=cam, record=record)
    move_ee(
        scene, franka, end_effector, motors_dof, fingers_dof, (x, y, safe_retract_z), gripper_opening=0.0, steps=50, cam=cam, record=record
    )

    move_ee(
        scene, franka, end_effector, motors_dof, fingers_dof, (x, y, lift_z), gripper_opening=0.0, steps=120, cam=cam, record=record
    )
    hold_pose(
        scene, franka, end_effector, motors_dof, fingers_dof, (x, y, lift_z), gripper_opening=0.0, steps=HOLD_STEPS, cam=cam, record=record
    )

    move_ee(
        scene, franka, end_effector, motors_dof, fingers_dof, (x, y, grasp_z), gripper_opening=0.0, steps=120, cam=cam, record=record
    )
    set_gripper(scene, franka, motors_dof, fingers_dof, start_open=0.0, end_open=0.04, steps=60, cam=cam, record=record)
    move_ee(
        scene, franka, end_effector, motors_dof, fingers_dof, (x, y, hover_z), gripper_opening=0.04, steps=100, cam=cam, record=record
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="", help="Save camera video if non-empty path is passed.")
    parser.add_argument("--show-viewer", action="store_true", help="Show GUI viewer.")
    parser.add_argument("--object-type", choices=["cube", "bottle"], default="cube")
    parser.add_argument(
        "--video-fps",
        type=int,
        default=0,
        help="Requested video fps. 0 means realtime fps based on dt and render stride.",
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=0,
        help="Render every N sim steps while recording. 0 means auto from --video-fps (or 1).",
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        gs.init(backend=gs.gpu, logging_level="warning")
    else:
        gs.init(backend=gs.cpu, logging_level="warning")

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.6, -1.1, 1.3),
            camera_lookat=(0.55, 0.0, 0.20),
            camera_fov=35,
            max_FPS=60,
        ),
        rigid_options=gs.options.RigidOptions(dt=DT),
        show_viewer=args.show_viewer,
    )

    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

    left_xy = (0.62, -0.14)
    right_xy = (0.62, 0.14)

    if args.object_type == "cube":
        cube_size = 0.06
        cube_z = cube_size * 0.5
        left_obj = scene.add_entity(
            material=gs.materials.Rigid(rho=250, friction=1.0),  # light
            morph=gs.morphs.Box(size=(cube_size, cube_size, cube_size), pos=(left_xy[0], left_xy[1], cube_z)),
            surface=gs.surfaces.Default(color=(0.3, 0.7, 0.9, 1.0)),
        )
        right_obj = scene.add_entity(
            material=gs.materials.Rigid(rho=1500, friction=1.0),  # heavy
            morph=gs.morphs.Box(size=(cube_size, cube_size, cube_size), pos=(right_xy[0], right_xy[1], cube_z)),
            surface=gs.surfaces.Default(color=(0.9, 0.6, 0.3, 1.0)),
        )
    else:
        left_obj = scene.add_entity(
            material=gs.materials.Rigid(rho=250),  # light
            morph=gs.morphs.URDF(
                file="urdf/3763/mobility_vhacd.urdf",
                scale=0.09,
                pos=(left_xy[0], left_xy[1], 0.036),
                euler=(0, 90, 0),
            ),
        )
        right_obj = scene.add_entity(
            material=gs.materials.Rigid(rho=1500),  # heavy
            morph=gs.morphs.URDF(
                file="urdf/3763/mobility_vhacd.urdf",
                scale=0.09,
                pos=(right_xy[0], right_xy[1], 0.036),
                euler=(0, 90, 0),
            ),
        )

    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    cam = scene.add_camera(
        model="thinlens",
        res=(960, 960),
        pos=(1.7, 0.0, 0.7),
        lookat=(0.60, 0.0, 0.16),
        fov=35,
        GUI=False,
    )

    scene.build(n_envs=1)

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    end_effector = franka.get_link("hand")
    hand_idx = next(i for i, link in enumerate(franka.links) if link.name == "hand")
    left_finger_idx = next(i for i, link in enumerate(franka.links) if link.name == "left_finger")
    finger_len = infer_finger_length_from_mjcf(PANDA_XML_PATH)

    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
    franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )

    home_q = ik_pose(franka, end_effector, pos=(0.45, 0.0, 0.28))
    home_q[0, -2:] = 0.04
    franka.set_dofs_position(home_q[0, :-2], motors_dof)
    franka.set_dofs_position(home_q[0, -2:], fingers_dof)

    if args.video:
        if args.render_every > 0:
            render_every = args.render_every
        elif args.video_fps > 0:
            render_every = max(1, int(round(SIM_HZ / args.video_fps)))
        else:
            render_every = 1
        global RENDER_EVERY
        RENDER_EVERY = render_every
        effective_fps = int(round(SIM_HZ / render_every))
        print(f"sim_hz={SIM_HZ}, render_every={render_every}, video_fps={effective_fps}")
        cam.start_recording()
    else:
        render_every = 1
        effective_fps = 0

    for _ in range(120):
        control_pose(franka, motors_dof, fingers_dof, home_q, gripper_opening=0.04)
        sim_step(scene, cam=cam, record=bool(args.video))

    # Left bottle (light)
    pick_hold_return(
        scene,
        franka,
        end_effector,
        motors_dof,
        fingers_dof,
        hand_idx,
        left_finger_idx,
        left_obj,
        finger_len,
        left_xy,
        cam=cam,
        record=bool(args.video),
    )
    # Right bottle (heavy)
    pick_hold_return(
        scene,
        franka,
        end_effector,
        motors_dof,
        fingers_dof,
        hand_idx,
        left_finger_idx,
        right_obj,
        finger_len,
        right_xy,
        cam=cam,
        record=bool(args.video),
    )

    move_ee(
        scene,
        franka,
        end_effector,
        motors_dof,
        fingers_dof,
        (0.45, 0.0, 0.30),
        gripper_opening=0.04,
        steps=140,
        cam=cam,
        record=bool(args.video),
    )

    for _ in range(120):
        control_pose(franka, motors_dof, fingers_dof, ik_pose(franka, end_effector, (0.45, 0.0, 0.30)), gripper_opening=0.04)
        sim_step(scene, cam=cam, record=bool(args.video))

    if args.video:
        cam.stop_recording(save_to_filename=args.video, fps=effective_fps)
        print(f"saved -> {args.video}")

    gs.destroy()


if __name__ == "__main__":
    main()
