# Save as: your-project/src/master_movement.py
import torch
import numpy as np
from make_step import make_step # Import the simplified function
import genesis as gs


ELASTIC = 0
if ELASTIC:
    INTERPOLATE = 10
else:
    INTERPOLATE = 1
def set_to_pose(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                 qpos, motors_dof, fingers_dof, steps=1):
    """Moves the robot to a target joint configuration (qpos) over several steps."""
    for _ in range(steps):
        franka.set_dofs_position(qpos[:-2], motors_dof)
        franka.set_dofs_position(qpos[-2:], fingers_dof)
        make_step(
            scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
            photo_path=photo_path, photo_interval=photo_interval, gso_object=gso_object, name=name
        )

def descend_to_object(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                      end_effector, x, y, z, motors_dof, fingers_dof,
                      quat=np.array([0, 1, 0, 0]), steps=300, gripper_opening=0.04):
    """Moves the end-effector down to a target position (x,y,z)."""
    qpos = franka.inverse_kinematics(link=end_effector, pos=np.array([x, y, z]), quat=quat)
    qpos[-2:] = gripper_opening
    path = franka.plan_path(qpos, num_waypoints=steps*INTERPOLATE)

    for waypoint in path:
        franka.control_dofs_position(waypoint)
        make_step(
            scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
            photo_path=photo_path, photo_interval=photo_interval, gso_object=gso_object, name=name
        )

def grasp_object(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                 end_effector, x, y, z, motors_dof, fingers_dof, grasp, grip_force,
                 quat=np.array([0, 1, 0, 0]), steps=1):
    """Applies force to the gripper to grasp or release an object."""
    qpos = franka.inverse_kinematics(link=end_effector, pos=np.array([x, y, z]), quat=quat)
    for i in range(steps):
        gripper_force_step = grip_force if grasp else -grip_force + i * grip_force / steps

        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(np.array([gripper_force_step, gripper_force_step]), fingers_dof)
        return make_step(
            scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
            photo_path=photo_path, photo_interval=photo_interval, gso_object=gso_object, name=name,
            gripper_force=gripper_force_step
        )

def lift_object(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                end_effector, x, y, z, motors_dof, fingers_dof, grip_force,
                quat=np.array([0, 1, 0, 0]), steps=1):
    """Lifts the object vertically from its current position."""
    qpos = franka.inverse_kinematics(link=end_effector, pos=np.array([x, y, z]), quat=quat)
    franka.control_dofs_position(qpos[:-2], motors_dof)
    franka.control_dofs_force(np.array([grip_force, grip_force]), fingers_dof)
    return make_step(
        scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
        photo_path=photo_path, photo_interval=photo_interval, gso_object=gso_object, name=name,
        gripper_force=grip_force
    )

def rotate_single_joint_by_angle(scene, cam, df, deform_csv, photo_path, photo_interval, name,
                                 franka, motors_dof, fingers_dof, gso_object, gripper_force,
                                 angle_degrees, joint_index, steps=100):
    """Directly rotates a single specified robot joint by a given angle over a number of steps."""
    q_start = franka.get_qpos().cpu().numpy()
    angle_rad_total = np.radians(angle_degrees)
    steps *= INTERPOLATE
    for i in range(steps):
        angle_rad_step = angle_rad_total / steps
        step_change = np.zeros(9)
        step_change[joint_index - 1] = angle_rad_step # joint_index is 1-based
        next_qpos = q_start + step_change

        franka.control_dofs_position(next_qpos[:-2], motors_dof)
        franka.control_dofs_force(np.array([gripper_force, gripper_force]), fingers_dof)
        make_step(
            scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
            photo_path=photo_path, photo_interval=photo_interval, gso_object=gso_object, name=name,
            gripper_force=gripper_force
        )
        q_start = next_qpos

def place_object(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                 end_effector, x, y, z, motors_dof, fingers_dof, grip_force,
                 quat=np.array([0, 1, 0, 0]), steps=330):
    """Moves the end-effector to a target position (x,y,z) to place the object."""
    eef_pos = franka.get_links_pos([8]).tolist()[0]
    qpos = franka.inverse_kinematics(link=end_effector, pos=np.array([(x+eef_pos[0])/2, (y+eef_pos[1])/2, eef_pos[2]]), quat=quat)
    path = franka.plan_path(qpos, num_waypoints=300*INTERPOLATE)
    for waypoint in path:
        franka.control_dofs_position(waypoint[:-2], motors_dof)
        make_step(
            scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
            photo_path=photo_path, photo_interval=photo_interval, gso_object=gso_object, name=name,
            gripper_force=grip_force
        )

    qpos = franka.inverse_kinematics(link=end_effector, pos=np.array([x, y, z]), quat=quat)
    path = franka.plan_path(qpos, num_waypoints=400*INTERPOLATE)
    for waypoint in path:
        franka.control_dofs_position(waypoint[:-2], motors_dof)
        make_step(
            scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
            photo_path=photo_path, photo_interval=photo_interval, gso_object=gso_object, name=name,
            gripper_force=grip_force
        )
    finger_pos = franka.get_dofs_position(fingers_dof).cpu().numpy()
    finger_margin = np.array([0.04, 0.04]) - finger_pos
    steps = 50 * INTERPOLATE
    for _ in range(steps):
        finger_pos += finger_margin / steps
        franka.control_dofs_position(finger_pos, fingers_dof)
        make_step(
            scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
            photo_path=photo_path, photo_interval=photo_interval, gso_object=gso_object, name=name,
            gripper_force=grip_force
        )
    for _ in range (70):
        franka.control_dofs_position(qpos[:-2], motors_dof)
        make_step(
            scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
            photo_path=photo_path, photo_interval=photo_interval, gso_object=gso_object, name=name,
            gripper_force=grip_force
        )

def move_to_place_xy(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                 end_effector, x, y, motors_dof, fingers_dof, grip_force,
                 quat=np.array([0, 1, 0, 0]), steps=300):
    """Moves the end-effector to a target position (x,y,z) to place the object."""
    eef_pos = franka.get_links_pos([8]).tolist()[0]
    qpos = franka.inverse_kinematics(link=end_effector, pos=np.array([(x+eef_pos[0])/2, (y+eef_pos[1])/2, eef_pos[2]]), quat=quat)
    path = franka.plan_path(qpos, num_waypoints=steps*INTERPOLATE)
    for waypoint in path:
        franka.control_dofs_position(waypoint[:-2], motors_dof)
        make_step(
            scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
            photo_path=photo_path, photo_interval=photo_interval, gso_object=gso_object, name=name,
            gripper_force=grip_force
        )

def release_object(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                 fingers_dof, grip_force, steps=50):
    finger_pos = franka.get_dofs_position(fingers_dof).cpu().numpy()
    finger_margin = np.array([0.04, 0.04]) - finger_pos
    for _ in range(steps*INTERPOLATE):
        finger_pos += finger_margin / steps
        franka.control_dofs_position(finger_pos, fingers_dof)
        make_step(
            scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
            photo_path=photo_path, photo_interval=photo_interval, gso_object=gso_object, name=name,
            gripper_force=grip_force
        )
    for _ in range (70):
        make_step(
            scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
            photo_path=photo_path, photo_interval=photo_interval, gso_object=gso_object, name=name,
            gripper_force=grip_force
        )

def descend_to_place(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                     end_effector, x, y, z, motors_dof, fingers_dof, grip_force,
                     quat=np.array([0, 1, 0, 0]), steps=400):
    qpos = franka.inverse_kinematics(link=end_effector, pos=np.array([x, y, z]), quat=quat)
    path = franka.plan_path(qpos, num_waypoints=steps*INTERPOLATE)
    for waypoint in path:
        franka.control_dofs_position(waypoint[:-2], motors_dof)
        make_step(
            scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
            photo_path=photo_path, photo_interval=photo_interval, gso_object=gso_object, name=name,
            gripper_force=grip_force
        )

def drop_in_box(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                  end_effector, x, y, z, motors_dof, fingers_dof, grip_force,
                  quat=np.array([0, 1, 0, 0]), steps=330):
    """Hover-drop with the same incremental style as rotate_single_joint_by_angle."""

    # Current joint state and EEF height
    q_start = franka.get_qpos().cpu().numpy()              # (9,)
    eef_pos = franka.get_links_pos([8]).tolist()[0]
    z_hover = eef_pos[2]

    # IK targets (midpoint at current Z, then target XY at same Z)  -- CONVERT TO NUMPY
    q_mid = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([(x + eef_pos[0]) * 0.5, (y + eef_pos[1]) * 0.5, z_hover]),
        quat=quat
    )
    if torch.is_tensor(q_mid):
        q_mid = q_mid.detach().cpu().numpy()

    q_goal = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([x, y, z_hover]),
        quat=quat
    )
    if torch.is_tensor(q_goal):
        q_goal = q_goal.detach().cpu().numpy()

    total_steps = int(steps * INTERPOLATE)
    n1 = max(1, total_steps // 2)
    n2 = max(1, total_steps - n1)

    # --- Segment A: q_start -> q_mid (constant per-tick delta) ---
    delta = (np.asarray(q_mid[:-2]) - np.asarray(q_start[:-2])) / float(n1)
    for _ in range(n1):
        next_qpos = q_start.copy()
        next_qpos[:-2] = next_qpos[:-2] + delta   # now pure NumPy addition
        franka.control_dofs_position(next_qpos[:-2], motors_dof)
        franka.control_dofs_force(np.array([grip_force, grip_force]), fingers_dof)
        make_step(scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
                  photo_path=photo_path, photo_interval=photo_interval,
                  gso_object=gso_object, name=name, gripper_force=grip_force)
        q_start = next_qpos

    # --- Segment B: q_mid -> q_goal (same constant-step logic) ---
    delta = (np.asarray(q_goal[:-2]) - np.asarray(q_start[:-2])) / float(n2)
    for _ in range(n2):
        next_qpos = q_start.copy()
        next_qpos[:-2] = next_qpos[:-2] + delta
        franka.control_dofs_position(next_qpos[:-2], motors_dof)
        franka.control_dofs_force(np.array([grip_force, grip_force]), fingers_dof)
        make_step(scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
                  photo_path=photo_path, photo_interval=photo_interval,
                  gso_object=gso_object, name=name, gripper_force=grip_force)
        q_start = next_qpos

    # # --- Drop: open gripper and settle (same pattern as your place_object) ---
    # finger_pos = franka.get_dofs_position(fingers_dof).cpu().numpy()
    # target_open = np.array([0.04, 0.04], dtype=np.float32)
    # open_steps = 50
    # step_delta = (target_open - finger_pos) / float(max(1, open_steps))
    # for _ in range(open_steps):
    #     finger_pos += step_delta
    #     franka.control_dofs_position(finger_pos, fingers_dof)
    #     make_step(scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
    #               photo_path=photo_path, photo_interval=photo_interval,
    #               gso_object=gso_object, name=name, gripper_force=grip_force)

    for _ in range(50):
        franka.control_dofs_force(np.array([0, 0]), fingers_dof)
        make_step(scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
                  photo_path=photo_path, photo_interval=photo_interval,
                  gso_object=gso_object, name=name, gripper_force=grip_force)

    return True

def shake_in_place(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                   end_effector, motors_dof, fingers_dof, grip_force,
                   amplitude=0.02, cycles=3, steps_per_half=60,
                   quat=np.array([0, 1, 0, 0])):
    """
    Simple in-hand shake: move EEF up/down by `amplitude` (meters) around current Z,
    for `cycles` full cycles. Matches the incremental style of rotate_single_joint_by_angle:
      - fixed per-tick joint delta
      - q_next = q_start + delta; update q_start each tick
      - hold gripper force during motion
    """
    # Current joints and EEF pose
    q_start = franka.get_qpos().cpu().numpy()
    eef_x, eef_y, eef_z0 = franka.get_links_pos([8]).tolist()[0]

    def _segment_to_z(q_from, z_target, n_steps):
        # single IK to target (x,y,z_target), then linear joint steps (no planner)
        q_goal = franka.inverse_kinematics(
            link=end_effector,
            pos=np.array([eef_x, eef_y, z_target]),
            quat=quat
        )
        if torch.is_tensor(q_goal):
            q_goal = q_goal.detach().cpu().numpy()

        delta = (np.asarray(q_goal[:-2]) - np.asarray(q_from[:-2])) / float(max(1, n_steps))
        q_curr = q_from.copy()
        for _ in range(max(1, n_steps)):
            next_qpos = q_curr.copy()
            next_qpos[:-2] = next_qpos[:-2] + delta
            franka.control_dofs_position(next_qpos[:-2], motors_dof)
            franka.control_dofs_force(np.array([grip_force, grip_force]), fingers_dof)
            make_step(
                scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
                photo_path=photo_path, photo_interval=photo_interval,
                gso_object=gso_object, name=name, gripper_force=grip_force
            )
            q_curr = next_qpos
        return q_curr

    q_curr = q_start
    for _ in range(int(cycles)):
        q_curr = _segment_to_z(q_curr, eef_z0 + amplitude, steps_per_half)  # up
        q_curr = _segment_to_z(q_curr, eef_z0 - amplitude, steps_per_half)  # down

    # Return to nominal hover height
    q_curr = _segment_to_z(q_curr, eef_z0, steps_per_half)

    return True

def wiggle_rotation(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                    end_effector, motors_dof, fingers_dof, grip_force,
                    joint_index=7, angle_degrees=20, cycles=3, steps_per_half=60):
    """
    Rotates a single joint back and forth like a 'wrist wiggle' while holding an object.
    """
    assert 1 <= joint_index <= 7, "Use 1-based indexing for joint_index (1-7 inclusive)"
    joint_i = joint_index - 1
    base_qpos = franka.get_qpos().cpu().numpy()
    angle_rad = np.radians(angle_degrees)
    q_left  = base_qpos.copy()
    q_right = base_qpos.copy()
    q_left[joint_i]  -= angle_rad
    q_right[joint_i] += angle_rad

    for cycle in range(cycles):
        for target_q in [q_right, q_left]:
            for i in range(steps_per_half):
                alpha = i / steps_per_half
                interp_q = (1 - alpha) * base_qpos + alpha * target_q
                franka.control_dofs_position(interp_q[:-2], motors_dof)
                franka.control_dofs_force(np.array([grip_force, grip_force]), fingers_dof)
                make_step(
                    scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
                    photo_path=photo_path, photo_interval=photo_interval,
                    gso_object=gso_object, name=name, gripper_force=grip_force
                )
            base_qpos = target_q  # Update base to oscillate cleanly
    return True

def keep_holding(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                 motors_dof, fingers_dof, grip_force, steps=100):
    """Keeps holding the object with current gripper force for a number of steps."""
    position = franka.get_qpos().cpu().numpy()
    for _ in range(steps*INTERPOLATE):
        franka.control_dofs_position(position[:-2], motors_dof)
        franka.control_dofs_force(np.array([grip_force, grip_force]), fingers_dof)
        make_step(
            scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
            photo_path=photo_path, photo_interval=photo_interval,
            gso_object=gso_object, name=name, gripper_force=grip_force
        )