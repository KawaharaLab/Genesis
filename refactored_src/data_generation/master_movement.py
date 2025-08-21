# Save as: your-project/src/master_movement.py
import torch
import numpy as np
from make_step import make_step # Import the simplified function

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