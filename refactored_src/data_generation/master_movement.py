# Save as: your-project/src/master_movement.py

import math
import numpy as np
from make_step import make_step # Import the simplified function

def move_to_pose(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                 qpos, motors_dof, fingers_dof, steps=100):
    """Moves the robot to a target joint configuration (qpos) over several steps."""
    for _ in range(steps):
        franka.set_dofs_position(qpos[:-2], motors_dof)
        franka.set_dofs_position(qpos[-2:], fingers_dof)
        # --- CORRECTED CALL ---
        make_step(
            scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
            photo_path=photo_path, photo_interval=photo_interval, gso_object=gso_object, name=name
        )

def descend_to_object(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                      end_effector, x, y, z, motors_dof, fingers_dof,
                      quat=np.array([0, 1, 0, 0]), steps=200, gripper_opening=0.04):
    """Moves the end-effector down to a target position (x,y,z)."""
    qpos = franka.inverse_kinematics(link=end_effector, pos=np.array([x, y, z]), quat=quat)
    qpos[-2:] = gripper_opening
    
    for _ in range(steps):
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(qpos[-2:], fingers_dof)
        # --- CORRECTED CALL ---
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
        # --- CORRECTED CALL ---
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
    # --- CORRECTED CALL ---
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

    for i in range(steps):
        angle_rad_step = angle_rad_total / steps
        step_change = np.zeros(9)
        step_change[joint_index - 1] = angle_rad_step # joint_index is 1-based
        next_qpos = q_start + step_change
        
        franka.control_dofs_position(next_qpos[:-2], motors_dof)
        franka.control_dofs_force(np.array([gripper_force, gripper_force]), fingers_dof)
        # --- CORRECTED CALL ---
        make_step(
            scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
            photo_path=photo_path, photo_interval=photo_interval, gso_object=gso_object, name=name,
            gripper_force=gripper_force
        )
        q_start = next_qpos