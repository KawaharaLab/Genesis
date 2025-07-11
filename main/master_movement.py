from make_step import make_step # importing make_step from main file including all relevant functions

import numpy as np


def move_to_pose(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                 qpos, motors_dof, fingers_dof, steps=100):
    """
    Moves the Franka robot to the target joint configuration (qpos) over a number of steps.

    Parameters:
        scene: Genesis scene object
        cam: Genesis camera object
        franka: Franka robot instance
        df: DataFrame to record force/torque and DOF states
        photo_path: Path to store images
        photo_interval: Interval of frames to record
        name: Name of the object for naming saved files
        qpos: Full joint configuration including arm and fingers (length 9)
        motors_dof: indices of arm DOFs
        fingers_dof: indices of gripper DOFs
        steps: Number of simulation steps for smooth motion
    """
    for _ in range(steps):
        franka.set_dofs_position(qpos[:-2], motors_dof)
        franka.set_dofs_position(qpos[-2:], fingers_dof)
        make_step(scene, cam, franka, df, photo_path, photo_interval, gso_object, deform_csv, name)

def descend_to_object(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                      end_effector, x, y, z, motors_dof, fingers_dof,
                      quat=np.array([0, 1, 0, 0]), steps=200, gripper_opening=0.04):
    qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=np.array([x, y, z]),
            quat=quat
        )
    qpos[-2:] = gripper_opening
    
    for i in range(steps):
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(qpos[-2:], fingers_dof)
        make_step(scene, cam, franka, df, photo_path, photo_interval, gso_object, deform_csv, name)

def grasp_object(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                 end_effector, x, y, z, motors_dof, fingers_dof, grasp, grip_force,
                 quat=np.array([0, 1, 0, 0]), steps=200):
    qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=np.array([x, y, z]),
            quat=quat
        )
    for i in range(steps):
        
        if grasp:
            # close the gripper
            franka.control_dofs_position(qpos[:-2], motors_dof)
            # grip_force = -0.025 * i
            grip_frc = grip_force
            print(f"grip_frc = {grip_frc}")
        else:
            # open the gripper
            grip_frc = -grip_force/steps*i
        franka.control_dofs_force(np.array([grip_frc, grip_frc]), fingers_dof)
        make_step(scene, cam, franka, df, photo_path, photo_interval, gso_object, deform_csv, name)

def lift_object(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                end_effector, x, y, z_start, motors_dof, fingers_dof, grip_force,
                quat=np.array([0, 1, 0, 0]), dz=0.00075, steps=200):
    z = z_start
    for _ in range(steps):
        z += dz
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=np.array([x, y, z]),
            quat=quat
        )
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(np.array([grip_force, grip_force]), fingers_dof)
        make_step(scene, cam, franka, df, photo_path, photo_interval, gso_object, deform_csv, name)
    

import math
import numpy as np

def rotate_robot_by_angle(scene, cam, gso_object, df, deform_csv, photo_path, photo_interval, name,
                          franka, motors_dof, fingers_dof, end_effector,
                          pos, angle_degrees, z_offset=0.08, gripper_force=-5.0, steps=100):
    """
    Rotates the Franka robot around the Z axis by a given angle (in degrees) and interpolates joint positions.

    Parameters:
        scene: Genesis scene
        cam: Genesis camera object
        df: pandas DataFrame
        photo_path: Path to save photos
        photo_interval: Interval to save photos
        name: Object name
        franka: Franka robot entity
        motors_dof: DOF indices for robot arm
        fingers_dof: DOF indices for gripper
        end_effector: Link object of end effector
        pos: (x, y) target position of end effector
        angle_degrees: Rotation angle in degrees (e.g., 180, -90, etc.)
        z_offset: Height from surface
        gripper_force: Force to apply during rotation
        steps: Number of interpolation steps
    """
    #x, y = pos
    z = z_offset
    import math
    print(angle_degrees)
    theta = math.radians(angle_degrees + 45)  # convert degrees to radians if necessary
    r = 1  # or any other length
    print(theta)
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    print(f"x = {x}, y = {y}")

    quat_z = np.array([0, 1, 0, 0], dtype=np.float32)

    # Compute IK target joint config
    '''q_target = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([x/2.0, y/2.0, z+0.1]),
        quat=quat_z
    )'''

    print(f"HERE x = {x/2.0}, y = {y/2.0}")

    #q_target[-2:] = 0.04  # gripper opening preserved, but we’ll use force
    '''franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )'''
    # Interpolate joint motion
    for i in range(600):
      q_target = franka.inverse_kinematics(
            link=end_effector,
            pos=np.array([x, y, z+0.1]),
            quat=quat_z
      )
      franka.control_dofs_position(q_target[:-2], motors_dof)
      franka.control_dofs_force(np.array([gripper_force, gripper_force]), fingers_dof)
      make_step(scene, cam, franka, df, photo_path, photo_interval, gso_object, deform_csv, name)
