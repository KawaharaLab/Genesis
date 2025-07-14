from make_step import make_step # importing make_step from main file including all relevant functions
import math
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
            # print(f"grip_frc = {grip_frc}")
        else:
            # open the gripper
            grip_frc = -grip_force/steps*i
            # print(f"grip_frc = {grip_frc}")
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
    

def rotate_robot_by_angle(scene, cam, df, photo_path, photo_interval, name,
                          franka, motors_dof, fingers_dof, end_effector, gso_object, deform_csv,
                           angle_degrees, z_obj, gripper_force=-5.0, steps=100):
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
    z = z_obj + 0.15 # height gained in lift = 0.15
    # print(str(angle_degrees) + " or " + str(angle_degrees))
    theta = math.radians(angle_degrees+45)  # convert degrees to radians if necessary
    r = 0.6369  # or any other length
    #print(theta)
    x = (r * math.cos(theta))
    y = (r * math.sin(theta))
    #print(f"x = {x}, y = {y}")

    quat = np.array([0, 1, 0, 0])

    #print("ZVAL"+str(z))
    q_start = franka.get_qpos()
    q_target = franka.inverse_kinematics(
          link=end_effector,
          pos=np.array([x, y, z]),
          quat=quat
    )
    for i in range(1000):
        alpha = (i / 900) * 0.8 # interpolation factor from 0 to 1
        q_interp = (1 - alpha) * q_start + alpha * q_target  # linear interpolation
        #print(f"(1 - {alpha}) * {q_start} + {alpha} * {q_target}")
        franka.control_dofs_position(q_interp[:-2], motors_dof)
        franka.control_dofs_force(np.array([gripper_force, gripper_force]), fingers_dof)
        make_step(scene, cam, franka, df, photo_path, photo_interval, gso_object, deform_csv, name)

# def rotate_robot_by_angle_2(scene, cam, df, photo_path, photo_interval, name,
#                            franka, motors_dof, fingers_dof, end_effector, gso_object, deform_csv,
#                            angle_degrees, z_obj=0.08, gripper_force=-5.0, steps=100):
#     """ Rotates the Franka robot around the Z axis by a given angle (in degrees) and interpolates joint positions."""
    
#     z = z_obj + 0.1 # 0.1 height gained in lift
#     r = 0.6369  # or any other length
#     theta = math.radians(angle_degrees + 45)  # convert degrees to radians
#     x = (r * math.cos(theta))
#     y = (r * math.sin(theta))

#     quat = np.array([0, 1, 0, 0])

#     q_target = franka.inverse_kinematics(
#         link=end_effector,
#         pos=np.array([x, y, z]),
#         quat=quat
#     )
#     for i in range(steps):
#         franka.control_dofs_position(q_target[:-2], motors_dof) 
#         franka.control_dofs_force(np.array([gripper_force, gripper_force]), fingers_dof)
#         make_step(scene, cam, franka, df, photo_path, photo_interval, gso_object, deform_csv, name)



def rotate_end_effector_z(scene, cam, franka, df, photo_path, photo_interval, gso_object, deform_csv, name,
                        end_effector, x, y, z, motors_dof, fingers_dof,
                        angle_degrees, steps=100, gripper_force=-5.0):
    """
    Rotates the end effector in place by updating only the orientation (quaternion)
    around the Z-axis, keeping position (x, y, z) constant.

    Parameters:
        scene: Genesis scene object
        cam: Genesis camera object
        franka: Franka robot instance
        df: DataFrame to record force/torque and DOF states
        photo_path: Path to store images
        photo_interval: Interval of frames to record
        name: Name of the object for naming saved files
        end_effector: The end-effector link
        x, y, z: Fixed position to maintain
        motors_dof: indices of arm DOFs
        fingers_dof: indices of gripper DOFs
        angle_degrees: Total angle to rotate around Z-axis (in degrees)
        steps: Number of interpolation steps
        gripper_force: Gripper holding force applied during rotation
    """

    q_start = franka.get_qpos()  # current full qpos includes gripper
    hand = franka.get_link("hand")
    d,a,b,c = hand.get_quat()
    theta =  (angle_degrees*math.pi*2)/360#math.radians(angle_degrees)
    quat = np.array([math.cos(theta/2.0), a * math.sin(theta/2.0), b * math.sin(theta/2.0), c * math.sin(theta/2.0)])

    q_target = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([x, y, z+0.15]),
        quat=quat
    )
    for i in range(700):
        alpha = (i / 1000) * 0.5
        q_interp = (1 - alpha) * q_start + alpha * q_target  # linear interpolation

        franka.control_dofs_position(q_interp[:-2], motors_dof)
        franka.control_dofs_force(np.array([gripper_force, gripper_force]), fingers_dof)
        make_step(scene, cam, franka, df, photo_path, photo_interval, gso_object, deform_csv, name)
