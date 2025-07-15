from make_step import make_step # importing make_step from main file including all relevant functions
import math
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


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
        make_step(scene, cam, franka, df, photo_path, photo_interval, gso_object, deform_csv, name, gripper_force=0.0)

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
        make_step(scene, cam, franka, df, photo_path, photo_interval, gso_object, deform_csv, name, gripper_force=0.0)

def grasp_object(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                 end_effector, x, y, z, motors_dof, fingers_dof, grasp, grip_force,
                 quat=np.array([0, 1, 0, 0]), steps=1):
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
            gripper_force = grip_force
            # print(f"grip_frc = {grip_frc}")
        else:
            # open the gripper
            # gripper_force = -grip_force/steps*i
            gripper_force = -grip_force + i*grip_force/steps
            # print(f"grip_frc = {grip_frc}")
        franka.control_dofs_force(np.array([gripper_force, gripper_force]), fingers_dof)
        return make_step(scene, cam, franka, df, photo_path, photo_interval, gso_object, deform_csv, name, gripper_force)

def lift_object(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                end_effector, x, y, z_start, motors_dof, fingers_dof, grip_force,
                quat=np.array([0, 1, 0, 0]), dz=0.00075, steps=1):
    z = z_start
    for _ in range(steps):
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=np.array([x, y, z]),
            quat=quat
        )
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(np.array([grip_force, grip_force]), fingers_dof)
        return make_step(scene, cam, franka, df, photo_path, photo_interval, gso_object, deform_csv, name, gripper_force=grip_force)

# def rotate_robot_by_angle(scene, cam, df, photo_path, photo_interval, name,
#                           franka, motors_dof, fingers_dof, end_effector, gso_object, deform_csv,
#                           angle_degrees, z_obj, gripper_force=-5.0, steps=500):
#     """
#     Rotates the Franka robot around the Z axis by a given angle in degrees,
#     following a circular path in Cartesian space while maintaining orientation.
#     """
#     # 1. Get the initial state of the end effector using the correct methods
#     initial_pos = end_effector.get_pos()      # Use the suggested get_pos()
#     initial_quat = end_effector.get_quat()    # Get the quaternion separately

#     # 2. Determine the center and radius of rotation from the initial position
#     center_x, center_y = 0.0, 0.0 # Assuming rotation is around the world origin
#     radius = math.sqrt((initial_pos[0] - center_x)**2 + (initial_pos[1] - center_y)**2)
#     start_angle = math.atan2(initial_pos[1] - center_y, initial_pos[0] - center_x)
    
#     # The height is the current Z plus the lift amount from the previous step
#     z = z_obj + 0.15 

#     total_angle_rad = math.radians(angle_degrees)

#     # 3. Iterate through the motion step-by-step
#     for i in range(steps):
#         alpha = i / (steps - 1)  # Interpolation factor from 0 to 1
        
#         # Calculate the new angle for this step
#         current_angle = start_angle + alpha * total_angle_rad
        
#         # Calculate the new (x, y) target on the circular path
#         x = center_x + radius * math.cos(current_angle)
#         y = center_y + radius * math.sin(current_angle)
        
#         # 4. Use inverse kinematics to find joint positions for the new Cartesian target
#         q_target = franka.inverse_kinematics(
#             link=end_effector,
#             pos=np.array([x, y, z]),
#             quat=initial_quat  # Use the initial orientation
#         )
        
#         # If IK solution is valid, control the robot
#         if q_target is not None:
#             franka.control_dofs_position(q_target[:-2], motors_dof)
        
#         # Maintain gripper force
#         franka.control_dofs_force(np.array([gripper_force, gripper_force]), fingers_dof)
        
#         # Advance the simulation
#         make_step(scene, cam, franka, df, photo_path, photo_interval, gso_object, deform_csv, name, gripper_force)

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
    for i in range(steps):
        alpha = (i / 900) * 0.8 # interpolation factor from 0 to 1
        q_interp = (1 - alpha) * q_start + alpha * q_target  # linear interpolation
        #print(f"(1 - {alpha}) * {q_start} + {alpha} * {q_target}")
        franka.control_dofs_position(q_interp[:-2], motors_dof)
        franka.control_dofs_force(np.array([gripper_force, gripper_force]), fingers_dof)
        make_step(scene, cam, franka, df, photo_path, photo_interval, gso_object, deform_csv, name, gripper_force)

def next_rotation(point, axis, angle_degrees):
    # Final position calculated with angle and rotation matrix
    angle_rad = np.radians(angle_degrees)
    axis = axis / np.linalg.norm(axis)  # Normalize the axis
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    cross_product = np.cross(axis, point)
    dot_product = np.dot(axis, point)

    p_rotated = (point * cos_theta) + \
                (cross_product * sin_theta) + \
                (axis * dot_product * (1 - cos_theta))
    return p_rotated

def rotate_robot_by_angle_quat(scene, cam, df, photo_path, photo_interval, name,
                               franka, motors_dof, fingers_dof, end_effector, gso_object, deform_csv,
                               angle_degrees, z_obj, gripper_force=-5.0, steps=100, n=1):
    # franka_qpos = franka.get_qpos()  # 7 angles and 2 gripper DOFs, accessible[0-8]
    # print(f"franka qpos = {franka_qpos}")
    # print(f"franka qpos [0] = {franka_qpos[0]}")
    # print(f"franka qpos [1] = {franka_qpos[1]}")
    # ee_pos = end_effector.get_pos() # position of the hand in x, y, z co-ordinates
    # print(f"End effector pose: {ee_pos}")
    # start_pos = franka_qpos[:-2] 
    # start_quat = end_effector.get_quat()  # quaterion of 'hand' link is (0, 1, 0, 0)
    # print(f"Start position: {start_pos}, Start quaternion: {start_quat}")

    # Rotate about this axis, z-axis from this point
    axis = [0,0,1] 

    q_start = franka.get_qpos().cpu().numpy()  # current full qpos includes gripper
    quat_init = end_effector.get_quat()  # Get the quaternion of the end effector
    start_pos = end_effector.get_pos().cpu().numpy()  # Get the current position of the end effector
    franka_limits = {
        'A1': {'lower': math.radians(-166), 'upper': math.radians(166)},
        'A2': {'lower': math.radians(-101), 'upper': math.radians(101)},
        'A3': {'lower': math.radians(-166), 'upper': math.radians(166)},
        'A4': {'lower': math.radians(-176), 'upper': math.radians(-4)},
        'A5': {'lower': math.radians(-166), 'upper': math.radians(166)},
        'A6': {'lower': math.radians(-1),   'upper': math.radians(215)},
        'A7': {'lower': math.radians(-166), 'upper': math.radians(166)}
    }

    for i in range(steps):
        angle_in = (angle_degrees/steps)*2*math.pi/360  # Interpolate angle
        step_change = np.zeros(9)
        step_change[n-1] = angle_in
        # step_change = np.array([angle_in, 0, 0, 0, 0, 0, 0, 0, 0])
        next_qpos = q_start + step_change  # Example linear motion for testing
        print(next_qpos)
        limits = franka_limits[f'A{n}']
        if next_qpos[n-1] < limits['lower'] or next_qpos[n-1] > limits['upper']:
            print(f"Next position exceeds limits for A{n}: {next_qpos[n-1]} not in [{limits['lower']}, {limits['upper']}]")
            break   # If the next position exceeds limits, stop the rotation

        franka.control_dofs_position(next_qpos[:-2], motors_dof)
        franka.control_dofs_force(np.array([gripper_force, gripper_force]), fingers_dof)
        make_step(scene, cam, franka, df, photo_path, photo_interval, gso_object, deform_csv, name, gripper_force)
        q_start = next_qpos

    # for i in range(steps):
    #     # angle_in = np.radians(angle_degrees * i / 350)  # Interpolate angle
    #     angle_in = angle_degrees/steps
    #     next_pos = next_rotation(start_pos, axis, angle_in)
    #     # next_pos = start_pos + np.array([0, 0.001, 0])  # Example linear motion for testing
    #     print(start_pos)
    #     q_target = franka.inverse_kinematics(
    #         link=end_effector,
    #         pos=np.array(next_pos),
    #         quat=quat_init  # Use the initial orientation
    #     )
    #     start_pos = next_pos  # Update start_pos for the next iteration
    #     franka.control_dofs_position(q_target[:-2], motors_dof)
    #     franka.control_dofs_force(np.array([gripper_force, gripper_force]), fingers_dof)
    #     make_step(scene, cam, franka, df, photo_path, photo_interval, gso_object, deform_csv, name, gripper_force)

    
    

    




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
#         make_step(scene, cam, franka, df, photo_path, photo_interval, gso_object, deform_csv, name, gripper_force)



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
        make_step(scene, cam, franka, df, photo_path, photo_interval, gso_object, deform_csv, name, gripper_force)
