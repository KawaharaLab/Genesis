import numpy as np
from make_step import make_step


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

def grasp_object_position(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                 end_effector, x, y, z, motors_dof, fingers_dof, grip_force,
                 quat=np.array([0, 1, 0, 0]), steps=50):
    """
    Closes the gripper to grasp an object using position control.
    The gripper closes incrementally over a number of steps to ensure a secure grasp.
    0.04 is fully open, 0.0 is fully closed.
    """
    qpos = franka.inverse_kinematics(link=end_effector, pos=np.array([x, y, z]), quat=quat)
    gripper_pos = 0.04
    for i in range(steps*INTERPOLATE):
        gripper_pos -= 0.04 / (steps*INTERPOLATE)
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([gripper_pos, gripper_pos]), fingers_dof)
        make_step(
            scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
            photo_path=photo_path, photo_interval=photo_interval, gso_object=gso_object, name=name,
            gripper_force=grip_force
        )

def lift_object(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                end_effector, x, y, z, motors_dof, fingers_dof, grip_force,
                quat=np.array([0, 1, 0, 0]), steps=1):
    """Lifts the object vertically from its current position."""
    qpos = franka.inverse_kinematics(link=end_effector, pos=np.array([x, y, z]), quat=quat)
    franka.control_dofs_position(qpos[:-2], motors_dof)
    # franka.control_dofs_force(np.array([grip_force, grip_force]), fingers_dof)
    franka.control_dofs_position(np.array([0.0, 0.0]), fingers_dof)  # Ensure gripper is fully closed
    return make_step(
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
        franka.control_dofs_position(np.array([0.0, 0.0]), fingers_dof)  # Ensure gripper is fully closed
        # franka.control_dofs_force(np.array([grip_force, grip_force]), fingers_dof)
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

def descend_to_place_cautiously(scene, cam, franka, gso_object, df, deform_csv, photo_path, photo_interval, name,
                        end_effector, x, y, z, motors_dof, fingers_dof, grip_force,
                        quat=np.array([0, 1, 0, 0]), steps=300):
    eef_pos = franka.get_links_pos([8]).tolist()[0]
    z_hover = eef_pos[2]
    for step in range(steps*INTERPOLATE):
        z_target = z_hover - (z_hover - z) * (step + 1) / (steps*INTERPOLATE)
        qpos = franka.inverse_kinematics(link=end_effector, pos=np.array([x, y, z_target]), quat=quat)
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(np.array([0.0, 0.0]), fingers_dof)  # Ensure gripper is fully closed
        # franka.control_dofs_force(np.array([grip_force, grip_force]), fingers_dof)
        make_step(
            scene=scene, cam=cam, franka=franka, df=df, deform_csv=deform_csv,
            photo_path=photo_path, photo_interval=photo_interval, gso_object=gso_object, name=name,
            gripper_force=grip_force
        )
