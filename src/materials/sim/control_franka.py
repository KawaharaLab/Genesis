import numpy as np
import genesis as gs
from .make_step import make_step

def control_franka(scene, cam, franka, grasp_pos, qpos_init, strength, df, base_photo_name, photo_interval):
    # DOF インデックスを NumPy で定義
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
    end_effector = franka.get_link("hand")
    # move to pre-grasp pose
    dz = 0.0001
    x = 0.45
    y = 0.45
    z = 0.5
    z_steps = int((z - grasp_pos[2]) // dz)
    dx = (x - grasp_pos[0]) / z_steps
    dy = (y - grasp_pos[1]) / z_steps
    print("x,y,z: ", x, y, z)
    qpos = qpos_init.clone().to(gs.device)
    print("qpos: ", qpos)
    franka.set_dofs_position(qpos[:-2], motors_dof)
    franka.set_dofs_position(qpos[-2:], fingers_dof)
    cam.start_recording()
    #=================この中を調整========================
    # reach
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=grasp_pos,
        quat=np.array([0, 1, 0, 0]),
    )
    qpos[-2:] = 0.04
    path = franka.plan_path(qpos, num_waypoints=2000)
    for waypoint in path:
        franka.control_dofs_position(waypoint)
        make_step(scene, cam, franka, df, base_photo_name, photo_interval)
    for _ in range(30):
        make_step(scene, cam, franka, df, base_photo_name, photo_interval)

    # grasp
    for i in range(300):
        force = np.array([-strength/300 * i, -strength/300 * i])
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(force, fingers_dof)
        make_step(scene, cam, franka, df, base_photo_name, photo_interval)

    # lift up
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([grasp_pos[0], grasp_pos[1], z]),
        quat=np.array([0, 1, 0, 0]),
    )
    path = franka.plan_path(qpos, num_waypoints=3000)
    for waypoint in path:
        franka.control_dofs_position(waypoint[:-2], motors_dof)
        make_step(scene, cam, franka, df, base_photo_name, photo_interval)

    # # 小幅往復
    # for i in range(3000):
    #     qpos = franka.get_qpos()
    #     qpos[-4] -= 0.0002
    #     franka.control_dofs_position(qpos[:-2], motors_dof)
    #     franka.control_dofs_force(np.array([-strength, -strength]), fingers_dof)
    #     make_step(scene, cam, franka, df, base_photo_name, photo_interval)

    # for i in range(3000):
    #     qpos = franka.get_qpos()
    #     qpos[-4] += 0.0002
    #     franka.control_dofs_position(qpos[:-2], motors_dof)
    #     franka.control_dofs_force(np.array([-strength, -strength]), fingers_dof)
    #     make_step(scene, cam, franka, df, base_photo_name, photo_interval)

    # for i in range(3000):
    #     qpos = franka.get_qpos()
    #     qpos[0] += 0.0002
    #     franka.control_dofs_position(qpos[:-2], motors_dof)
    #     franka.control_dofs_force(np.array([-strength, -strength]), fingers_dof)
    #     make_step(scene, cam, franka, df, base_photo_name, photo_interval)

    # for i in range(1000):
    #     force = np.array([-strength + strength/1000 * i, -strength + strength/1000 * i])
    #     qpos = franka.get_qpos()
    #     franka.control_dofs_position(qpos[:-2], motors_dof)
    #     franka.control_dofs_force(force, fingers_dof)
    #     make_step(scene, cam, franka, df, base_photo_name, photo_interval)

    # # フィンガー開閉
    # for i in range(400):
    #     finger_pos = np.array([0.0001 * i, 0.0001 * i])
    #     qpos = franka.get_qpos()
    #     franka.control_dofs_position(qpos[:-2], motors_dof)
    #     franka.control_dofs_position(finger_pos, fingers_dof)
    #     make_step(scene, cam, franka, df, base_photo_name, photo_interval)
