import torch
import genesis as gs
from .make_step import make_step

def control_franka(scene, cam, franka, grasp_pos, qpos_init, strength, df, base_photo_name, photo_interval):
    motors_dof = torch.arange(7, dtype=gs.tc_int, device=gs.device)
    fingers_dof = torch.arange(7, 9, dtype=gs.tc_int, device=gs.device)
    # Optional: set control gains
    franka.set_dofs_kp(
        torch.tensor([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100], device=gs.device),
    )
    franka.set_dofs_kv(
        torch.tensor([450, 450, 350, 350, 200, 200, 200, 10, 10], device=gs.device),
    )
    franka.set_dofs_force_range(
        torch.tensor([-87, -87, -87, -87, -12, -12, -12, -100, -100], device=gs.device),
        torch.tensor([87, 87, 87, 87, 12, 12, 12, 100, 100], device=gs.device),
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
    # 事前にテンソルを１回だけ作成（ループ内で再利用）
    pos_tensor = torch.empty(3, dtype=gs.tc_float, device=gs.device)
    quat_tensor = torch.tensor([0, 1, 0, 0], dtype=gs.tc_float, device=gs.device)
    force_tensor = torch.empty(2, dtype=gs.tc_float, device=gs.device)
    finger_pos_tensor = torch.empty(2, dtype=gs.tc_float, device=gs.device)

    # reach
    for i in range(z_steps):
        x -= dx; y -= dy; z -= dz
        pos_tensor[0], pos_tensor[1], pos_tensor[2] = x, y, z
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=pos_tensor,
            quat=quat_tensor,
        )
        qpos[-2:] = 0.04
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(qpos[-2:], fingers_dof)
        make_step(scene, cam, franka, df, base_photo_name, photo_interval)

    # grasp
    for i in range(300):
        # pos, quat は変わらないので再利用
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=pos_tensor,
            quat=quat_tensor,
        )
        force_tensor.fill_(-strength/300 * i)
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(force_tensor, fingers_dof)
        make_step(scene, cam, franka, df, base_photo_name, photo_interval)

    # lift up
    for i in range(z_steps):
        z += dz
        pos_tensor[2] = z
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=pos_tensor,
            quat=quat_tensor,
        )
        force_tensor.fill_(-strength)
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(force_tensor, fingers_dof)
        make_step(scene, cam, franka, df, base_photo_name, photo_interval)

    # 小幅往復
    for i in range(3000):
        qpos[-4] -= 0.0002
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(force_tensor, fingers_dof)
        make_step(scene, cam, franka, df, base_photo_name, photo_interval)

    for i in range(3000):
        qpos[-4] += 0.0002
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(force_tensor, fingers_dof)
        make_step(scene, cam, franka, df, base_photo_name, photo_interval)

    for i in range(3000):
        qpos[0] += 0.0002
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(force_tensor, fingers_dof)
        make_step(scene, cam, franka, df, base_photo_name, photo_interval)

    for i in range(1000):
        force_tensor.fill_(-strength + strength/1000 * i)
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(force_tensor, fingers_dof)
        make_step(scene, cam, franka, df, base_photo_name, photo_interval)

    # フィンガー開閉
    for i in range(400):
        finger_pos_tensor.fill_(0.0001 * i)
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_position(finger_pos_tensor, fingers_dof)
        make_step(scene, cam, franka, df, base_photo_name, photo_interval)
    