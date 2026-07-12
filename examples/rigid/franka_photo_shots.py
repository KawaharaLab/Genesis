import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import genesis as gs


def main():
    parser = argparse.ArgumentParser(description="Capture multiple camera shots of Franka Emika Panda.")
    parser.add_argument("--num-shots", type=int, default=10, help="Number of shots to save (legacy; ignored in variant mode).")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/franka_shots"),
        help="Directory to save images.",
    )
    parser.add_argument("--res-x", type=int, default=1280)
    parser.add_argument("--res-y", type=int, default=960)
    parser.add_argument("--vis", action="store_true", default=False, help="Show viewer while capturing.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    gs.init(backend=gs.gpu, precision="32", logging_level=None)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.8, -1.8, 1.5),
            camera_lookat=(0.2, 0.0, 0.55),
            camera_fov=35,
            res=(args.res_x, args.res_y),
        ),
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.add_entity(gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.65, 0.0, 0.025)))

    # Base views (former shot 2 and shot 10), then lower the camera in several patterns.
    base_views = {
        "view2": ((2.6, 1.9, 0.82), (0.25, 0.00, 0.80), 32),
        "view10": ((3.1, -0.3, 1.1), (0.4, 0.0, 0.45), 28),
    }
    lower_dz_patterns = [0.00, 0.08, 0.16, 0.24, 0.32]

    camera_variants = []
    for view_name, (base_pos, lookat, fov) in base_views.items():
        for dz in lower_dz_patterns:
            pos = (base_pos[0], base_pos[1], base_pos[2] - dz)
            variant_name = f"{view_name}_lower_{int(dz * 100):02d}cm"
            camera_variants.append((variant_name, pos, lookat, fov))

    cams = []
    for variant_name, pos, lookat, fov in camera_variants:
        cam = scene.add_camera(
            res=(args.res_x, args.res_y),
            pos=pos,
            lookat=lookat,
            fov=fov,
            GUI=args.vis,
        )
        cams.append((variant_name, cam))

    scene.build()

    # Solve IK so the end-effector stays right above the cube, then hold that pose.
    cube_top_center = np.array([0.65, 0.0, 0.025], dtype=np.float32)
    ee_target_pos = cube_top_center + np.array([0.0, 0.0, 0.22], dtype=np.float32)
    ee_target_quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    end_effector = franka.get_link("hand")
    qpos_ik = franka.inverse_kinematics(link=end_effector, pos=ee_target_pos, quat=ee_target_quat)
    if hasattr(qpos_ik, "detach"):
        qpos_target = qpos_ik.detach().cpu().numpy().astype(np.float32)
    else:
        qpos_target = np.asarray(qpos_ik, dtype=np.float32)
    qpos_target[-2:] = 0.04
    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100], dtype=np.float32))
    franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10], dtype=np.float32))
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100], dtype=np.float32),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100], dtype=np.float32),
    )
    franka.set_qpos(qpos_target)

    dofs_all = np.arange(9)
    for _ in range(120):
        franka.control_dofs_position(qpos_target, dofs_all)
        scene.step()

    for variant_name, cam in cams:
        for _ in range(8):
            franka.control_dofs_position(qpos_target, dofs_all)
            scene.step()
        rgb, _, _, _ = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        out_path = args.out_dir / f"franka_{variant_name}.png"
        Image.fromarray(rgb).save(out_path)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
