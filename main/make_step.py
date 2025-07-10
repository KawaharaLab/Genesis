import imageio.v3 as iio
"""https://pypi.org/project/imageio/"""

def make_step(scene, cam, franka, df, photo_path, photo_interval, gso_object, deform_csv, name):
    """フランカを目標位置に移動させるステップ関数"""
    scene.step()
    t = int(scene.t) - 1

    # obtain deformation values if the object is an MPMEntity
    # if isinstance(gso_object, gs.engine.entities.MPMEntity):

    # Assume it is an MPMEntity
    # 1. Get all deformation values
    all_deformation_values = scene.sim.mpm_solver.deformation_metric.to_numpy()

    # 2. Get the slice indices for your specific object
    start_idx = gso_object.particle_start
    end_idx = start_idx + gso_object.n_particles

    # 3. Slice the array
    object_deformation = all_deformation_values[:, start_idx:end_idx]

    # 4. Calculate the max deformation for just this object
    max_deformation = object_deformation.max()

    if max_deformation > 0.01:
        print(f"Object deformed at step {scene.t}! Max value: {max_deformation:.4f}")

    # Record deformation values
    deform_csv.loc[len(deform_csv)] = [
        scene.t,
        max_deformation
    ]
    
    # generate clean video from one of the camera angles
    camera_angle = 1
    if t % photo_interval == 0:
        # cam.render()
        rgb, _, _, _  = cam.render(rgb=True)
        if camera_angle == 0:
            cam.set_pose(pos = (2.1, -1.2, 0.1), lookat = (0.45, 0.45, 0.5))
        elif camera_angle == 1:
            cam.set_pose(pos = (-1.5, 1.5, 0.25), lookat = (0.45, 0.45, 0.4))
        elif camera_angle == 2:
            cam.set_pose(pos = (2, 2, 0.1), lookat = (0, 0, 0.1))
        if photo_path is not None:
            filepath = photo_path + f"{name}{t:05d}Camera{camera_angle}.png"
            iio.imwrite(filepath, rgb)
        dofs = franka.get_dofs_position()
        dofs = [x.item() for x in dofs]
        links_force_torque = franka.get_links_force_torque([9, 10]) # 手先のlocal_indexは9, 10
        links_force_torque = [x.item() for x in links_force_torque[0]] + [x.item() for x in links_force_torque[1]]
        df.loc[len(df)] = [
            scene.t,
            links_force_torque[0], links_force_torque[1], links_force_torque[2],
            links_force_torque[3], links_force_torque[4], links_force_torque[5],
            links_force_torque[6], links_force_torque[7], links_force_torque[8],
            links_force_torque[9], links_force_torque[10], links_force_torque[11],
            dofs[0], dofs[1], dofs[2], dofs[3], dofs[4], dofs[5], dofs[6], dofs[7], dofs[8]
        ]

        
        
# if DEBUG:
#     rgb, _, _, _  = cam.render(rgb=True)
# else:
#     if t % photo_interval == 0:
#         for i in range(3):
#             rgb, _, _, _  = cam.render(rgb=True)
#             if i == 0:
#                 cam.set_pose(pos = (2.1, -1.2, 0.1), lookat = (0.45, 0.45, 0.5))
#             elif i == 1:
#                 cam.set_pose(pos = (-1.5, 1.5, 0.25), lookat = (0.45, 0.45, 0.4))
#             elif i == 2:
#                 cam.set_pose(pos = (2, 2, 0.1), lookat = (0, 0, 0.1))
#             if photo_path is not None:
#                 filepath = photo_path + f"{name}{t:05d}Angle{i}.png"
#                 iio.imwrite(filepath, rgb)
#     dofs = franka.get_dofs_position()
#     dofs = [x.item() for x in dofs]
#     links_force_torque = franka.get_links_force_torque([9, 10]) # 手先のlocal_indexは9, 10
#     links_force_torque = [x.item() for x in links_force_torque[0]] + [x.item() for x in links_force_torque[1]]
#     df.loc[len(df)] = [
#         scene.t,
#         links_force_torque[0], links_force_torque[1], links_force_torque[2],
#         links_force_torque[3], links_force_torque[4], links_force_torque[5],
#         links_force_torque[6], links_force_torque[7], links_force_torque[8],
#         links_force_torque[9], links_force_torque[10], links_force_torque[11],
#         dofs[0], dofs[1], dofs[2], dofs[3], dofs[4], dofs[5], dofs[6], dofs[7], dofs[8]
#     ]