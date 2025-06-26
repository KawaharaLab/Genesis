import imageio.v3 as iio
"""https://pypi.org/project/imageio/"""

def make_step(scene, cam, franka, df, photo_path, photo_interval):
    """フランカを目標位置に移動させるステップ関数"""
    scene.step()
    t = int(scene.t) - 1
    # if t % 20 == 0:
    #     cam.render()
    if t % photo_interval == 0:
        rgb, _, _, _ = cam.render(rgb=True)

        filepath = photo_path + f"_{t:05d}.png"
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
    # #force