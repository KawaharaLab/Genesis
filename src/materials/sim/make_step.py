import imageio.v3 as iio
"""https://pypi.org/project/imageio/"""

def make_step(scene, cam, franka, df, photo_path, photo_interval):
    """フランカを目標位置に移動させるステップ関数"""
    scene.step()
    t = int(scene.t) - 1
    if t % photo_interval == 0:
        rgb, _, _, _ = cam.render(rgb=True)
        filepath = photo_path + f"_{t:05d}.png"
        iio.imwrite(filepath, rgb)
    raw_dofs = franka.get_dofs_position()
    dofs = [x.item() for x in raw_dofs]
    raw_lft = franka.get_links_force_torque([9, 10])
    # 各リンクの力・モーメント配列を結合して要素を取得
    links_force_torque = [x.item() for x in raw_lft[0]] + [x.item() for x in raw_lft[1]]
    # DataFrameへの追加はタプル展開でリスト生成コストを削減
    df.loc[len(df)] = (
        scene.t,
        *links_force_torque,
        *dofs
    )
