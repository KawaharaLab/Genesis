import os
import imageio.v3 as iio


def get_bounding_box(gso_object):
    aabbs = gso_object.get_AABB().cpu().numpy()
    return aabbs[0].tolist() + aabbs[1].tolist()

def _execute_simulation_step(scene, cam, franka, df, deform_csv, photo_path, photo_interval,
                           name, gso_object, gripper_force=0.0, force_photo=False):
    """
    Executes a single step in the simulation, records data, and optionally saves images.
    This is an internal helper function consolidating logic from make_step and final_make_step.

    Args:
        force_photo (bool): If True, saves a photo regardless of the photo_interval.
    """

    scene.step()
    t = int(scene.t) - 1

    # Record robot state (DOFs and force/torque on end-effector links)
    dofs = franka.get_dofs_position().tolist()
    # Links 9 and 10 are the gripper fingers. 9 is the left finger, 10 is the right finger.
    links_f = franka.get_links_contact_force([9, 10], sensor=True)
    links_t = franka.get_links_contact_torque([9, 10], sensor=True)
    left_finger_force = links_f[0].tolist()
    right_finger_force = links_f[1].tolist()
    left_finger_torque = links_t[0].tolist()
    right_finger_torque = links_t[1].tolist()
    force_torques = left_finger_force + left_finger_torque + right_finger_force +right_finger_torque

    eef_pos = franka.get_links_pos([8, 9, 10]).flatten().tolist()
    finger_ctrl = franka.get_dofs_control_force([7, 8])
    finger_control = finger_ctrl.tolist()
    obj_com = gso_object.get_root_COM().tolist()
    obj_mass = [gso_object.get_mass()]
    obj_bounding_box = get_bounding_box(gso_object)

    obj_contacts = [0, 0, 0]
    obj_contact_info = gso_object.get_contacts()
    obj_contact_pairs = obj_contact_info["link_a"]
    if franka.get_link("left_finger").idx in obj_contact_pairs:
        obj_contacts[0] = 1
    if franka.get_link("right_finger").idx in obj_contact_pairs:
        obj_contacts[1] = 1
    if 0 in obj_contact_pairs:
        obj_contacts[2] = 1
    df.loc[len(df)] = [scene.t] + force_torques + dofs + eef_pos + finger_control + obj_com + obj_mass + obj_bounding_box + obj_contacts

    # Save photos from multiple camera angles if the condition is met
    if force_photo or (t % photo_interval == 0):
        camera_poses = [
            # {'pos': (2.1, -1.2, 0.1), 'lookat': (0.45, 0.45, 0.5)},
            {'pos': (1.6, -1.6, 0.2), 'lookat': (0.4, 0.4, 0.2)},
            {'pos': (-1.6, 1.6, 0.2), 'lookat': (0.4, 0.4, 0.2)},
            {'pos': (2, 2, 0.2), 'lookat': (0, 0, 0.2)}
        ]
        for i, pose in enumerate(camera_poses):
            cam.set_pose(**pose)
            rgb, _, _, _ = cam.render(rgb=True)
            if photo_path:
                filepath = os.path.join(photo_path, f"camera_{i}/{name}_{t:05d}.png")
                iio.imwrite(filepath, rgb)
    if t % 100 == 0 or force_photo:
        print(f"Step: {t:05d} | Object: {name}")

    # # Return False to stop the simulation if forces are too high (indicating instability)
    # if abs(df.iloc[-1, 8]) > 100:
    #     return False
    return True

# Define the public-facing functions that call the internal helper
def make_step(*args, **kwargs):
    return _execute_simulation_step(*args, force_photo=False, **kwargs)

def final_make_step(*args, **kwargs):
    return _execute_simulation_step(*args, force_photo=True, **kwargs)
