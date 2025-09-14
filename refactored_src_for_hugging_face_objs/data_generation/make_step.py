# Save as: your-project/src/make_step.py

import os
import imageio.v3 as iio
import numpy as np

ELASTIC = 0
MADE_CONTACT = 0

#uncoment this for dectecting grasp and preventing force spike when an object is dropped (look at line 46 as well)
'''def get_contact_state(franka):
    global MADE_CONTACT
    #print(len(franka.detect_collision()))
    if len(franka.detect_collision()) == 1 and MADE_CONTACT == 0:
        MADE_CONTACT = 1
    elif MADE_CONTACT == 1 and len(franka.detect_collision()) == 0:
        MADE_CONTACT = 2
    # else:
    #     MADE_CONTACT = 0

    contact_state = MADE_CONTACT
    return contact_state'''

def get_bounding_box(gso_object):
    if ELASTIC:
        particle_positions = gso_object.get_state().pos.detach().cpu().numpy()[0]
        min_coords = np.min(particle_positions, axis=0)
        max_coords = np.max(particle_positions, axis=0)
        return min_coords.tolist() + max_coords.tolist()
    else:
        AABBs = gso_object.get_AABB().cpu().numpy()
        return AABBs[0].tolist() + AABBs[1].tolist()

def _execute_simulation_step(scene, cam, franka, df, deform_csv, photo_path, photo_interval,
                           name, gso_object, gripper_force=0.0, force_photo=False):
    """
    Executes a single step in the simulation, records data, and optionally saves images.
    This is an internal helper function consolidating logic from make_step and final_make_step.

    Args:
        force_photo (bool): If True, saves a photo regardless of the photo_interval.
    """

    #uncoment this for dectecting grasp and preventing force spike when an object is dropped
    '''    contact = get_contact_state(franka)
    if contact == 2:
        fingers_dof = np.arange(7, 9)
        franka.control_dofs_force(np.array([0, 0]), fingers_dof)
        franka.control_dofs_position(np.array([0.1, 0.1]), fingers_dof)'''

    scene.step()
    t = int(scene.t) - 101

    # Get the maximum deformation value for the specified object
    if ELASTIC:
        all_defs = scene.sim.mpm_solver.deformation_metric.to_numpy()
        obj_defs = all_defs[:, gso_object.particle_start : gso_object.particle_start + gso_object.n_particles]
        max_deformation = obj_defs.max() if obj_defs.size > 0 else 0.0

        deform_csv.loc[len(deform_csv)] = [scene.t, max_deformation, gripper_force]

    # Record robot state (DOFs and force/torque on end-effector links)
    dofs = franka.get_dofs_position().tolist()
    # Links 9 and 10 are the gripper fingers
    links_ft = franka.get_links_force_torque([9, 10], sensor=True)
    forces_torques = links_ft[0].tolist() + links_ft[1].tolist()

    eef_pos = franka.get_links_pos([8, 9, 10]).flatten().tolist()
    finger_ctrl = franka.get_dofs_control_force([7, 8])
    finger_control = finger_ctrl.tolist()
    if ELASTIC:
        obj_com = gso_object.get_COM().tolist()
    else:
        obj_com = gso_object.get_root_COM().tolist()
    obj_mass = [gso_object.get_mass()]
    obj_bounding_box = get_bounding_box(gso_object)

    if ELASTIC:
        obj_contacts = [None, None, None]
    else:
        obj_contacts = [0, 0, 0]
        obj_contact_info = gso_object.get_contacts()
        obj_contact_pairs = obj_contact_info["link_a"]
        if franka.get_link("left_finger").idx in obj_contact_pairs:
            obj_contacts[0] = 1
        if franka.get_link("right_finger").idx in obj_contact_pairs:
            obj_contacts[1] = 1
        if 0 in obj_contact_pairs:
            obj_contacts[2] = 1
    df.loc[len(df)] = [scene.t] + forces_torques + dofs + eef_pos + finger_control + obj_com + obj_mass + obj_bounding_box + obj_contacts

    # Save photos from multiple camera angles if the condition is met
    if force_photo or (t % photo_interval == 0):
        camera_poses = [
            {'pos': (2.1, -1.2, 0.1), 'lookat': (0.45, 0.45, 0.5)},
            #{'pos': (-1.5, 1.5, 0.25), 'lookat': (0.45, 0.45, 0.4)},
            #{'pos': (2, 2, 0.1), 'lookat': (0, 0, 0.1)}
        ]
        for i, pose in enumerate(camera_poses):
            cam.set_pose(**pose)
            rgb, _, _, _ = cam.render(rgb=True)
            if photo_path:
                filepath = os.path.join(photo_path, f"camera_{i}/{name}_{t:05d}.png")
                iio.imwrite(filepath, rgb)
    if t % 100 == 0 or force_photo:
        print(f"Step: {t:05d} | Object: {name}")

    # Return False to stop the simulation if forces are too high (indicating instability)
    if abs(df.iloc[-1, 8]) > 100:
        return False
    return True

# Define the public-facing functions that call the internal helper
def make_step(*args, **kwargs):
    return _execute_simulation_step(*args, force_photo=False, **kwargs)

def final_make_step(*args, **kwargs):
    return _execute_simulation_step(*args, force_photo=True, **kwargs)