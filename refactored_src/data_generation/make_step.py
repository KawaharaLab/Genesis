# Save as: your-project/src/make_step.py

import os
import imageio.v3 as iio
import numpy as np

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

    # Get the maximum deformation value for the specified object
    all_defs = scene.sim.mpm_solver.deformation_metric.to_numpy()
    obj_defs = all_defs[:, gso_object.particle_start : gso_object.particle_start + gso_object.n_particles]
    max_deformation = obj_defs.max() if obj_defs.size > 0 else 0.0

    deform_csv.loc[len(deform_csv)] = [scene.t, max_deformation, gripper_force]

    # Record robot state (DOFs and force/torque on end-effector links)
    dofs = franka.get_dofs_position().tolist()
    # Links 9 and 10 are the gripper fingers
    links_ft = franka.get_links_force_torque([9, 10])
    forces_torques = links_ft[0].tolist() + links_ft[1].tolist()

    df.loc[len(df)] = [scene.t] + forces_torques + dofs

    # Save photos from multiple camera angles if the condition is met
    if force_photo or (t % photo_interval == 0):
        camera_poses = [
            {'pos': (2.1, -1.2, 0.1), 'lookat': (0.45, 0.45, 0.5)},
            {'pos': (-1.5, 1.5, 0.25), 'lookat': (0.45, 0.45, 0.4)},
            {'pos': (2, 2, 0.1), 'lookat': (0, 0, 0.1)}
        ]
        for i, pose in enumerate(camera_poses):
            cam.set_pose(**pose)
            rgb, _, _, _ = cam.render(rgb=True)
            if photo_path:
                filepath = os.path.join(photo_path, f"{name}_{t:05d}_Camera{i}.png")
                iio.imwrite(filepath, rgb)

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