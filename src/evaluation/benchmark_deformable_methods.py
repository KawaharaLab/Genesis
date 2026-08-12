"""Compare rigid, MPM, implicit FEM+SAP, and implicit FEM+IPC grasp simulations.

The externally visible time step is fixed at 0.01 s (100 Hz).  Every method is
driven with the same end-effector trajectory and low (-0.5 N per finger) grasp
force.  Method-specific substeps are only used inside one 100 Hz sample.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import trimesh

import genesis as gs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERATION_DIR = PROJECT_ROOT / "src" / "generation"
sys.path.insert(0, str(GENERATION_DIR))

from run_genesis_sim import get_obj_bounds, set_grasp

DT = 1e-2
STAGE_STEPS = {
    "settle_and_move_to_hover": 50,
    "approach": 50,
    "close": 80,
    "hold": 20,
    "lift": 60,
}
FINGER_FORCE_N = -0.5
METHOD_SUBSTEPS = {"rigid": 1, "mpm": 80, "fem_sap": 2, "fem_ipc": 1}
CYLINDER_OBJECTS = {
    "001_chips_can",
    "002_master_chef_can",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
}
USE_TORCH_CUDA_SYNC = True


def _sync():
    if USE_TORCH_CUDA_SYNC and torch.cuda.is_available():
        torch.cuda.synchronize()


def _as_numpy(value):
    return value.detach().cpu().numpy()


def _finite_geometry(pos, vel):
    pos = np.asarray(pos).reshape(-1, 3)
    vel = np.asarray(vel).reshape(-1, 3)
    finite = np.isfinite(pos).all(axis=1) & np.isfinite(vel).all(axis=1)
    valid_pos = pos[finite]
    return {
        "finite_fraction": float(finite.mean()) if len(finite) else 0.0,
        "bbox_min": np.min(valid_pos, axis=0).tolist() if len(valid_pos) else [None] * 3,
        "bbox_max": np.max(valid_pos, axis=0).tolist() if len(valid_pos) else [None] * 3,
        "extent": np.ptp(valid_pos, axis=0).tolist() if len(valid_pos) else [None] * 3,
        "center": np.mean(valid_pos, axis=0).tolist() if len(valid_pos) else [None] * 3,
        "max_speed": float(np.linalg.norm(vel[finite], axis=1).max()) if finite.any() else None,
    }


def _tet_signed_volumes(pos, elems):
    tet = pos[elems]
    return (
        np.einsum(
            "ij,ij->i",
            tet[:, 1] - tet[:, 0],
            np.cross(tet[:, 2] - tet[:, 0], tet[:, 3] - tet[:, 0]),
        )
        / 6.0
    )


def _snapshot(entity, method, rest_tet_volumes=None):
    if method == "rigid":
        bounds = _as_numpy(entity.get_AABB()).reshape(2, 3)
        vel = _as_numpy(entity.get_vel()).reshape(-1, 3)
        result = {
            "finite_fraction": float(np.isfinite(bounds).all() and np.isfinite(vel).all()),
            "bbox_min": bounds[0].tolist(),
            "bbox_max": bounds[1].tolist(),
            "extent": (bounds[1] - bounds[0]).tolist(),
            "center": ((bounds[0] + bounds[1]) * 0.5).tolist(),
            "max_speed": float(np.linalg.norm(vel, axis=1).max()),
        }
        return result, rest_tet_volumes

    state = entity.get_state()
    pos = _as_numpy(state.pos).reshape(-1, 3)
    vel = _as_numpy(state.vel).reshape(-1, 3)
    result = _finite_geometry(pos, vel)

    if method == "mpm":
        active = _as_numpy(state.active).reshape(-1).astype(bool)
        F = _as_numpy(state.F).reshape(-1, 3, 3)
        finite_f = np.isfinite(F).all(axis=(1, 2))
        valid_f = active & finite_f
        det_f = np.linalg.det(F[valid_f]) if valid_f.any() else np.array([np.nan])
        result.update(
            particles_total=int(active.size),
            particles_active=int(active.sum()),
            det_F_min=float(np.nanmin(det_f)),
            det_F_max=float(np.nanmax(det_f)),
        )
        return result, rest_tet_volumes

    active = _as_numpy(state.active).reshape(-1).astype(bool)
    elems = np.asarray(entity.elems, dtype=np.int64)
    volumes = _tet_signed_volumes(pos, elems)
    if rest_tet_volumes is None:
        rest_tet_volumes = volumes.copy()
    valid_rest = np.abs(rest_tet_volumes) > 1e-15
    ratios = np.full_like(volumes, np.nan)
    ratios[valid_rest] = volumes[valid_rest] / rest_tet_volumes[valid_rest]
    result.update(
        vertices_total=int(pos.shape[0]),
        elements_total=int(elems.shape[0]),
        elements_active=int(active.sum()),
        inverted_elements=int(np.sum(ratios[valid_rest] <= 0.0)),
        volume_ratio_min=float(np.nanmin(ratios)),
        volume_ratio_max=float(np.nanmax(ratios)),
    )
    return result, rest_tet_volumes


def _render(camera, output_dir, run_name, stage):
    rgb, _, _, _ = camera.render(rgb=True)
    path = output_dir / f"{run_name}_{stage}.png"
    iio.imwrite(path, rgb)
    return str(path.resolve())


def _sync_ipc_abd_to_genesis(scene):
    """Teleport IPC's coupled rigid links to the current Genesis pose."""
    coupler = scene._sim._coupler
    coupler._store_gs_rigid_states()
    coupler._abd_state_feature.copy_to(coupler._abd_state_geom)
    transforms = coupler._abd_state_geom.instances().find("transform").view()
    velocities = coupler._abd_state_geom.instances().find("velocity").view()
    for link, body_indices in coupler._coupling_data.abd_body_idx_by_link.items():
        for env_idx, body_idx in enumerate(body_indices):
            transforms[body_idx] = coupler._abd_transforms_by_link[link][env_idx]
            velocities[body_idx] = 0.0
    coupler._abd_state_feature.copy_from(coupler._abd_state_geom)
    coupler._ipc_world.retrieve()


def _get_comparison_mesh(source_path, output_dir):
    """Create a watertight convex reference geometry.

    YCB OBJ files duplicate vertices at UV seams and are not tetrahedralizable.
    The cached hull is used directly by rigid and MPM and supplies the outer
    dimensions for low-complexity FEM proxies.
    """
    mesh_dir = output_dir / "comparison_meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    hull_path = mesh_dir / f"{source_path.parent.name}_convex_hull.obj"
    if not hull_path.exists():
        source = trimesh.load(source_path, force="mesh", process=False)
        source.convex_hull.export(hull_path)
    return hull_path, "convex_hull"


def _fem_proxy_morph(object_name, mesh_min, mesh_max, scale, spawn, euler):
    """Make a low-complexity FEM proxy with the YCB mesh's pose and outer dimensions."""
    scale_vec = np.full(3, scale) if np.isscalar(scale) else np.asarray(scale)
    size = (mesh_max - mesh_min) * scale_vec
    local_center = (mesh_min + mesh_max) * 0.5 * scale_vec
    angle = np.deg2rad(euler[2])
    rotation_z = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    center = np.asarray(spawn) + rotation_z @ local_center
    tet_options = {"nobisect": True}
    if object_name in CYLINDER_OBJECTS:
        radius = float((size[0] + size[1]) * 0.25)
        morph = gs.morphs.Cylinder(pos=tuple(center), euler=euler, radius=radius, height=float(size[2]), **tet_options)
        proxy = {"type": "cylinder", "size": [2.0 * radius, 2.0 * radius, float(size[2])]}
    else:
        morph = gs.morphs.Box(pos=tuple(center), euler=euler, size=tuple(size), **tet_options)
        proxy = {"type": "box", "size": size.tolist()}
    return morph, proxy


def _run_stage(scene, commands):
    _sync()
    start = time.perf_counter()
    for command in commands:
        command()
        scene.step()
    _sync()
    seconds = time.perf_counter() - start
    return {"steps": len(commands), "physical_seconds": len(commands) * DT, "seconds": seconds}


def _scene_options(method, grid_density):
    kwargs = {"sim_options": gs.options.SimOptions(dt=DT, substeps=METHOD_SUBSTEPS[method])}
    if method == "mpm":
        kwargs["mpm_options"] = gs.options.MPMOptions(
            lower_bound=(0.0, -0.1, -0.05), upper_bound=(0.85, 1.0, 1.0), grid_density=grid_density
        )
    elif method == "fem_sap":
        kwargs["fem_options"] = gs.options.FEMOptions(use_implicit_solver=True, pcg_threshold=1e-10)
        kwargs["coupler_options"] = gs.options.SAPCouplerOptions(
            pcg_threshold=1e-10,
            sap_convergence_atol=1e-10,
            sap_convergence_rtol=1e-10,
            linesearch_ftol=1e-10,
            fem_floor_contact_type="vert",
            enable_fem_self_tet_contact=False,
            rigid_floor_contact_type="none",
            rigid_rigid_contact_type="none",
        )
        kwargs["rigid_options"] = gs.options.RigidOptions(enable_self_collision=False)
    elif method == "fem_ipc":
        kwargs["coupler_options"] = gs.options.IPCCouplerOptions(
            constraint_strength_translation=10.0,
            constraint_strength_rotation=10.0,
            enable_rigid_rigid_contact=False,
            enable_rigid_ground_contact=False,
        )
    return kwargs


def _materials(method):
    if method == "mpm":
        return gs.materials.Rigid(coup_friction=1.0, friction=1.0), gs.materials.MPM.Elastic(
            E=5e4, nu=0.4, rho=1000.0, model="corotation"
        )
    if method == "fem_sap":
        return gs.materials.Rigid(coup_friction=1.0, friction=1.0), gs.materials.FEM.Elastic(
            E=5e4, nu=0.4, rho=1000.0, friction_mu=0.5, model="linear_corotated"
        )
    if method == "fem_ipc":
        return gs.materials.Rigid(
            coup_friction=0.8,
            coup_type="two_way_soft_constraint",
            coup_links=("left_finger", "right_finger"),
        ), gs.materials.FEM.Elastic(E=5e4, nu=0.4, rho=1000.0, friction_mu=0.5, model="stable_neohookean")
    return gs.materials.Rigid(coup_friction=1.0, friction=1.0), gs.materials.Rigid(rho=1000.0)


def run(args):
    global USE_TORCH_CUDA_SYNC
    precision = "64" if args.method == "fem_sap" else "32"
    genesis_backend = gs.cpu if args.method == "fem_ipc" else gs.gpu
    USE_TORCH_CUDA_SYNC = args.method != "fem_ipc"
    gs.init(backend=genesis_backend, precision=precision, logging_level="warning")

    source_mesh_path = PROJECT_ROOT / "data" / "objects" / "ycb" / args.object / "model.obj"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    scale, euler = set_grasp(source_mesh_path)
    mesh_min, mesh_max = get_obj_bounds(source_mesh_path)
    scale_vec = np.full(3, scale) if np.isscalar(scale) else np.asarray(scale)
    # Keep the object on the robot's forward axis.  The neutral-to-hover motion
    # otherwise sweeps through objects placed diagonally at (0.45, 0.45).
    center_xy = np.array([0.65, 0.0])
    spawn_z = 0.002 + max(0.0, -scale_vec[2] * mesh_min[2])
    spawn = (float(center_xy[0]), float(center_xy[1]), float(spawn_z))
    mesh_path, geometry_approximation = _get_comparison_mesh(source_mesh_path, output_dir)

    scene = gs.Scene(**_scene_options(args.method, args.grid_density), show_viewer=False)
    # SAP represents the floor internally and rejects a user plane collision mesh.
    if args.method != "fem_sap":
        scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(coup_friction=0.0))
    robot_material, object_material = _materials(args.method)
    robot_xml = "panda_non_overlap.xml" if args.method == "fem_ipc" else "panda.xml"
    franka = scene.add_entity(gs.morphs.MJCF(file=f"xml/franka_emika_panda/{robot_xml}"), material=robot_material)
    fem_proxy = None
    if args.method.startswith("fem"):
        object_morph, fem_proxy = _fem_proxy_morph(args.object, mesh_min, mesh_max, scale, spawn, euler)
        geometry_approximation = f"dimension_matched_{fem_proxy['type']}_proxy"
    else:
        object_morph = gs.morphs.Mesh(file=str(mesh_path), scale=scale, pos=spawn, euler=euler)
    obj = scene.add_entity(
        material=object_material,
        morph=object_morph,
        surface=gs.surfaces.Default(color=(0.1, 0.8, 0.1, 1.0)),
    )
    camera = None
    if args.render:
        camera = scene.add_camera(res=(800, 600), pos=(1.25, 0.75, 0.55), lookat=(0.65, 0.0, 0.18), fov=35)

    _sync()
    build_start = time.perf_counter()
    scene.build()
    _sync()
    build_seconds = time.perf_counter() - build_start

    motors = np.arange(7)
    fingers = np.arange(7, 9)
    hand = franka.get_link("hand")
    franka.set_dofs_kp(np.array([450, 450, 350, 350, 200, 200, 200, 50, 50]))
    franka.set_dofs_kv(np.array([45, 45, 35, 35, 20, 20, 20, 10, 10]))
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -2, -2]),
        np.array([87, 87, 87, 87, 12, 12, 12, 2, 2]),
    )

    initial, rest_tet_volumes = _snapshot(obj, args.method)
    object_top = initial["bbox_max"][2]
    hover_z = object_top + 0.16
    grasp_z = object_top + 0.065

    def ik(z):
        return franka.inverse_kinematics(
            link=hand,
            pos=np.array([center_xy[0], center_xy[1], z]),
            quat=np.array([0.0, 1.0, 0.0, 0.0]),
        )

    q_hover = ik(hover_z)
    q_hover[-2:] = 0.04
    # Start every method at the same collision-free pose.  Letting the neutral
    # Franka posture travel to hover sweeps the forearm through the test object.
    franka.set_qpos(q_hover)
    if args.method == "fem_ipc":
        _sync_ipc_abd_to_genesis(scene)
    q_approach = [
        ik(hover_z + (grasp_z - hover_z) * (i + 1) / STAGE_STEPS["approach"]) for i in range(STAGE_STEPS["approach"])
    ]
    q_grasp = ik(grasp_z)
    q_lift = [ik(grasp_z + 0.12 * (i + 1) / STAGE_STEPS["lift"]) for i in range(STAGE_STEPS["lift"])]

    def position_command(q):
        return lambda: (
            franka.control_dofs_position(q[motors], dofs_idx_local=motors),
            franka.control_dofs_position(0.04, dofs_idx_local=fingers),
        )

    def force_command(q):
        return lambda: (
            franka.control_dofs_position(q[motors], dofs_idx_local=motors),
            franka.control_dofs_force(np.full(2, FINGER_FORCE_N), dofs_idx_local=fingers),
        )

    commands = {
        "settle_and_move_to_hover": [position_command(q_hover)] * STAGE_STEPS["settle_and_move_to_hover"],
        "approach": [position_command(q) for q in q_approach],
        "close": [force_command(q_grasp)] * STAGE_STEPS["close"],
        "hold": [force_command(q_grasp)] * STAGE_STEPS["hold"],
        "lift": [force_command(q) for q in q_lift],
    }

    run_name = f"{args.object}_{args.method}"
    report = {
        "object": args.object,
        "method": args.method,
        "source_mesh": str(source_mesh_path.resolve()),
        "simulation_mesh": None if args.method.startswith("fem") else str(mesh_path.resolve()),
        "reference_convex_hull": str(mesh_path.resolve()),
        "geometry_approximation": geometry_approximation,
        "fem_proxy": fem_proxy,
        "genesis_backend": "cpu" if args.method == "fem_ipc" else "gpu",
        "ipc_backend": "cuda" if args.method == "fem_ipc" else None,
        "dt": DT,
        "sample_rate_hz": 1.0 / DT,
        "substeps": METHOD_SUBSTEPS[args.method],
        "internal_dt": DT / METHOD_SUBSTEPS[args.method],
        "grid_density": args.grid_density if args.method == "mpm" else None,
        "youngs_modulus": None if args.method == "rigid" else 5e4,
        "poisson_ratio": None if args.method == "rigid" else 0.4,
        "finger_force_n_each": FINGER_FORCE_N,
        "stage_steps": STAGE_STEPS,
        "physical_seconds": sum(STAGE_STEPS.values()) * DT,
        "build_seconds": build_seconds,
        "stages": {},
        "snapshots": {"initial": initial},
        "images": {},
    }
    if camera is not None:
        report["images"]["initial"] = _render(camera, output_dir, run_name, "initial")

    for stage, stage_commands in commands.items():
        report["stages"][stage] = _run_stage(scene, stage_commands)
        snapshot, rest_tet_volumes = _snapshot(obj, args.method, rest_tet_volumes)
        report["snapshots"][stage] = snapshot
        if camera is not None and stage in ("approach", "close", "lift"):
            report["images"][stage] = _render(camera, output_dir, run_name, stage)

    timed_seconds = sum(stage["seconds"] for stage in report["stages"].values())
    total_steps = sum(stage["steps"] for stage in report["stages"].values())
    report["timed_total"] = {
        "steps": total_steps,
        "physical_seconds": total_steps * DT,
        "seconds": timed_seconds,
        "steps_per_second": total_steps / timed_seconds,
        "realtime_factor": total_steps * DT / timed_seconds,
    }

    base_extent = np.asarray(initial["extent"], dtype=float)
    base_center = np.asarray(initial["center"], dtype=float)
    numerically_healthy = True
    geometry_warning = False
    kinematic_warning = False
    for snapshot in report["snapshots"].values():
        extent = np.asarray(snapshot["extent"], dtype=float)
        center = np.asarray(snapshot["center"], dtype=float)
        snapshot["max_extent_ratio_to_initial"] = float(np.nanmax(extent / base_extent))
        snapshot["center_displacement"] = float(np.linalg.norm(center - base_center))
        kinematic_warning |= snapshot["center_displacement"] >= 0.25
        numerically_healthy &= snapshot["finite_fraction"] == 1.0
        if args.method == "mpm":
            numerically_healthy &= snapshot["particles_active"] == snapshot["particles_total"]
            numerically_healthy &= snapshot["det_F_min"] > 0.0 and np.isfinite(snapshot["det_F_max"])
            geometry_warning |= snapshot["max_extent_ratio_to_initial"] >= 3.0
        elif args.method.startswith("fem"):
            numerically_healthy &= snapshot["inverted_elements"] == 0
            geometry_warning |= snapshot["max_extent_ratio_to_initial"] >= 3.0
    report["numerically_healthy"] = bool(numerically_healthy)
    report["geometry_warning"] = bool(geometry_warning)
    report["kinematic_warning"] = bool(kinematic_warning)
    report["healthy"] = bool(numerically_healthy and not geometry_warning and not kinematic_warning)
    report["final_center_z_delta"] = float(
        report["snapshots"]["lift"]["center"][2] - report["snapshots"]["approach"]["center"][2]
    )

    report_path = output_dir / f"{run_name}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"REPORT={report_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=tuple(METHOD_SUBSTEPS), required=True)
    parser.add_argument("--object", default="001_chips_can")
    parser.add_argument("--grid-density", type=float, default=128)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "benchmark_deformable_methods_20260812"))
    run(parser.parse_args())
