"""Benchmark an identical Franka grasp trajectory with rigid and MPM-elastic objects."""

import argparse
import json
import sys
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

import genesis as gs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERATION_DIR = PROJECT_ROOT / "src" / "generation"
sys.path.insert(0, str(GENERATION_DIR))

from run_genesis_sim import get_obj_bounds, set_grasp


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _snapshot(entity, material):
    if material == "elastic":
        state = entity.get_state()
        pos = state.pos.detach().cpu().numpy().reshape(-1, 3)
        vel = state.vel.detach().cpu().numpy().reshape(-1, 3)
        active = state.active.detach().cpu().numpy().reshape(-1).astype(bool)
        F = state.F.detach().cpu().numpy().reshape(-1, 3, 3)
        pos, vel, F = pos[active], vel[active], F[active]
        finite = np.isfinite(pos).all(axis=1) & np.isfinite(vel).all(axis=1) & np.isfinite(F).all(axis=(1, 2))
        valid_pos = pos[finite]
        extent = np.ptp(valid_pos, axis=0) if len(valid_pos) else np.full(3, np.nan)
        det_f = np.linalg.det(F[finite]) if finite.any() else np.array([np.nan])
        return {
            "particles_total": int(active.size),
            "particles_active": int(active.sum()),
            "finite_fraction": float(finite.mean()) if len(finite) else 0.0,
            "bbox_min": np.min(valid_pos, axis=0).tolist() if len(valid_pos) else [None] * 3,
            "bbox_max": np.max(valid_pos, axis=0).tolist() if len(valid_pos) else [None] * 3,
            "extent": extent.tolist(),
            "max_speed": float(np.linalg.norm(vel[finite], axis=1).max()) if finite.any() else None,
            "det_F_min": float(np.nanmin(det_f)),
            "det_F_max": float(np.nanmax(det_f)),
        }

    bounds = entity.get_AABB().detach().cpu().numpy()
    return {
        "finite_fraction": float(np.isfinite(bounds).all()),
        "bbox_min": bounds[0].tolist(),
        "bbox_max": bounds[1].tolist(),
        "extent": (bounds[1] - bounds[0]).tolist(),
    }


def _render(camera, output_dir, run_name, stage):
    rgb, _, _, _ = camera.render(rgb=True)
    path = output_dir / f"{run_name}_{stage}.png"
    iio.imwrite(path, rgb)
    return str(path.resolve())


def _timed_stage(scene, n_steps, command):
    _sync()
    start = time.perf_counter()
    for step in range(n_steps):
        command(step, n_steps)
        scene.step()
    _sync()
    elapsed = time.perf_counter() - start
    return {"steps": n_steps, "seconds": elapsed, "steps_per_second": n_steps / elapsed}


def run(args):
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        gs.init(backend=gs.gpu)
    else:
        gs.init(backend=gs.cpu)

    mesh_path = PROJECT_ROOT / "data" / "objects" / "ycb" / args.object / "model.obj"
    scale, euler = set_grasp(mesh_path)
    mesh_min, _ = get_obj_bounds(mesh_path)
    scale_vec = np.full(3, scale) if np.isscalar(scale) else np.asarray(scale)
    center_xy = np.array([0.45, 0.45])
    spawn_z = 0.002 + max(0.0, -scale_vec[2] * mesh_min[2])
    spawn = (float(center_xy[0]), float(center_xy[1]), float(spawn_z))

    if args.material == "elastic":
        dt = 1e-3 if args.dt is None else args.dt
        substeps = 10 if args.substeps is None else args.substeps
        mpm_options = gs.options.MPMOptions(
            lower_bound=(0.0, -0.1, -0.05),
            upper_bound=(0.75, 1.0, 1.0),
            grid_density=args.grid_density,
        )
    else:
        dt = 1e-2 if args.dt is None else args.dt
        substeps = 1 if args.substeps is None else args.substeps
        mpm_options = None
    sim_options = gs.options.SimOptions(dt=dt, substeps=substeps)

    scene = gs.Scene(sim_options=sim_options, mpm_options=mpm_options, show_viewer=False, show_FPS=False)
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(coup_friction=0.0))
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        material=gs.materials.Rigid(coup_friction=3.0, friction=1.0),
    )
    obj_material = gs.materials.MPM.Elastic() if args.material == "elastic" else gs.materials.Rigid(rho=1000.0)
    obj = scene.add_entity(
        material=obj_material,
        morph=gs.morphs.Mesh(file=str(mesh_path), scale=scale, pos=spawn, euler=euler),
        surface=gs.surfaces.Default(color=(0.1, 0.8, 0.1, 1.0)),
    )
    camera = scene.add_camera(res=(800, 600), pos=(1.1, 1.1, 0.55), lookat=(0.45, 0.45, 0.18), fov=35)

    _sync()
    build_start = time.perf_counter()
    scene.build()
    _sync()
    build_seconds = time.perf_counter() - build_start

    franka.set_dofs_kp(np.array([450, 450, 350, 350, 200, 200, 200, 100, 100]))
    franka.set_dofs_kv(np.array([45, 45, 35, 35, 20, 20, 20, 10, 10]))
    hand = franka.get_link("hand")
    initial = _snapshot(obj, args.material)
    object_top = initial["bbox_max"][2]
    hover_z = object_top + 0.16
    grasp_z = object_top + 0.065

    def ik(z, finger):
        q = franka.inverse_kinematics(
            link=hand, pos=np.array([center_xy[0], center_xy[1], z]), quat=np.array([0.0, 1.0, 0.0, 0.0])
        )
        q[-2:] = finger
        return q

    q_hover = ik(hover_z, 0.04)
    franka.set_dofs_position(q_hover, zero_velocity=True)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{args.object}_{args.material}_dt{dt:g}_substeps{substeps}"
    if args.material == "elastic":
        run_name += f"_grid{args.grid_density:g}"
    if args.step_scale != 1.0:
        run_name += f"_stepscale{args.step_scale:g}"

    def scaled_steps(n_steps):
        return max(1, round(n_steps * args.step_scale))

    report = {
        "object": args.object,
        "material": args.material,
        "dt": float(sim_options.dt),
        "substeps": int(sim_options.substeps),
        "build_seconds": build_seconds,
        "stages": {},
        "snapshots": {"initial": initial},
        "images": {"initial": _render(camera, output_dir, run_name, "initial")},
    }

    report["stages"]["settle"] = _timed_stage(
        scene, scaled_steps(30), lambda _i, _n: franka.control_dofs_position(q_hover)
    )
    report["snapshots"]["settled"] = _snapshot(obj, args.material)

    def approach(i, n):
        alpha = (i + 1) / n
        franka.control_dofs_position(ik((1 - alpha) * hover_z + alpha * grasp_z, 0.04))

    report["stages"]["approach"] = _timed_stage(scene, scaled_steps(80), approach)
    report["snapshots"]["pre_contact"] = _snapshot(obj, args.material)
    report["images"]["pre_contact"] = _render(camera, output_dir, run_name, "pre_contact")

    def close(i, n):
        alpha = (i + 1) / n
        franka.control_dofs_position(ik(grasp_z, 0.04 * (1 - alpha)))

    report["stages"]["close"] = _timed_stage(scene, scaled_steps(120), close)
    report["snapshots"]["after_close"] = _snapshot(obj, args.material)
    report["images"]["after_close"] = _render(camera, output_dir, run_name, "after_close")

    q_closed = ik(grasp_z, 0.0)
    report["stages"]["hold"] = _timed_stage(
        scene, scaled_steps(80), lambda _i, _n: franka.control_dofs_position(q_closed)
    )
    report["snapshots"]["after_hold"] = _snapshot(obj, args.material)

    def lift(i, n):
        alpha = (i + 1) / n
        franka.control_dofs_position(ik(grasp_z + 0.12 * alpha, 0.0))

    report["stages"]["lift"] = _timed_stage(scene, scaled_steps(100), lift)
    report["snapshots"]["after_lift"] = _snapshot(obj, args.material)
    report["images"]["after_lift"] = _render(camera, output_dir, run_name, "after_lift")

    timed_seconds = sum(stage["seconds"] for stage in report["stages"].values())
    timed_steps = sum(stage["steps"] for stage in report["stages"].values())
    report["timed_total"] = {
        "steps": timed_steps,
        "seconds": timed_seconds,
        "steps_per_second": timed_steps / timed_seconds,
    }
    base_extent = np.asarray(initial["extent"], dtype=float)
    for snapshot in report["snapshots"].values():
        extent = np.asarray(snapshot["extent"], dtype=float)
        snapshot["max_extent_ratio_to_initial"] = float(np.nanmax(extent / base_extent))

    report_path = output_dir / f"{run_name}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"REPORT={report_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", choices=("rigid", "elastic"), required=True)
    parser.add_argument("--object", default="004_sugar_box")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--substeps", type=int)
    parser.add_argument("--grid-density", type=float, default=128)
    parser.add_argument("--step-scale", type=float, default=1.0)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "benchmark_rigid_elastic_20260812"))
    run(parser.parse_args())
