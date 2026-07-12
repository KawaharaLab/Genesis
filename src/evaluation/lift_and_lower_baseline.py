import argparse
import importlib.util
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import genesis as gs
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PATCHTST_ROOT = ROOT / "external" / "PatchTST_tmp" / "PatchTST_self_supervised"
if str(PATCHTST_ROOT) not in sys.path:
    sys.path.append(str(PATCHTST_ROOT))

LO_PATH = ROOT / "src" / "evaluation" / "lift_and_lower_object.py"
spec = importlib.util.spec_from_file_location("lift_and_lower_object", LO_PATH)
lo = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(lo)

DT = lo.DT
PANDA_XML_PATH = lo.PANDA_XML_PATH
PROJECT_ROOT = lo.ROOT
YCB_WOOD_BLOCK_OBJ_PATH = lo.YCB_WOOD_BLOCK_OBJ_PATH
control_pose = lo.control_pose
get_contact_pairs = lo.get_contact_pairs
get_force_vector_12 = lo.get_force_vector_12
ik_pose = lo.ik_pose
infer_finger_length_from_mjcf = lo.infer_finger_length_from_mjcf
move_ee = lo.move_ee
sample_object_params = lo.sample_object_params
scaled_steps = lo.scaled_steps
set_gripper = lo.set_gripper
sim_step = lo.sim_step
set_grasp_for_obj = lo.set_grasp_for_obj
_scale_to_vec = lo._scale_to_vec


def compute_release_by_spike(
    scene,
    franka,
    end_effector,
    motors_dof,
    fingers_dof,
    hand_idx,
    left_finger_idx,
    obj_entity,
    finger_len,
    obj_xy,
    threshold,
    motion_speed,
    max_wait_steps,
    cam=None,
):
    compute_grasp_heights = lo.compute_grasp_heights
    hold_pose = lo.hold_pose

    x, y = obj_xy
    hover_z = 0.24
    grasp_z, clamp_z, safe_retract_z, lift_z, lower_back_z = compute_grasp_heights(
        franka=franka,
        hand_idx=hand_idx,
        left_finger_idx=left_finger_idx,
        obj_entity=obj_entity,
        finger_len=finger_len,
        hover_z=hover_z,
    )

    move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, hover_z), 0.04, scaled_steps(150, motion_speed), cam, False)
    move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, grasp_z), 0.04, scaled_steps(100, motion_speed), cam, False)
    move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, clamp_z), 0.04, scaled_steps(35, motion_speed), cam, False)
    set_gripper(scene, franka, motors_dof, fingers_dof, 0.04, 0.0, scaled_steps(60, motion_speed), cam, False)
    move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, safe_retract_z), 0.0, scaled_steps(50, motion_speed), cam, False)
    move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, lift_z), 0.0, scaled_steps(120, motion_speed), cam, False)
    hold_pose(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, lift_z), 0.0, int(round(1.0 / DT)), cam, False)
    move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, safe_retract_z), 0.0, scaled_steps(80, motion_speed), cam, False)

    start_pos = franka.get_links_pos([8])[0].cpu().numpy().reshape(3)
    target_pos = np.array([x, y, lower_back_z], dtype=float)
    descend_steps = scaled_steps(140, motion_speed)

    floor_touch_step = None
    release_step = None
    spike_trace = []
    prev_fz = None

    for i in range(descend_steps):
        alpha = (i + 1) / descend_steps
        interp = (1.0 - alpha) * start_pos + alpha * target_pos
        qpos = ik_pose(franka, end_effector, interp)
        control_pose(franka, motors_dof, fingers_dof, qpos, gripper_opening=0.0)
        sim_step(scene, cam=cam, record=False)

        v = get_force_vector_12(franka)
        fz = float(np.abs(v[2]) + np.abs(v[8]))
        spike = 0.0 if prev_fz is None else float(abs(fz - prev_fz))
        prev_fz = fz
        spike_trace.append({"step": int(scene.t), "fz_sum_abs": fz, "spike": spike})

        pairs = get_contact_pairs(obj_entity)
        if (0 in pairs) and floor_touch_step is None:
            floor_touch_step = int(scene.t)

        if spike >= threshold:
            release_step = int(scene.t)
            break

    if release_step is None:
        qpos_hold = ik_pose(franka, end_effector, (x, y, lower_back_z))
        for _ in range(max_wait_steps):
            control_pose(franka, motors_dof, fingers_dof, qpos_hold, gripper_opening=0.0)
            sim_step(scene, cam=cam, record=False)
            v = get_force_vector_12(franka)
            fz = float(np.abs(v[2]) + np.abs(v[8]))
            spike = 0.0 if prev_fz is None else float(abs(fz - prev_fz))
            prev_fz = fz
            spike_trace.append({"step": int(scene.t), "fz_sum_abs": fz, "spike": spike})
            pairs = get_contact_pairs(obj_entity)
            if (0 in pairs) and floor_touch_step is None:
                floor_touch_step = int(scene.t)
            if spike >= threshold:
                release_step = int(scene.t)
                break

    released = release_step is not None
    if released:
        set_gripper(scene, franka, motors_dof, fingers_dof, 0.0, 0.04, scaled_steps(40, motion_speed), None, False)

    delay_sec = None
    if released and floor_touch_step is not None:
        delay_sec = (release_step - floor_touch_step) * DT

    move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, hover_z), 0.04, scaled_steps(100, motion_speed), cam, False)
    return {
        "released": released,
        "release_step": release_step,
        "floor_touch_step": floor_touch_step,
        "floor_touched_at_release": bool(released and floor_touch_step is not None and release_step >= floor_touch_step),
        "delay_sec_after_touch": delay_sec,
        "spike_max": float(max([r["spike"] for r in spike_trace], default=0.0)),
        "spike_trace": spike_trace,
        "spike_trace_tail": spike_trace[-20:],
    }


def run_trial(args, params: dict, threshold: float) -> dict:
    if torch.cuda.is_available():
        gs.init(backend=gs.gpu, logging_level="warning")
    else:
        gs.init(backend=gs.cpu, logging_level="warning")

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(camera_pos=(2.6, -1.1, 1.3), camera_lookat=(0.55, 0.0, 0.20), camera_fov=35, max_FPS=60),
        rigid_options=gs.options.RigidOptions(dt=DT),
        show_viewer=args.show_viewer,
    )
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    cam = scene.add_camera(
        model="thinlens",
        res=(960, 960),
        pos=(1.7, 0.0, 0.7),
        lookat=(0.60, 0.0, 0.16),
        fov=35,
        GUI=False,
    )

    obj_xy = (0.62, 0.0)
    if args.object_type == "cube":
        cube_size = params["size"]
        obj_entity = scene.add_entity(
            material=gs.materials.Rigid(rho=params["rho"], friction=1.0),
            morph=gs.morphs.Box(size=(cube_size, cube_size, cube_size), pos=(obj_xy[0], obj_xy[1], cube_size * 0.5)),
            surface=gs.surfaces.Default(color=(0.35, 0.75, 0.95, 1.0)),
        )
    else:
        object_scale, object_euler, min_corner = set_grasp_for_obj(YCB_WOOD_BLOCK_OBJ_PATH)
        scale_vec = _scale_to_vec(object_scale)
        spawn_z = 0.001 + max(0.0, -scale_vec[2] * min_corner[2])
        obj_entity = scene.add_entity(
            material=gs.materials.Rigid(rho=params["rho"], friction=1.0),
            morph=gs.morphs.Mesh(
                file=str(YCB_WOOD_BLOCK_OBJ_PATH),
                scale=object_scale,
                pos=(obj_xy[0], obj_xy[1], float(spawn_z)),
                euler=object_euler,
            ),
            surface=gs.surfaces.Default(color=(0.0, 1.0, 0.0, 1.0)),
        )

    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build(n_envs=1)

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    end_effector = franka.get_link("hand")
    hand_idx = next(i for i, link in enumerate(franka.links) if link.name == "hand")
    left_finger_idx = next(i for i, link in enumerate(franka.links) if link.name == "left_finger")
    finger_len = infer_finger_length_from_mjcf(PANDA_XML_PATH)

    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
    franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
    franka.set_dofs_force_range(np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]), np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]))

    home_q = ik_pose(franka, end_effector, pos=(0.45, 0.0, 0.28))
    home_q[0, -2:] = 0.04
    franka.set_dofs_position(home_q[0, :-2], motors_dof)
    franka.set_dofs_position(home_q[0, -2:], fingers_dof)
    trial_dir = getattr(args, "_current_trial_dir", None)
    if trial_dir is not None:
        img_dir = Path(trial_dir) / "images" / "camera_0"
        img_dir.mkdir(parents=True, exist_ok=True)
        lo.TRACE_RECORDER["enabled"] = True
        lo.TRACE_RECORDER["images_dir"] = img_dir
        lo.TRACE_RECORDER["image_stride"] = 20
    else:
        lo.TRACE_RECORDER["enabled"] = False

    for _ in range(120):
        control_pose(franka, motors_dof, fingers_dof, home_q, gripper_opening=0.04)
        sim_step(scene, cam=cam, record=False)

    metrics = compute_release_by_spike(
        scene, franka, end_effector, motors_dof, fingers_dof, hand_idx, left_finger_idx, obj_entity, finger_len, obj_xy,
        threshold=threshold, motion_speed=args.motion_speed, max_wait_steps=args.post_descend_wait_steps,
        cam=cam,
    )
    out = {
        "object_params": params,
        "metrics": metrics,
    }
    if trial_dir is not None:
        csv_dir = Path(trial_dir) / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        lo.save_dict_rows_to_csv(metrics["spike_trace"], csv_dir / "spike_trace.csv")
        lo.save_dict_rows_to_csv([metrics], csv_dir / "trial_metrics.csv")
        steps = [r["step"] for r in metrics["spike_trace"]]
        fz = [r["fz_sum_abs"] for r in metrics["spike_trace"]]
        sp = [r["spike"] for r in metrics["spike_trace"]]
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(steps, fz, label="fz_sum_abs")
        ax.plot(steps, sp, label="spike")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(csv_dir / "force_plot.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
    gs.destroy()
    return out


def success_from_metrics(m: dict) -> bool:
    if not m["released"]:
        return False
    if not m["floor_touched_at_release"]:
        return False
    d = m["delay_sec_after_touch"]
    return d is not None and 0.0 <= d <= 0.5


def classify_outcome(m: dict) -> str:
    if not m["released"]:
        return "no_release"
    if not m["floor_touched_at_release"]:
        return "early_release"
    d = m["delay_sec_after_touch"]
    if d is not None and 0.0 <= d <= 0.5:
        return "success"
    return "late_or_other"


def evaluate_threshold(args, threshold: float, trials: int, seed_offset: int) -> dict:
    rng = random.Random(args.seed + seed_offset)
    results = []
    for _ in range(trials):
        params = sample_object_params("cube", rng)
        r = run_trial(args, params, threshold)
        results.append(r)
    counts = {"success": 0, "early_release": 0, "no_release": 0, "late_or_other": 0}
    for r in results:
        counts[classify_outcome(r["metrics"])] += 1
    n = max(1, trials)
    rates = {k: counts[k] / n for k in counts}
    return {
        "threshold": threshold,
        "trials": trials,
        "counts": counts,
        "rates": rates,
    }


def select_threshold_with_constraints(args) -> tuple[float, list[dict], dict]:
    candidates = np.linspace(args.threshold_min, args.threshold_max, args.search_points)
    history = []
    for i, th in enumerate(candidates):
        ev = evaluate_threshold(args, float(th), args.search_trials, seed_offset=1000 + i)
        ev["candidate_index"] = i + 1
        history.append(ev)

    feasible = [
        h
        for h in history
        if h["rates"]["early_release"] <= args.max_early_release_rate and h["rates"]["no_release"] <= args.max_no_release_rate
    ]

    if feasible:
        best = max(feasible, key=lambda h: (h["rates"]["success"], h["threshold"]))
        selection_mode = "feasible_max_success"
    else:
        # Fallback: minimize constraint violation first, then maximize success, then prefer larger threshold.
        def key(h):
            early_excess = max(0.0, h["rates"]["early_release"] - args.max_early_release_rate)
            no_release_excess = max(0.0, h["rates"]["no_release"] - args.max_no_release_rate)
            violation = early_excess + no_release_excess
            return (-violation, h["rates"]["success"], h["threshold"])

        best = max(history, key=key)
        selection_mode = "fallback_min_violation"

    return float(best["threshold"]), history, {"selection_mode": selection_mode, "selected_record": best}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "data" / "evaluation_outputs"))
    parser.add_argument("--object-type", choices=["cube", "ycb_wood_block"], default="ycb_wood_block")
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--motion-speed", type=float, default=1.0)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--post-descend-wait-steps", type=int, default=1200)
    parser.add_argument("--search-points", type=int, default=10)
    parser.add_argument("--search-trials", type=int, default=5)
    parser.add_argument("--threshold-min", type=float, default=0.1)
    parser.add_argument("--threshold-max", type=float, default=10.0)
    parser.add_argument("--max-early-release-rate", type=float, default=0.10)
    parser.add_argument("--max-no-release-rate", type=float, default=0.20)
    args = parser.parse_args()

    chosen, history, selection_meta = select_threshold_with_constraints(args)
    print("threshold search history:")
    for h in history:
        print(
            f"idx={h['candidate_index']} th={h['threshold']:.4f} "
            f"succ={h['rates']['success']:.3f} early={h['rates']['early_release']:.3f} "
            f"norelease={h['rates']['no_release']:.3f}"
        )
    print(f"selection_mode={selection_meta['selection_mode']} chosen_threshold={chosen:.4f}")

    rng = random.Random(args.seed)
    run_root = Path(args.output_dir) / f"lift_and_lower_baseline_trials_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    final_results = []
    for i in range(args.trials):
        trial_dir = run_root / f"trial_{i+1:02d}"
        args._current_trial_dir = trial_dir
        if args.object_type == "cube":
            params = sample_object_params("cube", rng)
        else:
            params = {"rho": float(200.0 * rng.uniform(1.0, 10.0)), "size": None, "scale": None}
        r = run_trial(args, params, chosen)
        final_results.append(r)
        print(f"[trial {i+1:02d}/{args.trials}] released={r['metrics']['released']} floor_at_release={r['metrics']['floor_touched_at_release']} delay={r['metrics']['delay_sec_after_touch']}")

    success = sum(1 for r in final_results if success_from_metrics(r["metrics"]))
    no_release = sum(1 for r in final_results if not r["metrics"]["released"])
    early_release = sum(
        1 for r in final_results if r["metrics"]["released"] and not r["metrics"]["floor_touched_at_release"]
    )
    late_or_other = args.trials - success - no_release - early_release
    payload = {
        "method": "fz_spike_baseline",
        "threshold_search": {
            "points": args.search_points,
            "search_trials": args.search_trials,
            "constraints": {
                "max_early_release_rate": args.max_early_release_rate,
                "max_no_release_rate": args.max_no_release_rate,
            },
            "selection_mode": selection_meta["selection_mode"],
            "selected_record": selection_meta["selected_record"],
            "history": history,
            "chosen_threshold": chosen,
        },
        "final_eval": {
            "trials": args.trials,
            "success_count": success,
            "success_rate": success / max(1, args.trials),
        },
        "trial_results": final_results,
        "summary": {
            "trials": args.trials,
            "success_count": success,
            "success_rate": success / max(1, args.trials),
            "no_release_count": no_release,
            "early_release_count": early_release,
            "late_or_other_count": late_or_other,
        },
    }

    out_dir = Path(args.output_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"lift_and_lower_baseline_cube_summary_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
