import argparse
import importlib.util
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import genesis as gs

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PATCHTST_ROOT = ROOT / "external" / "PatchTST_tmp" / "PatchTST_self_supervised"
if str(PATCHTST_ROOT) not in sys.path:
    sys.path.append(str(PATCHTST_ROOT))

PH_PATH = ROOT / "src" / "evaluation" / "place_heavier_with_patchtst.py"
spec = importlib.util.spec_from_file_location("place_heavier_with_patchtst", PH_PATH)
ph = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(ph)

DT = ph.DT
PANDA_XML_PATH = ph.PANDA_XML_PATH
PROJECT_ROOT = ph.ROOT


def score_by_fz(force_seq: np.ndarray, aggregation: str = "mean") -> float:
    # 12D = [fLxyz,tLxyz,fRxyz,tRxyz], use |Fz| of both fingers.
    per_step = np.abs(force_seq[:, 2]) + np.abs(force_seq[:, 8])
    if aggregation == "max":
        return float(np.max(per_step))
    return float(np.mean(per_step))


def save_force_plot_pair(left_force: np.ndarray, right_force: np.ndarray, out_path: Path):
    l = np.asarray(left_force, dtype=np.float32)
    r = np.asarray(right_force, dtype=np.float32)
    t = np.arange(min(len(l), len(r)), dtype=np.int32)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(t, l[: len(t), 2], label="left_fz")
    ax.plot(t, r[: len(t), 2], label="right_fz")
    ax.set_xlabel("Step")
    ax.set_ylabel("Force (N)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_trial(trial_idx: int, args, params: dict, run_root: Path) -> dict:
    trial_dir = run_root / f"trial_{trial_idx + 1:02d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = trial_dir / "csv"
    img_dir = trial_dir / "images" / "camera_0"
    csv_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    ph.SIM_STEP_COUNT = 0
    ph.TRACE_RECORDER["enabled"] = True
    ph.TRACE_RECORDER["images_dir"] = img_dir
    ph.TRACE_RECORDER["image_stride"] = 20

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.6, -1.1, 1.3), camera_lookat=(0.55, 0.0, 0.20), camera_fov=35, max_FPS=60
        ),
        rigid_options=gs.options.RigidOptions(dt=DT),
        show_viewer=args.show_viewer,
    )
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

    left_xy = (0.62, -0.14)
    right_xy = (0.62, 0.14)
    tile_xy = (0.45, 0.38)

    cube_size = params["size"]
    left_obj = scene.add_entity(
        material=gs.materials.Rigid(rho=params["left_rho"], friction=1.0),
        morph=gs.morphs.Box(size=(cube_size, cube_size, cube_size), pos=(left_xy[0], left_xy[1], cube_size * 0.5)),
        surface=gs.surfaces.Default(color=(0.3, 0.7, 0.9, 1.0)),
    )
    right_obj = scene.add_entity(
        material=gs.materials.Rigid(rho=params["right_rho"], friction=1.0),
        morph=gs.morphs.Box(size=(cube_size, cube_size, cube_size), pos=(right_xy[0], right_xy[1], cube_size * 0.5)),
        surface=gs.surfaces.Default(color=(0.9, 0.6, 0.3, 1.0)),
    )

    scene.add_entity(
        material=gs.materials.Rigid(rho=800, friction=1.0),
        morph=gs.morphs.Box(size=(0.16, 0.16, 0.004), pos=(tile_xy[0], tile_xy[1], 0.002), fixed=True),
        surface=gs.surfaces.Default(color=(0.1, 0.35, 0.9, 1.0)),
    )

    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    cam = scene.add_camera(model="thinlens", res=(960, 960), pos=(1.7, 0.0, 0.7), lookat=(0.60, 0.0, 0.16), fov=35, GUI=False)
    scene.build(n_envs=1)

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    end_effector = franka.get_link("hand")
    hand_idx = next(i for i, link in enumerate(franka.links) if link.name == "hand")
    left_finger_idx = next(i for i, link in enumerate(franka.links) if link.name == "left_finger")
    finger_len = ph.infer_finger_length_from_mjcf(PANDA_XML_PATH)

    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
    franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
    franka.set_dofs_force_range(np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]), np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]))

    home_q = ph.ik_pose(franka, end_effector, pos=(0.45, 0.0, 0.28))
    home_q[0, -2:] = 0.04
    franka.set_dofs_position(home_q[0, :-2], motors_dof)
    franka.set_dofs_position(home_q[0, -2:], fingers_dof)

    for _ in range(120):
        ph.control_pose(franka, motors_dof, fingers_dof, home_q, gripper_opening=0.04)
        ph.sim_step(scene, cam=cam, record=False)

    left_result = ph.pick_lift_hold_and_putback(
        scene, franka, end_effector, motors_dof, fingers_dof, hand_idx, left_finger_idx, left_obj, finger_len, left_xy, args.motion_speed, cam, False
    )
    right_result = ph.pick_lift_hold_and_putback(
        scene, franka, end_effector, motors_dof, fingers_dof, hand_idx, left_finger_idx, right_obj, finger_len, right_xy, args.motion_speed, cam, False
    )

    left_force = left_result["hold_seq"] if args.judge_timing == "hold" else left_result["lift_seq"]
    right_force = right_result["hold_seq"] if args.judge_timing == "hold" else right_result["lift_seq"]
    left_score = score_by_fz(left_force, args.fz_aggregation)
    right_score = score_by_fz(right_force, args.fz_aggregation)
    ph.save_matrix_csv(left_result["hold_seq"], csv_dir / "left_hold_force_seq.csv")
    ph.save_matrix_csv(right_result["hold_seq"], csv_dir / "right_hold_force_seq.csv")
    ph.save_matrix_csv(left_result["lift_seq"], csv_dir / "left_lift_force_seq.csv")
    ph.save_matrix_csv(right_result["lift_seq"], csv_dir / "right_lift_force_seq.csv")
    ph.save_matrix_csv(left_force, csv_dir / "left_force_seq_for_judgement.csv")
    ph.save_matrix_csv(right_force, csv_dir / "right_force_seq_for_judgement.csv")
    save_force_plot_pair(left_force, right_force, csv_dir / "force_plot.png")
    predicted_heavy = "left" if left_score >= right_score else "right"
    gt_heavy = "left" if params["left_mass_g"] >= params["right_mass_g"] else "right"
    heavy_selection_correct = predicted_heavy == gt_heavy

    if predicted_heavy == "left":
        ph.pick_and_place_to_tile(scene, franka, end_effector, motors_dof, fingers_dof, hand_idx, left_finger_idx, left_obj, finger_len, left_xy, tile_xy, args.motion_speed, cam, False)
        placed_obj = left_obj
    else:
        ph.pick_and_place_to_tile(scene, franka, end_effector, motors_dof, fingers_dof, hand_idx, left_finger_idx, right_obj, finger_len, right_xy, tile_xy, args.motion_speed, cam, False)
        placed_obj = right_obj

    ph.move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (0.45, 0.0, 0.30), 0.04, ph.scaled_steps(120, args.motion_speed), cam, False)
    for _ in range(40):
        ph.sim_step(scene, cam=cam, record=False)

    px, py = ph.get_obj_xy(placed_obj)
    dist = float(np.linalg.norm(np.array([px - tile_xy[0], py - tile_xy[1]], dtype=np.float32)))
    placed_at_target = dist <= 0.10
    task_success = bool(heavy_selection_correct and placed_at_target)

    out = {
        "trial_index": trial_idx + 1,
        "left_mass_g": params["left_mass_g"],
        "right_mass_g": params["right_mass_g"],
        "judge_timing": args.judge_timing,
        "fz_aggregation": args.fz_aggregation,
        "left_fz_score": left_score,
        "right_fz_score": right_score,
        "predicted_heavy_side": predicted_heavy,
        "ground_truth_heavy_side": gt_heavy,
        "heavy_selection_correct": heavy_selection_correct,
        "placed_at_target": placed_at_target,
        "place_distance_to_target_xy": dist,
        "task_success": task_success,
    }
    ph.save_dict_rows_to_csv([out], csv_dir / "trial_scores.csv")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "data" / "evaluation_outputs"))
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--motion-speed", type=float, default=1.8)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--judge-timing",
        choices=["hold", "lift"],
        default="hold",
        help="Use hold phase or lift phase force sequence for heavy/light judgement.",
    )
    parser.add_argument(
        "--fz-aggregation",
        choices=["mean", "max"],
        default="mean",
        help="Aggregate per-step |Fz_left|+|Fz_right| by mean or max.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = output_dir / f"place_heavier_baseline_cube_trials_{ts}"
    run_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    trials = []
    for i in range(args.trials):
        if torch.cuda.is_available():
            gs.init(backend=gs.gpu, logging_level="warning")
        else:
            gs.init(backend=gs.cpu, logging_level="warning")
        params = ph.sample_object_params("cube", rng)
        result = run_trial(i, args, params, run_root)
        gs.destroy()
        trials.append(result)
        print(f"[trial {i+1:02d}/{args.trials}] pred={result['predicted_heavy_side']} gt={result['ground_truth_heavy_side']} select_ok={result['heavy_selection_correct']} place_ok={result['placed_at_target']}")

    acc = sum(1 for t in trials if t["heavy_selection_correct"]) / max(1, len(trials))
    place = sum(1 for t in trials if t["placed_at_target"]) / max(1, len(trials))
    task = sum(1 for t in trials if t["task_success"]) / max(1, len(trials))
    payload = {
        "method": "fz_compare_baseline",
        "judge_timing": args.judge_timing,
        "fz_aggregation": args.fz_aggregation,
        "trials": args.trials,
        "seed": args.seed,
        "summary": {
            "heavy_selection_accuracy": acc,
            "place_success_rate": place,
            "task_success_rate": task,
        },
        "trial_results": trials,
    }
    out = run_root / "summary.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
