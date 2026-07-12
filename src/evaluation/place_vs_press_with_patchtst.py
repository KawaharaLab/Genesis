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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PATCHTST_ROOT = ROOT / "external" / "PatchTST_tmp" / "PatchTST_self_supervised"
if str(PATCHTST_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCHTST_ROOT))

PH_PATH = ROOT / "src" / "evaluation" / "place_heavier_with_patchtst.py"
spec = importlib.util.spec_from_file_location("place_heavier_with_patchtst", PH_PATH)
ph = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(ph)

DT = ph.DT
SEQ_LEN = ph.SEQ_LEN
DEFAULT_CKPT = ph.DEFAULT_CKPT


def _pad_to_seq(arr: np.ndarray) -> np.ndarray:
    if arr.shape[0] < SEQ_LEN:
        pad = np.repeat(arr[-1:, :], SEQ_LEN - arr.shape[0], axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    return arr[-SEQ_LEN:, :]


def _save_force_plot(force_seq: np.ndarray, out_path: Path):
    arr = np.asarray(force_seq, dtype=np.float32)
    t = np.arange(arr.shape[0], dtype=np.int32)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(t, arr[:, 2], label="left_fz")
    ax.plot(t, arr[:, 8], label="right_fz")
    ax.set_xlabel("Step")
    ax.set_ylabel("Force (N)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_trial_collect_seq(args, rng: random.Random, trial_dir: Path | None = None) -> dict:
    if torch.cuda.is_available():
        ph.gs.init(backend=ph.gs.gpu, logging_level="warning")
    else:
        ph.gs.init(backend=ph.gs.cpu, logging_level="warning")

    ph.SIM_STEP_COUNT = 0
    if trial_dir is not None:
        img_dir = trial_dir / "images" / "camera_0"
        img_dir.mkdir(parents=True, exist_ok=True)
        ph.TRACE_RECORDER["enabled"] = True
        ph.TRACE_RECORDER["images_dir"] = img_dir
        ph.TRACE_RECORDER["image_stride"] = 20
    else:
        ph.TRACE_RECORDER["enabled"] = False

    scene = ph.gs.Scene(
        viewer_options=ph.gs.options.ViewerOptions(camera_pos=(2.6, -1.1, 1.3), camera_lookat=(0.55, 0.0, 0.20), camera_fov=35, max_FPS=60),
        rigid_options=ph.gs.options.RigidOptions(dt=DT),
        show_viewer=args.show_viewer,
    )
    scene.add_entity(ph.gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

    src_xy = (0.62, 0.0)
    tile_xy = (0.45, 0.38)

    size = float(rng.uniform(0.03, 0.075))
    rho = float(200.0 * rng.uniform(1.0, 10.0))
    obj = scene.add_entity(
        material=ph.gs.materials.Rigid(rho=rho, friction=1.0),
        morph=ph.gs.morphs.Box(size=(size, size, size), pos=(src_xy[0], src_xy[1], size * 0.5)),
        surface=ph.gs.surfaces.Default(color=(0.35, 0.75, 0.95, 1.0)),
    )
    scene.add_entity(
        material=ph.gs.materials.Rigid(rho=800, friction=1.0),
        morph=ph.gs.morphs.Box(size=(0.16, 0.16, 0.004), pos=(tile_xy[0], tile_xy[1], 0.002), fixed=True),
        surface=ph.gs.surfaces.Default(color=(0.1, 0.35, 0.9, 1.0)),
    )

    franka = scene.add_entity(ph.gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    cam = scene.add_camera(
        model="thinlens",
        res=(960, 960),
        pos=(1.7, 0.0, 0.7),
        lookat=(0.60, 0.0, 0.16),
        fov=35,
        GUI=False,
    )
    scene.build(n_envs=1)

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    end_effector = franka.get_link("hand")
    hand_idx = next(i for i, link in enumerate(franka.links) if link.name == "hand")
    left_finger_idx = next(i for i, link in enumerate(franka.links) if link.name == "left_finger")
    finger_len = ph.infer_finger_length_from_mjcf(ph.PANDA_XML_PATH)

    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
    franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
    franka.set_dofs_force_range(np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]), np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]))

    home_q = ph.ik_pose(franka, end_effector, pos=(0.45, 0.0, 0.28))
    home_q[0, -2:] = 0.04
    franka.set_dofs_position(home_q[0, :-2], motors_dof)
    franka.set_dofs_position(home_q[0, -2:], fingers_dof)
    for _ in range(120):
        ph.control_pose(franka, motors_dof, fingers_dof, home_q, gripper_opening=0.04)
        ph.sim_step(scene, cam, False)

    hover_z, grasp_z, clamp_z, safe_retract_z, lift_z = ph.compute_grasp_heights(franka, hand_idx, left_finger_idx, obj, finger_len)
    x, y = src_xy
    ph.move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, hover_z), 0.04, ph.scaled_steps(120, args.motion_speed), cam, False)
    ph.move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, grasp_z), 0.04, ph.scaled_steps(100, args.motion_speed), cam, False)
    ph.move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, clamp_z), 0.04, ph.scaled_steps(35, args.motion_speed), cam, False)
    ph.set_gripper(scene, franka, motors_dof, fingers_dof, 0.04, 0.0, ph.scaled_steps(60, args.motion_speed), cam, False)
    ph.move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, safe_retract_z), 0.0, ph.scaled_steps(50, args.motion_speed), cam, False)
    ph.move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (x, y, lift_z), 0.0, ph.scaled_steps(100, args.motion_speed), cam, False)

    tx, ty = tile_xy
    ph.move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (tx, ty, lift_z), 0.0, ph.scaled_steps(160, args.motion_speed), cam, False)
    place_z = max(clamp_z, 0.09)
    ph.move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (tx, ty, place_z), 0.0, ph.scaled_steps(120, args.motion_speed), cam, False)

    gt_action = rng.choice(["place", "press"])
    seq = []
    if gt_action == "press":
        press_depth = 0.03
        press_steps = 120
        for step in range(press_steps):
            z_target = place_z - press_depth * (step + 1) / press_steps
            qpos = ph.ik_pose(franka, end_effector, (tx, ty, z_target))
            ph.control_pose(franka, motors_dof, fingers_dof, qpos, gripper_opening=0.0)
            ph.sim_step(scene, cam, False)
            seq.append(ph.get_force_vector_12(franka))
    else:
        # keep slight contact window for place signal
        qpos = ph.ik_pose(franka, end_effector, (tx, ty, place_z))
        for _ in range(max(20, int(0.2 / DT))):
            ph.control_pose(franka, motors_dof, fingers_dof, qpos, gripper_opening=0.0)
            ph.sim_step(scene, cam, False)
            seq.append(ph.get_force_vector_12(franka))

    ph.set_gripper(scene, franka, motors_dof, fingers_dof, 0.0, 0.04, ph.scaled_steps(60, args.motion_speed), cam, False)
    for _ in range(max(60, int(0.6 / DT))):
        ph.sim_step(scene, cam, False)
        seq.append(ph.get_force_vector_12(franka))
    ph.move_ee(scene, franka, end_effector, motors_dof, fingers_dof, (tx, ty, lift_z), 0.04, ph.scaled_steps(90, args.motion_speed), cam, False)

    arr = np.asarray(seq, dtype=np.float32)
    arr = _pad_to_seq(arr)
    out = {"gt_action": gt_action, "force_seq": arr, "rho": rho, "size": size}
    if trial_dir is not None:
        csv_dir = trial_dir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        ph.save_matrix_csv(arr, csv_dir / "force_seq_for_judgement.csv")
        _save_force_plot(arr, csv_dir / "force_plot.png")
    ph.gs.destroy()
    return out


def run_trial_encoder(args, model, proj, place_vec, press_vec, rng: random.Random):
    payload = run_trial_collect_seq(args, rng)
    emb = ph.embed_force_segment(model, proj, payload["force_seq"])
    place_score = float(torch.dot(emb, place_vec))
    press_score = float(torch.dot(emb, press_vec))
    pred_action = "press" if press_score > place_score else "place"
    return {
        "gt_action": payload["gt_action"],
        "pred_action": pred_action,
        "place_score": place_score,
        "press_score": press_score,
        "correct": bool(pred_action == payload["gt_action"]),
        "rho": payload["rho"],
        "size": payload["size"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "data" / "evaluation_outputs"))
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--motion-speed", type=float, default=1.4)
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--show-viewer", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, proj = ph.load_contrastive_patchtst(Path(args.model_path), device)
    place_vec = ph.load_text_embedding("place")
    press_vec = ph.load_text_embedding("press")

    rng = random.Random(args.seed)
    rows = []
    run_root = Path(args.output_dir) / f"place_vs_press_patchtst_trials_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    for i in range(args.trials):
        trial_dir = run_root / f"trial_{i+1:02d}"
        payload = run_trial_collect_seq(args, rng, trial_dir=trial_dir)
        emb = ph.embed_force_segment(model, proj, payload["force_seq"])
        place_score = float(torch.dot(emb, place_vec))
        press_score = float(torch.dot(emb, press_vec))
        pred_action = "press" if press_score > place_score else "place"
        r = {
            "gt_action": payload["gt_action"],
            "pred_action": pred_action,
            "place_score": place_score,
            "press_score": press_score,
            "correct": bool(pred_action == payload["gt_action"]),
            "rho": payload["rho"],
            "size": payload["size"],
            "trial_dir": str(trial_dir),
        }
        ph.save_dict_rows_to_csv([r], trial_dir / "csv" / "trial_scores.csv")
        rows.append(r)
        print(f"[trial {i+1:02d}/{args.trials}] gt={r['gt_action']} pred={r['pred_action']} correct={r['correct']}")

    acc = sum(1 for r in rows if r["correct"]) / max(1, len(rows))
    out = {
        "task": "place_vs_press_classification",
        "trials": args.trials,
        "seed": args.seed,
        "summary": {"accuracy": acc},
        "trial_results": rows,
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output_dir) / f"place_vs_press_patchtst_summary_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["summary"], ensure_ascii=False, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
