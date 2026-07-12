import argparse
import importlib.util
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PATCHTST_ROOT = ROOT / "external" / "PatchTST_tmp" / "PatchTST_self_supervised"
if str(PATCHTST_ROOT) not in sys.path:
    sys.path.insert(0, str(PATCHTST_ROOT))

SD_PATH = ROOT / "src" / "evaluation" / "slip_detection_with_patchtst.py"
spec = importlib.util.spec_from_file_location("slip_detection_with_patchtst", SD_PATH)
sd = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(sd)
ph = sd.ph
run_trial_collect_seq = sd.run_trial_collect_seq


def slip_score_from_fz(seq_12: np.ndarray) -> float:
    fz = np.abs(seq_12[:, 2]) + np.abs(seq_12[:, 8])
    dz = np.diff(fz, prepend=fz[:1])
    return float(np.std(dz))


def evaluate_threshold(args, threshold, trials, seed):
    rng = random.Random(seed)
    rows = []
    for _ in range(trials):
        payload = run_trial_collect_seq(args, rng, trial_dir=None)
        score = slip_score_from_fz(payload["force_seq"])
        pred = "slip" if score > threshold else "stable"
        rows.append({"gt_state": payload["gt_state"], "pred_state": pred, "correct": pred == payload["gt_state"], "score": score})
    acc = sum(1 for r in rows if r["correct"]) / max(1, len(rows))
    return {"threshold": threshold, "accuracy": acc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "data" / "evaluation_outputs"))
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--motion-speed", type=float, default=1.4)
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--search-trials", type=int, default=8)
    parser.add_argument("--threshold-min", type=float, default=0.01)
    parser.add_argument("--threshold-max", type=float, default=5.0)
    parser.add_argument("--search-points", type=int, default=9)
    args = parser.parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_dir) / f"slip_detection_baseline_trials_{ts}"
    run_root.mkdir(parents=True, exist_ok=True)

    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.search_points)
    hist = [evaluate_threshold(args, float(th), args.search_trials, args.seed + i) for i, th in enumerate(thresholds)]
    best = max(hist, key=lambda x: x["accuracy"])

    rng = random.Random(args.seed)
    rows = []
    for i in range(args.trials):
        trial_dir = run_root / f"trial_{i+1:02d}"
        payload = run_trial_collect_seq(args, rng, trial_dir=trial_dir)
        score = slip_score_from_fz(payload["force_seq"])
        pred = "slip" if score > best["threshold"] else "stable"
        correct = pred == payload["gt_state"]
        row = {"gt_state": payload["gt_state"], "pred_state": pred, "correct": correct, "score": score, "trial_dir": str(trial_dir)}
        rows.append(row)
        sd.ph.save_dict_rows_to_csv([row], trial_dir / "csv" / "trial_scores.csv")
        print(f"[trial {i+1:02d}/{args.trials}] gt={payload['gt_state']} pred={pred} correct={correct}")

    acc = sum(1 for r in rows if r["correct"]) / max(1, len(rows))
    out = {
        "task": "slip_detection_baseline",
        "threshold_search": {"history": hist, "chosen_threshold": best["threshold"]},
        "summary": {"accuracy": acc},
        "trial_results": rows,
    }
    out_path = Path(args.output_dir) / f"slip_detection_baseline_summary_{ts}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["summary"], ensure_ascii=False, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
