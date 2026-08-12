"""Aggregate reports produced by benchmark_deformable_methods.py."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METHODS = ("rigid", "mpm", "fem_sap", "fem_ipc")


def _percentile(values, percentile):
    return float(np.percentile(values, percentile)) if values else None


def main(args):
    input_dir = Path(args.input)
    object_rows = []
    for method in METHODS:
        for path in sorted(input_dir.glob(f"*_{method}.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            snapshots = list(report["snapshots"].values())
            max_extent_ratio = max(snapshot["max_extent_ratio_to_initial"] for snapshot in snapshots)
            max_center_displacement = max(snapshot["center_displacement"] for snapshot in snapshots)
            if method == "mpm":
                numerically_healthy = all(
                    snapshot["finite_fraction"] == 1.0
                    and snapshot["particles_active"] == snapshot["particles_total"]
                    and snapshot["det_F_min"] > 0.0
                    and np.isfinite(snapshot["det_F_max"])
                    for snapshot in snapshots
                )
            elif method.startswith("fem"):
                numerically_healthy = all(
                    snapshot["finite_fraction"] == 1.0 and snapshot["inverted_elements"] == 0 for snapshot in snapshots
                )
            else:
                numerically_healthy = all(snapshot["finite_fraction"] == 1.0 for snapshot in snapshots)
            object_rows.append(
                {
                    "object": report["object"],
                    "method": method,
                    "status": "completed",
                    "geometry_approximation": report["geometry_approximation"],
                    "numerically_healthy": numerically_healthy,
                    "geometry_warning": method != "rigid" and max_extent_ratio >= 3.0,
                    "kinematic_warning": max_center_displacement >= 0.25,
                    "stable": numerically_healthy
                    and (method == "rigid" or max_extent_ratio < 3.0)
                    and max_center_displacement < 0.25,
                    "build_seconds": report["build_seconds"],
                    "step_seconds": report["timed_total"]["seconds"],
                    "realtime_factor": report["timed_total"]["realtime_factor"],
                    "max_extent_ratio": max_extent_ratio,
                    "max_center_displacement": max_center_displacement,
                    "max_inverted_elements": max(
                        snapshot.get("inverted_elements", 0) for snapshot in report["snapshots"].values()
                    ),
                    "min_det_F": min(snapshot.get("det_F_min", 1.0) for snapshot in report["snapshots"].values()),
                    "max_det_F": max(snapshot.get("det_F_max", 1.0) for snapshot in report["snapshots"].values()),
                    "final_center_z_delta": report["final_center_z_delta"],
                }
            )

    columns = list(object_rows[0])
    with (input_dir / "per_object_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(object_rows)

    method_rows = []
    for method in METHODS:
        rows = [row for row in object_rows if row["method"] == method]
        step_seconds = [row["step_seconds"] for row in rows]
        build_seconds = [row["build_seconds"] for row in rows]
        method_rows.append(
            {
                "method": method,
                "status": "completed",
                "runs": len(rows),
                "numerically_healthy_runs": sum(row["numerically_healthy"] for row in rows),
                "numerically_healthy_percent": 100.0 * sum(row["numerically_healthy"] for row in rows) / len(rows),
                "geometry_warning_runs": sum(row["geometry_warning"] for row in rows),
                "kinematic_warning_runs": sum(row["kinematic_warning"] for row in rows),
                "stable_runs": sum(row["stable"] for row in rows),
                "stable_percent": 100.0 * sum(row["stable"] for row in rows) / len(rows),
                "build_seconds_median": _percentile(build_seconds, 50),
                "step_seconds_median": _percentile(step_seconds, 50),
                "step_seconds_min": min(step_seconds),
                "step_seconds_max": max(step_seconds),
                "realtime_factor_median": _percentile([row["realtime_factor"] for row in rows], 50),
            }
        )
    (input_dir / "method_summary.json").write_text(json.dumps(method_rows, indent=2), encoding="utf-8")
    with (input_dir / "method_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(method_rows[0]))
        writer.writeheader()
        writer.writerows(method_rows)

    available = method_rows
    labels = [row["method"].replace("fem_", "FEM+").upper() for row in available]
    colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(labels, [row["step_seconds_median"] for row in available], color=colors)
    axes[0].axhline(2.6, color="black", linestyle="--", linewidth=1, label="real time (2.6 s)")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("wall time for 2.6 s physics [s, log]")
    axes[0].set_title("Median simulation time (10 YCB objects)")
    axes[0].legend()
    x_positions = np.arange(len(labels))
    numerical_bars = axes[1].bar(
        x_positions - 0.18,
        [row["numerically_healthy_percent"] for row in available],
        width=0.36,
        color=colors,
        alpha=0.45,
        label="finite/no loss/no inversion",
    )
    stable_bars = axes[1].bar(
        x_positions + 0.18,
        [row["stable_percent"] for row in available],
        width=0.36,
        color=colors,
        label="also no >0.25 m drift or >3x extent",
    )
    axes[1].bar_label(numerical_bars, fmt="%.0f%%", fontsize=8)
    axes[1].bar_label(stable_bars, fmt="%.0f%%", fontsize=8)
    axes[1].set_xticks(x_positions, labels)
    axes[1].set_ylim(0, 110)
    axes[1].set_ylabel("runs [%]")
    axes[1].set_title("Numerical and gross-motion stability")
    axes[1].legend(fontsize=8)
    fig.suptitle("100 Hz, dt=0.01 s, 2.6 s common grasp trajectory")
    fig.tight_layout()
    fig.savefig(input_dir / "method_comparison.png", dpi=180)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    main(parser.parse_args())
