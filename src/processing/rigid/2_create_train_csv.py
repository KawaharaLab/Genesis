import glob
import os
import sys
from pathlib import Path

import pandas as pd

from force_window_filter import filter_all_zero_force_windows

MODE = os.environ.get("MODE", "train_21072026")  # "train" or "eval"
DATA_DIR = f"/home/user/Genesis/data/{MODE}"
out_path = f"{DATA_DIR}/{MODE}.csv"

def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "action" not in df.columns:
        df["action"] = None
    if "interaction" not in df.columns:
        df["interaction"] = None
    if "weight" not in df.columns:
        df["weight"] = None
    if "placement_outcome" not in df.columns:
        df["placement_outcome"] = None

    df["action"] = df["action"].fillna("").astype(str).str.strip()
    df["interaction"] = df["interaction"].fillna("").astype(str).str.strip()
    df["weight"] = df["weight"].fillna("").astype(str).str.strip()
    df["placement_outcome"] = df["placement_outcome"].fillna("").astype(str).str.strip().str.lower()

    def _fallback_action(annotation: str) -> str:
        text = (annotation or "").lower()
        if "accidental drop" in text:
            return "accidental drop"
        if "pressing" in text and "then releasing" in text:
            return "press then release"
        if "pressing" in text:
            return "press"
        if "bumping" in text:
            return "bump"
        if "placing" in text:
            return "place"
        if "place firmly" in text:
            return "place firmly"
        if "place gently" in text:
            return "place gently"
        if "drop" in text:
            return "drop"
        if "hold" in text:
            return "hold"
        if "grasp" in text:
            return "grasp"
        if "lift" in text:
            return "lift"
        return ""

    def _fallback_interaction(annotation: str) -> str:
        text = (annotation or "").lower()
        if "stable" in text:
            return "stable"
        if "slip quickly" in text:
            return "fast slip"
        if "slip slowly" in text:
            return "slow slip"
        if "slip" in text:
            return "slow slip"

    def _fallback_weight(annotation: str) -> str:
        text = (annotation or "").lower()
        if "light" in text:
            return "light"
        if "heavy" in text:
            return "heavy"
        return ""

    def _fallback_placement_outcome(annotation: str) -> str:
        text = (annotation or "").lower()
        if "topple" in text:
            return "topple"
        if "remain" in text and "upright" in text:
            return "upright"
        return ""

    missing_action = df["action"] == ""
    if missing_action.any():
        df.loc[missing_action, "action"] = df.loc[missing_action, "annotation"].map(_fallback_action)

    missing_interaction = df["interaction"] == ""
    if missing_interaction.any():
        df.loc[missing_interaction, "interaction"] = df.loc[missing_interaction, "annotation"].map(_fallback_interaction)

    missing_weight = df["weight"] == ""
    if missing_weight.any():
        df.loc[missing_weight, "weight"] = df.loc[missing_weight, "annotation"].map(_fallback_weight)

    df["placement_outcome"] = df["placement_outcome"].replace(
        {
            "toppled": "topple",
            "topples after release": "topple",
            "remains upright": "upright",
        }
    )
    missing_placement_outcome = df["placement_outcome"] == ""
    if missing_placement_outcome.any():
        df.loc[missing_placement_outcome, "placement_outcome"] = df.loc[
            missing_placement_outcome, "annotation"
        ].map(_fallback_placement_outcome)

    # Placement outcome is defined only for windows labeled as placement.
    df.loc[df["action"] != "place", "placement_outcome"] = ""

    # For drop categories, interaction is intentionally omitted.
    df.loc[df["action"].isin(["accidental drop", "drop"]), "interaction"] = ""

    def _to_label(row: pd.Series) -> str:
        action = row["action"]
        if action == "lift then accidental drop":
            return "accidental drop"
        if action == "hold":
            return "hold"
        if action in {"place gently"}:
            return "place gently"
        if action in {"place firmly"}:
            return "place firmly"
        if "press" in action:
            return "press"
        if "grasp" in action:
            return "grasp"
        if action in {"lift", "place", "bump", "hold", "accidental drop", "drop"}:
            return action
        return "unknown"

    df["label"] = df.apply(_to_label, axis=1)
    return df


def parse_annotation_path(fp: str) -> tuple[str, str, str]:
    """
    Parse object/material/deformation from:
    .../csv/{obj}/{material}/{deformation}/{obj}_{material}_{deformation}_annotations.csv
    """
    p = Path(fp)
    deformation = p.parent.name
    material = p.parent.parent.name
    obj_name = p.parent.parent.parent.name
    return obj_name, material, deformation


def main() -> int:
    annotation_dir = os.path.join(DATA_DIR, "csv", "*", "Rigid", "*")
    pattern = os.path.join(annotation_dir, "*_annotations.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No annotation files found with pattern: {pattern}")
        return 1

    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        obj_name, material, deformation = parse_annotation_path(fp)
        if deformation == "bump":
            continue
        df["csv_path"] = f"{DATA_DIR}/csv/{obj_name}/{material}/{deformation}/{obj_name}_{material}_{deformation}.csv"
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out = add_labels(out)
    before_filter = len(out)
    out, removed_zero = filter_all_zero_force_windows(out)
    if removed_zero:
        print(f"Removed {removed_zero} all-zero force windows ({before_filter} -> {len(out)}).")
    out.to_csv(out_path, index=False)

    print(f"Wrote {out_path} with {len(out)} rows from {len(files)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
