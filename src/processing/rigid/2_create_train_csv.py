import glob
import os
import sys
from pathlib import Path

import pandas as pd

MODE = "eval_04072026"  # "train" or "eval"
DATA_DIR = f"/home/user/Genesis/data/{MODE}"
out_path = f"{DATA_DIR}/{MODE}.csv"

def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "action" not in df.columns:
        df["action"] = None
    if "interaction" not in df.columns:
        df["interaction"] = None
    if "weight" not in df.columns:
        df["weight"] = None

    df["action"] = df["action"].fillna("").astype(str).str.strip()
    df["interaction"] = df["interaction"].fillna("").astype(str).str.strip()

    def _fallback_action(annotation: str) -> str:
        text = (annotation or "").lower()
        if "accidental drop" in text:
            return "accidental drop"
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
        return ""

    def _fallback_interaction(annotation: str) -> str:
        text = (annotation or "").lower()
        if "stable" in text:
            return "stable"
        if "slipping quickly" in text or "slips quickly" in text or "fast slip" in text:
            return "fast slip"
        if "slipping slowly" in text or "slips slowly" in text or "slow slip" in text:
            return "slow slip"
        if "slip" in text:
            return "slow slip"
        return ""

    missing_action = df["action"] == ""
    if missing_action.any():
        df.loc[missing_action, "action"] = df.loc[missing_action, "annotation"].map(_fallback_action)

    missing_interaction = df["interaction"] == ""
    if missing_interaction.any():
        df.loc[missing_interaction, "interaction"] = df.loc[missing_interaction, "annotation"].map(_fallback_interaction)

    # For drop categories, interaction is intentionally omitted.
    df.loc[df["action"].isin(["accidental drop", "drop"]), "interaction"] = ""

    def _to_label(row: pd.Series) -> str:
        action = row["action"]
        interaction = row["interaction"]
        if action == "hold":
            if interaction == "stable":
                return "stable"
            if interaction in {"slow slip", "fast slip"}:
                return "slip"
        if action in {"place gently"}:
            return "place gently"
        if action in {"place firmly"}:
            return "place firmly"
        if action in {"grasp", "hold", "accidental drop", "drop"}:
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
        df["csv_path"] = f"{DATA_DIR}/csv/{obj_name}/{material}/{deformation}/{obj_name}_{material}_{deformation}.csv"
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out = add_labels(out)
    out.to_csv(out_path, index=False)

    print(f"Wrote {out_path} with {len(out)} rows from {len(files)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
