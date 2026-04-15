import glob
import os
import sys

import pandas as pd

MODE = "eval_03062026"  # "train" or "eval"
DATA_DIR = f"/home/user/Genesis/data/{MODE}"
out_path = f"{DATA_DIR}/{MODE}.csv"

def add_labels(df: pd.DataFrame) -> pd.DataFrame:

    df['label'] = None
    df['action'] = None
    df['weight'] = None
    df['interaction'] = None
    for idx, row in df.iterrows():
        annotation = row["annotation"]
        #label
        if "empty" in annotation:
            df.at[idx, "label"] = "empty"
        elif "place" in annotation:
            df.at[idx, "label"] = "place"
        elif "grasp" in annotation:
            df.at[idx, "label"] = "grasp"
        elif "drop" in annotation:
            df.at[idx, "label"] = "drop"
        elif "slip" in annotation:
            df.at[idx, "label"] = "slip"
        elif "stable" in annotation:
            df.at[idx, 'label'] = "stable"
        else:
            df.at[idx, "label"] = "touch"
        #action
        if "grasp" in annotation:
            df.at[idx, "action"] = "grasp"
        elif "place" in annotation:
            df.at[idx, "action"] = "place"
        elif "shake" in annotation:
            df.at[idx, "action"] = "shake"
        elif "rotate" in annotation:
            df.at[idx, "action"] = "rotate"
        elif "lift" in annotation:
            df.at[idx, "action"] = "lift"
        elif "descend" in annotation:
            df.at[idx, "action"] = "descend"
        elif "hold" in annotation:
            df.at[idx, "action"] = "hold"
        
        #weight
        if "light" in annotation:
            df.at[idx, "weight"] = "light"
        elif "heavy" in annotation:
            df.at[idx, "weight"] = "heavy"
        
        #interaction
        if "stable" in annotation:
            df.at[idx, "interaction"] = "stable"
        elif "slip" in annotation:
            df.at[idx, "interaction"] = "slip"
        elif "drop" in annotation:
            df.at[idx, "interaction"] = "drop"
    return df


def main() -> int:
    files = []
    annotation_dir = os.path.join(DATA_DIR, "csv", "*", "Rigid", "none")
    pattern = os.path.join(annotation_dir, "*_annotations.csv")
    files.extend(sorted(glob.glob(pattern)))
    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        obj_name = os.path.basename(fp).replace("_Rigid_none_annotations.csv", "")
        df["csv_path"] = f"{DATA_DIR}/csv/{obj_name}/Rigid/none/{obj_name}_Rigid_none.csv"
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out = add_labels(out)
    out.to_csv(out_path, index=False)

    print(f"Wrote {out_path} with {len(out)} rows from {len(files)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
