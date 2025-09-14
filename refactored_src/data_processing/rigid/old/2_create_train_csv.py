import glob
import os
import sys

import pandas as pd

DATA_TYPE = "train"
out_path = f"/home/user/Genesis/data/{DATA_TYPE}/{DATA_TYPE}.csv"

def add_labels(df: pd.DataFrame) -> pd.DataFrame:

    df['label'] = None

    for idx, row in df.iterrows():
        annotation = row["annotation"]
        if "empty" in annotation:
            df.at[idx, "label"] = "empty"
        elif "drop" in annotation:
            df.at[idx, "label"] = "drop"
        elif "slip" in annotation:
            df.at[idx, "label"] = "slip"
        elif "place" in annotation:
            df.at[idx, "label"] = "place"
        elif "grasp" in annotation:
            df.at[idx, "label"] = "grasp"
        elif "stable" in annotation:
            df.at[idx, 'label'] = "stable"
        else:
            df.at[idx, "label"] = "touch"
    return df


def main() -> int:
    base_dir = os.path.join("data", "processed", DATA_TYPE)
    annotation_dir = os.path.join(base_dir, "simple_annotations")
    pattern = os.path.join(annotation_dir, "*.csv")
    files = sorted(glob.glob(pattern))

    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        obj_name = os.path.basename(fp).replace("_Rigid_none_annotations.csv", "")
        df["csv_path"] = f"{obj_name}/Rigid/none/{obj_name}_Rigid_none.csv"
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out = add_labels(out)
    out.to_csv(out_path, index=False)

    print(f"Wrote {out_path} with {len(out)} rows from {len(files)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
