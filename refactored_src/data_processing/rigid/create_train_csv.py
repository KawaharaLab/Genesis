import glob
import os
import sys

import pandas as pd
DATA_TYPE = "YCB_0824"

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
    out_path = os.path.join(base_dir, "train.csv")
    out.to_csv(out_path, index=False)

    print(f"Wrote {out_path} with {len(out)} rows from {len(files)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
