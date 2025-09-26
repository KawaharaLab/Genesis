import glob
import os
import sys

import pandas as pd

DATA_DIR = "/home/user/Genesis/data/"
out_path = f"/home/user/Genesis/data/train_eval_mixed.csv"

# DATA_TYPE = "train_old"  # "train_old" or "eval_heavy"
# DATA_DIR = "/home/user/Genesis/data/"
# out_path = f"/home/user/Genesis/data/{DATA_TYPE}/{DATA_TYPE}.csv"


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
    
    # 1. Collect file paths for both data types
    base_dir_train = os.path.join(DATA_DIR, "processed", "train_old")
    annotation_dir_train = os.path.join(base_dir_train, "com")
    pattern_train = os.path.join(annotation_dir_train, "*.csv")
    train_old_files = sorted(glob.glob(pattern_train))

    base_dir_eval = os.path.join(DATA_DIR, "processed", "eval_heavy")
    annotation_dir_eval = os.path.join(base_dir_eval, "com")
    pattern_eval = os.path.join(annotation_dir_eval, "*.csv")
    eval_heavy_files = sorted(glob.glob(pattern_eval))

    print(f"Found {len(train_old_files)} train_old files and {len(eval_heavy_files)} eval_heavy files.")

    # 2. Interleave the file paths with a 1-in-11 ratio
    interleaved_files = []
    eval_index = 0
    for i, train_file in enumerate(train_old_files):
        interleaved_files.append(train_file)
        
        # Add an eval file after every 11th train file
        if (i + 1) % 11 == 0 and eval_index < len(eval_heavy_files):
            interleaved_files.append(eval_heavy_files[eval_index])
            eval_index += 1

    # Add any remaining eval files to the end
    while eval_index < len(eval_heavy_files):
        interleaved_files.append(eval_heavy_files[eval_index])
        eval_index += 1
    
    all_dfs = []

    # 3. Loop through the interleaved files and process each
    for fp in interleaved_files:
        source_type = 1 if "eval_heavy" in fp else 0
        
        df = pd.read_csv(fp)
        obj_name = os.path.basename(fp).replace("_Rigid_none_annotations.csv", "")
        df["csv_path"] = f"{obj_name}/Rigid/none/{obj_name}_Rigid_none.csv"
        df["source_type"] = source_type
        all_dfs.append(df)

    # 4. Concatenate all dataframes and save
    out = pd.concat(all_dfs, ignore_index=True)
    out = add_labels(out)
    out.to_csv(out_path, index=False)

    print(f"Wrote combined data to {out_path} with {len(out)} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
