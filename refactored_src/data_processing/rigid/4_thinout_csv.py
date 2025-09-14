import pandas as pd
import numpy as np

DEBUG = False
DATA_TYPE = "eval_heavy"
DATA_DIR = f"/home/user/Genesis/data/{DATA_TYPE}"
N = 20  # each label should be under N% of the final dataset
P = N / 100.0
fix_seed = True
SEED = 42
FOR_VISION = True

import os
if DEBUG:
    # csv_path = f"{DATA_DIR}/{DATA_TYPE}.csv"
    csv_path = "/home/user/Genesis/data/train_old/train_old_thin_20pct.csv"
    # csv_path = "/home/user/Genesis/data/eval_heavy/eval_heavy_thin_5pct_an.csv"
    if not os.path.exists(csv_path):
        print(f"[INFO] CSV not found: {csv_path}")
    else:
        df = pd.read_csv(csv_path)
        total = len(df)
        print(f"dataset: {csv_path}")
        print(f"total: {total}")

    # label column distribution
        if "label" in df.columns:
            labels = df["label"].fillna("<<MISSING>>").replace("", "<<EMPTY>>")
            vc = labels.value_counts()
            print("\nlabel distribution:")
            for lbl, cnt in vc.items():
                pct = cnt / total * 100 if total else 0
                print(f"  {lbl}: {cnt} ({pct:.2f}%)")
        else:
            print("\nlabel column が存在しません")

    # annotation column distribution (if exists)
        if "annotation" in df.columns:
            ann = df["annotation"].fillna("<<MISSING>>").replace("", "<<EMPTY>>")
            vc2 = ann.value_counts()
            print("\nannotation distribution:")
            for a, cnt in vc2.items():
                pct = cnt / total * 100 if total else 0
                print(f"  {a}: {cnt} ({pct:.2f}%)")

        # # vision 用の start %80 フィルタに該当する行の割合
        # if "start" in df.columns:
        #     mod0 = (df["start"] % 80 == 0).sum()
        #     print(f"\nstart % 80 == 0: {mod0} ({mod0 / total * 100:.2f}%)")
    exit(0)
#####################################################


csv_path = f"{DATA_DIR}/{DATA_TYPE}.csv"
df = pd.read_csv(csv_path)
data_len = len(df)
print(f"original size: {data_len}")

# When FOR_VISION, synchronize label with annotation keywords (slip/stable) before further processing
if FOR_VISION and "annotation" in df.columns and "label" in df.columns:
    ann_series = df["annotation"].astype(str)
    slip_mask_sync = ann_series.str.contains("slip", case=False, na=False)
    stable_mask_sync = ann_series.str.contains("stable", case=False, na=False)
    # Apply slip first so "slip" dominates if both appear (rare but deterministic)
    updated_slip = slip_mask_sync.sum()
    updated_stable = (~slip_mask_sync & stable_mask_sync).sum()
    if updated_slip or updated_stable:
        df.loc[slip_mask_sync, "label"] = "letting slip"
        df.loc[~slip_mask_sync & stable_mask_sync, "label"] = "keeping stable"
        print(f"[vision sync] set label=letting slip for {updated_slip} rows; label=keeping stable for {updated_stable} rows")

if "label" not in df.columns:
    raise ValueError("CSV に 'label' 列が必要です")

# Normalize label column (handle missing and empty strings)
labels_series = df["annotation"].fillna("<<MISSING>>").astype(str).str.strip().replace("", "<<EMPTY>>")
# labels_series = df["label"].fillna("<<MISSING>>").astype(str).str.strip().replace("", "<<EMPTY>>")
df["__normalized_label"] = labels_series  # working column

# ------------------------------------------------------------------
# FOR_VISION 特別処理: slip / stable を 1:1 で均衡させ、他ラベルは除外
# ------------------------------------------------------------------
if FOR_VISION:
    # 1) filter rows where start % 80 == 0
    if "start" in df.columns:
        start_mask = (df["start"] % 80 == 0)
        removed = (~start_mask).sum()
        print(f"filtered out (start %80 != 0): {removed}")
        df_v = df.loc[start_mask].copy()
    else:
        print("[WARN] 'start' 列が無いので start%80 フィルタはスキップします")
        df_v = df.copy()

    if df_v.empty:
        print("[WARN] Data is empty after filtering. Exiting.")
        exit(0)

    labels_v = df_v["annotation"].fillna("<<MISSING>>").astype(str).str.strip().replace("", "<<EMPTY>>")
    # labels_v = df_v["label"].fillna("<<MISSING>>").astype(str).str.strip().replace("", "<<EMPTY>>")
    df_v["__normalized_label"] = labels_v

    # 3) extract slip / stable
    slip_mask = labels_v.str.contains("slip", case=False, na=False)
    stable_mask = labels_v.str.contains("stable", case=False, na=False)

    slip_indices = df_v.index[slip_mask].tolist()
    stable_indices = df_v.index[stable_mask].tolist()

    print(f"slip count (raw after start filter): {len(slip_indices)}")
    print(f"stable count (raw after start filter): {len(stable_indices)}")

    if len(slip_indices) == 0 or len(stable_indices) == 0:
        print("[WARN] slip もしくは stable が見つかりません。他ラベルは出力しません。")
        keep_indices = set(slip_indices + stable_indices)
    else:
        target = min(len(slip_indices), len(stable_indices))
        rng_local = np.random.default_rng(SEED) if fix_seed else np.random.default_rng()
        if len(slip_indices) > target:
            slip_indices = rng_local.choice(slip_indices, size=target, replace=False).tolist()
        if len(stable_indices) > target:
            stable_indices = rng_local.choice(stable_indices, size=target, replace=False).tolist()
        keep_indices = set(slip_indices + stable_indices)
        print(f"balanced target per label: {target}")

    df_final = df_v.loc[sorted(keep_indices)].copy()
    final_size = len(df_final)
    print(f"final size (vision): {final_size}")

    # show distribution
    vision_counts = df_final["__normalized_label"].value_counts()
    print("[after vision] distribution (by count desc):")
    for lbl, cnt in vision_counts.sort_values(ascending=False).items():
        print(f"  {lbl}: {cnt} ({cnt / final_size * 100 if final_size else 0:.2f}%)")

    # 作業列削除
    df_final.drop(columns=["__normalized_label"], inplace=True)
    out_path = f"{csv_path.replace('.csv', '')}_vision_wide.csv"
    df_final.to_csv(out_path, index=False)
    print(f"saved -> {out_path}")
    exit(0)


label_counts = labels_series.value_counts().sort_index()
print("\n[before] label distribution:")
for lbl, cnt in label_counts.items():
    print(f"  {lbl}: {cnt} ({cnt / data_len * 100:.2f}%)")

unique_labels = label_counts.index.tolist()
L = len(unique_labels)
if L == 0:
    print("No labels found. Nothing to thin.")
    exit(0)

# Feasibility check: if P < 1/L it's impossible to make every label <= P
P_min = 1.0 / L
if P < P_min:
    raise ValueError(f"N%={N}% is impossible for {L} unique labels (minimum required ratio={P_min*100:.2f}%). Increase N to >= {P_min*100:.2f}.")

# random number generator
rng = np.random.default_rng(SEED) if fix_seed else np.random.default_rng()

# initialize indices to keep per label
label_to_indices: dict[str, list[int]] = {}
for lbl in unique_labels:
    label_to_indices[lbl] = df.index[df["__normalized_label"] == lbl].tolist()

# Iteratively downsample labels that exceed the threshold
iteration = 0
while True:
    iteration += 1
    # current active total
    active_total = sum(len(v) for v in label_to_indices.values())
    if active_total == 0:
        break

    changed = False
    threshold = P * active_total
    # downsample any label that exceeds the threshold
    for lbl, inds in list(label_to_indices.items()):
        c = len(inds)
        if c > threshold:
            allowed = int(np.floor(threshold))
            if allowed < 1:
                allowed = 1  # keep at least 1 item per label (change to 0 if desired)
            if c > allowed:
                # randomly keep only 'allowed' items
                keep = rng.choice(inds, size=allowed, replace=False).tolist()
                label_to_indices[lbl] = keep
                changed = True
    if not changed:
        break

# final set of indices to keep
keep_indices = set()
for lst in label_to_indices.values():
    keep_indices.update(lst)

df_final = df.loc[sorted(keep_indices)].copy()
final_size = len(df_final)

final_counts = df_final["__normalized_label"].value_counts()
print("\n[after] label distribution (by count desc):")
for lbl, cnt in final_counts.sort_values(ascending=False).items():
    pct = cnt / final_size * 100 if final_size else 0
    print(f"  {lbl}: {cnt} ({pct:.2f}%)")

worst_lbl = max(final_counts.items(), key=lambda x: x[1] / final_size if final_size else 0)[0]
worst_ratio = final_counts[worst_lbl] / final_size * 100 if final_size else 0
print(f"\nWorst label ratio: {worst_lbl} = {worst_ratio:.2f}% (limit {N:.2f}%)")

print(f"dropped: {data_len - final_size}")
print(f"final size: {final_size} (was {data_len})")

# 作業列削除
df_final.drop(columns=["__normalized_label"], inplace=True)

out_path = f"{csv_path.replace('.csv', '')}_an_thin_{N}pct.csv"
df_final.to_csv(out_path, index=False)
print(f"saved -> {out_path}")