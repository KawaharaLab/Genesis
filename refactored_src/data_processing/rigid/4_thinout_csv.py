import pandas as pd
import numpy as np

DATA_TYPE = "eval"
DATA_DIR = f"/home/user/Genesis/data/{DATA_TYPE}"
N = 10
df = pd.read_csv(f"{DATA_DIR}/{DATA_TYPE}.csv")
# df = pd.read_csv(f"{DATA_DIR}/{DATA_TYPE}.csv").iloc[385:390]
data_len = df.shape[0]
print(data_len)
unique_csv_paths = df["csv_path"].unique()

data_cache = {
    path: pd.read_csv(DATA_DIR + "/csv/" + path)
    for path in unique_csv_paths
}

hold = []
nc = []
for i, row in df.iterrows():
    force_df = data_cache[row['csv_path']]
    if np.all(force_df['obj_left_finger'][row['start']:row['start']+80]) and np.all(force_df['obj_right_finger'][row['start']:row['start']+80]):
        if ("slip" in row['annotation']) or ("grasp" in row['annotation']):
            continue
        hold.append(i)
    elif not (np.any(force_df['obj_left_finger'][row['start']:row['start']+80]) or np.any(force_df['obj_right_finger'][row['start']:row['start']+80])):
        nc.append(i)
print("hold", len(hold))
print("no contact", len(nc))
# ---------------- Thin out to N% each for hold and no-contact ---------------- #
target = int(np.floor(data_len * (N / 100)))
print(f"target per class: {target} rows (N={N}% of total {data_len})")

rng = np.random.default_rng()

def compute_drop(indices: list[int], target_count: int) -> set[int]:
    if len(indices) <= target_count:
        return set()
    keep = set(rng.choice(indices, size=target_count, replace=False).tolist())
    return set(indices) - keep

drop_hold = compute_drop(hold, target)
drop_nc = compute_drop(nc, target)

to_drop = sorted(drop_hold.union(drop_nc))

print(f"drop hold: {len(drop_hold)} (from {len(hold)})")
print(f"drop nc:   {len(drop_nc)} (from {len(nc)})")
print(f"total drop: {len(to_drop)}")

df_thin = df.drop(index=to_drop)
print(f"final size: {len(df_thin)} (was {data_len})")
print("new hold", len(set(hold) - drop_hold))
print("new no contact", len(set(nc) - drop_nc))

out_path = f"{DATA_DIR}/{DATA_TYPE}_thin_{N}pct.csv"
df_thin.to_csv(out_path, index=False)
print(f"saved -> {out_path}")