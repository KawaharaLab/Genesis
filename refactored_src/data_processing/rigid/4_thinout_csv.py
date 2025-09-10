import pandas as pd
import numpy as np

FOR_VISION = True
DATA_TYPE = "eval"
DATA_DIR = f"/home/user/Genesis/data/{DATA_TYPE}"
N = 10  # "hold" and "no-contact" should be N% of the whole dataset
P = N / 100.0

if P >= 0.5:
    raise ValueError("N has to be smaller than 50 (1 - 2p > 0)")

df = pd.read_csv(f"{DATA_DIR}/{DATA_TYPE}.csv")
data_len = df.shape[0]
print(f"original size: {data_len}")

unique_csv_paths = df["csv_path"].unique()
data_cache = {path: pd.read_csv(DATA_DIR + "/csv/" + path) for path in unique_csv_paths}

hold: list[int] = []
nc: list[int] = []
grasp: list[int] = []
slip: list[int] = []
drop_data: list[int] = []
for i, row in df.iterrows():
    if FOR_VISION and row['start'] %80 != 0:
        drop_data.append(i)
        continue
    if "slip" in row['label']:
        slip.append(i) 
        continue
    force_df = data_cache[row['csv_path']]
    seg = slice(row['start'], row['start'] + 80)
    left_seg = force_df['obj_left_finger'][seg]
    right_seg = force_df['obj_right_finger'][seg]

    left_all = np.all(left_seg)
    right_all = np.all(right_seg)
    left_any = np.any(left_seg)
    right_any = np.any(right_seg)

    if left_all and right_all:
        hold.append(i)
    elif "hold" in row['label']:
        drop_data.append(i)
    elif not (left_any or right_any):
        nc.append(i)


print(f"slip (raw)	{len(slip)}")
print(f"hold (raw)	{len(hold)}")
print(f"no contact (raw)	{len(nc)}")
print(f"drop (raw)	{len(drop_data)}")

O = data_len - len(hold) - len(nc) - len(drop_data)
print(f"others (O)	{O}")


# The theorical number of target k to keep
k_theoretical = (P * O) / (1 - 2 * P) if O > 0 else min(len(hold), len(nc))
k = int(np.floor(k_theoretical))

# adjust if larger than class size
max_possible = min(len(hold), len(nc))
if k > max_possible:
    print("[WARN] The target k to achieve the goal ratio is larger than the class size.")
    print(f"       k_theoretical={k_theoretical:.2f}, adjusted k={max_possible}")
    k = max_possible
if FOR_VISION:
    k = len(slip)
print(f"target hold / nc keep count k = {k} (theoretical {k_theoretical:.2f})")

rng = np.random.default_rng()

def choose_keep(indices: list[int], keep_count: int) -> set[int]:
    if len(indices) <= keep_count:
        return set(indices)  # そのまま全部残す
    return set(rng.choice(indices, size=keep_count, replace=False).tolist())

keep_hold = choose_keep(hold, k)
keep_nc = choose_keep(nc, k)

drop_hold = set(hold) - keep_hold
drop_nc = set(nc) - keep_nc

to_drop = sorted(drop_hold.union(drop_nc).union(set(drop_data)))

df_final = df.drop(index=to_drop)
final_size = len(df_final)

final_hold = len(keep_hold)
final_nc = len(keep_nc)
prop_hold = final_hold / final_size if final_size else 0
prop_nc = final_nc / final_size if final_size else 0

print(f"drop hold: {len(drop_hold)} (from {len(hold)})")
print(f"drop nc:   {len(drop_nc)} (from {len(nc)})")
print(f"total drop: {len(to_drop)}")
print(f"final size: {final_size} (was {data_len})")
print(f"final hold kept: {final_hold} (was {len(hold)}) {N}%")
print(f"final nc kept:   {final_nc} (was {len(nc)}) {N}%")
if FOR_VISION:
    out_path = f"{DATA_DIR}/vision_{DATA_TYPE}.csv"
else:
    out_path = f"{DATA_DIR}/{DATA_TYPE}_thin_{N}pct.csv"
df_final.to_csv(out_path, index=False)
print(f"saved -> {out_path}")