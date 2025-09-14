import pandas as pd
import numpy as np

"""目的:
最終的な (間引き後の) データフレーム df_final において
  hold 比率 = N% かつ no-contact 比率 = N%
となるように hold / no-contact のサンプル数を調整する。

他カテゴリ (grasp など) はドロップせず残すものとする。

導出:
  O = その他カテゴリの件数 (固定)
  k = 最終的に保持する hold 件数 = 最終的に保持する no-contact 件数
  p = N/100
  k / (O + 2k) = p => k = p * O / (1 - 2p) (p < 0.5 で有効)

制約:
  k は各クラスの元件数以下。
  不可能な場合 (k > min(len(hold), len(nc))) は "最大限近い" 状態 (両方とも min(len(hold), len(nc))) を採用し、
  目標未達を警告表示する。
"""

DATA_TYPE = "train"
DATA_DIR = f"/home/user/Genesis/data/{DATA_TYPE}"
N = 15  # 目標: hold と no-contact を最終データサイズの N%
P = N / 100.0

if P >= 0.5:
    raise ValueError("N は 50 未満である必要があります (1 - 2p > 0 条件)。")

df = pd.read_csv(f"{DATA_DIR}/{DATA_TYPE}.csv")
data_len = df.shape[0]
print(f"original size: {data_len}")

unique_csv_paths = df["csv_path"].unique()
data_cache = {path: pd.read_csv(DATA_DIR + "/csv/" + path) for path in unique_csv_paths}

hold: list[int] = []
nc: list[int] = []
grasp: list[int] = []

for i, row in df.iterrows():
    force_df = data_cache[row['csv_path']]
    seg = slice(row['start'], row['start'] + 80)
    left_seg = force_df['obj_left_finger'][seg]
    right_seg = force_df['obj_right_finger'][seg]

    left_all = np.all(left_seg)
    right_all = np.all(right_seg)
    left_any = np.any(left_seg)
    right_any = np.any(right_seg)

    ann = row['annotation'] if 'annotation' in row else ''

    if left_all and right_all:
        # grasp / slip を含む場合は hold とみなさない
        if ("slip" in ann) or ("grasp" in ann):
            continue
        hold.append(i)
    elif not (left_any or right_any):
        nc.append(i)
    elif "grasp" in ann:
        grasp.append(i)

print(f"hold (raw)	{len(hold)}")
print(f"no contact (raw)	{len(nc)}")
print(f"grasp (raw)	{len(grasp)}")

O = data_len - len(hold) - len(nc)
print(f"others (O)	{O}")

if O < 0:
    raise RuntimeError("計算ミス: others が負です。")

# 目標保持数 k の理論値
k_theoretical = (P * O) / (1 - 2 * P) if O > 0 else min(len(hold), len(nc))
k = int(np.floor(k_theoretical))

# クラス元件数より大きい場合は調整
max_possible = min(len(hold), len(nc))
if k > max_possible:
    print("[WARN] 目標比率を満たすために必要な件数 k がクラスの元件数を超えています。")
    print(f"       k_theoretical={k_theoretical:.2f}, 調整後 k={max_possible}")
    k = max_possible

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

to_drop = sorted(drop_hold.union(drop_nc))

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

out_path = f"{DATA_DIR}/{DATA_TYPE}_thin_{N}pct.csv"
df_final.to_csv(out_path, index=False)
print(f"saved -> {out_path}")