import os

import numpy as np
import pandas as pd

from force_window_filter import filter_all_zero_force_windows

DATA_TYPE = os.environ.get("DATA_TYPE", "train_21072026")  # train / eval
DATA_DIR = f"/home/user/Genesis/data/{DATA_TYPE}"
N = int(os.environ.get("THIN_PCT", "15"))  # percentage of maximum allowed label ratio
P = N / 100.0
fix_seed = True
SEED = 42
FOR_VISION = False
BALANCE_PLACEMENT_OUTCOME = os.environ.get(
    "BALANCE_PLACEMENT_OUTCOME", "0" if "04272026" in DATA_TYPE else "1"
).lower() in {"1", "true", "yes"}

csv_path = f"{DATA_DIR}/{DATA_TYPE}.csv"
df = pd.read_csv(csv_path)
data_len = len(df)
print(f"original size: {data_len}")

df, removed_zero_windows = filter_all_zero_force_windows(df)
if removed_zero_windows:
    print(f"filtered out all-zero force windows: {removed_zero_windows}")
data_len = len(df)

if "label" not in df.columns:
    raise ValueError("CSV に 'label' 列が必要です")

labels_series = df["label"].fillna("<<MISSING>>").astype(str).str.strip().replace("", "<<EMPTY>>")
df["__normalized_label"] = labels_series

if "weight" in df.columns:
    weights = df["weight"].fillna("<<MISSING>>").astype(str).str.strip().str.lower().replace("", "<<EMPTY>>")
else:
    weights = pd.Series(["<<MISSING>>"] * len(df), index=df.index)

weights = weights.replace({"lightweight": "light", "heavyweight": "heavy"})
weights = np.where(np.isin(weights, ["light", "heavy"]), weights, "unknown")
df["__normalized_weight"] = weights

if "interaction" in df.columns:
    interactions = df["interaction"].fillna("").astype(str).str.strip().str.lower()
else:
    interactions = pd.Series([""] * len(df), index=df.index)
interaction_aliases = {
    "stable contact": "stable",
    "slowly slipping": "slow slip",
    "quickly slipping": "fast slip",
}
interactions = interactions.replace(interaction_aliases)
interactions = np.where(np.isin(interactions, ["stable", "slow slip", "fast slip"]), interactions, "unknown")
df["__normalized_interaction"] = interactions

if "placement_outcome" in df.columns:
    placement_outcomes = df["placement_outcome"].fillna("").astype(str).str.strip().str.lower()
else:
    placement_outcomes = pd.Series([""] * len(df), index=df.index)
placement_aliases = {
    "toppled": "topple",
    "topples after release": "topple",
    "remains upright": "upright",
}
placement_outcomes = placement_outcomes.replace(placement_aliases)
placement_outcomes = np.where(np.isin(placement_outcomes, ["topple", "upright"]), placement_outcomes, "unknown")
df["__normalized_placement_outcome"] = placement_outcomes

HELPER_COLUMNS = [
    "__normalized_label",
    "__normalized_weight",
    "__normalized_interaction",
    "__normalized_placement_outcome",
]


# FOR_VISION mode keeps the existing slip/stable balancing behavior.
if FOR_VISION:
    if "start" in df.columns:
        start_mask = (df["start"] != 0) & (df["start"] % 80 == 0)
        removed = (~start_mask).sum()
        print(f"filtered out (start %80 != 0): {removed}")
        df_v = df.loc[start_mask].copy()
    else:
        print("[WARN] 'start' 列が無いので start%80 フィルタはスキップします")
        df_v = df.copy()

    if df_v.empty:
        print("[WARN] Data is empty after filtering. Exiting.")
        raise SystemExit(0)

    labels_v = df_v["label"].fillna("<<MISSING>>").astype(str).str.strip().replace("", "<<EMPTY>>")
    df_v["__normalized_label"] = labels_v

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

    vision_counts = df_final["__normalized_label"].value_counts()
    print("[after vision] distribution (by count desc):")
    for lbl, cnt in vision_counts.sort_values(ascending=False).items():
        print(f"  {lbl}: {cnt} ({cnt / final_size * 100 if final_size else 0:.2f}%)")

    df_final.drop(columns=HELPER_COLUMNS, inplace=True)
    out_path = f"{csv_path.replace('.csv', '')}_vision.csv"
    df_final.to_csv(out_path, index=False)
    print(f"saved -> {out_path}")
    raise SystemExit(0)


rng = np.random.default_rng(SEED) if fix_seed else np.random.default_rng()


def apply_label_ratio_cap(df_in: pd.DataFrame, ratio_cap: float, rng_local: np.random.Generator) -> pd.DataFrame:
    label_counts = df_in["__normalized_label"].value_counts().sort_index()
    unique_labels = label_counts.index.tolist()
    L = len(unique_labels)
    if L == 0:
        return df_in.iloc[0:0].copy()

    p_min = 1.0 / L
    if ratio_cap < p_min:
        raise ValueError(
            f"N%={N}% is impossible for {L} unique labels (minimum required ratio={p_min*100:.2f}%). "
            f"Increase N to >= {p_min*100:.2f}."
        )

    label_to_indices: dict[str, list[int]] = {}
    for lbl in unique_labels:
        label_to_indices[lbl] = df_in.index[df_in["__normalized_label"] == lbl].tolist()

    while True:
        active_total = sum(len(v) for v in label_to_indices.values())
        if active_total == 0:
            break

        changed = False
        threshold = ratio_cap * active_total
        for lbl, inds in list(label_to_indices.items()):
            c = len(inds)
            if c > threshold:
                allowed = max(1, int(np.floor(threshold)))
                if c > allowed:
                    keep = rng_local.choice(inds, size=allowed, replace=False).tolist()
                    label_to_indices[lbl] = keep
                    changed = True
        if not changed:
            break

    keep_indices = set()
    for lst in label_to_indices.values():
        keep_indices.update(lst)
    return df_in.loc[sorted(keep_indices)].copy()


def weight_balance(df_in: pd.DataFrame, rng_local: np.random.Generator) -> pd.DataFrame:
    known = df_in[df_in["__normalized_weight"].isin(["light", "heavy"])].copy()
    if known.empty:
        print("[weight] no light/heavy rows found; skip weight balancing")
        return df_in

    light_df = known[known["__normalized_weight"] == "light"]
    heavy_df = known[known["__normalized_weight"] == "heavy"]
    if light_df.empty or heavy_df.empty:
        print("[weight] only one side exists; skip weight balancing")
        return df_in

    target_per_weight = min(len(light_df), len(heavy_df))
    print(f"[weight] target per class: {target_per_weight} (light={len(light_df)}, heavy={len(heavy_df)})")

    # Preserve label distribution inside each weight class as much as possible.
    label_probs = known["__normalized_label"].value_counts(normalize=True)

    selected_indices: list[int] = []
    for w in ["light", "heavy"]:
        wdf = known[known["__normalized_weight"] == w]
        counts = wdf["__normalized_label"].value_counts()

        desired = (label_probs * target_per_weight).fillna(0.0)
        base = desired.astype(int)
        base = np.minimum(base, counts.reindex(base.index).fillna(0).astype(int))

        picked = int(base.sum())
        need = target_per_weight - picked
        remainder = (desired - base).sort_values(ascending=False)

        if need > 0:
            for lbl in remainder.index:
                if need <= 0:
                    break
                available = int(counts.get(lbl, 0) - base.get(lbl, 0))
                if available <= 0:
                    continue
                add = min(available, need)
                base.loc[lbl] = int(base.get(lbl, 0)) + add
                need -= add

        if need > 0:
            extra_pool = wdf.index.tolist()
            take_extra = rng_local.choice(extra_pool, size=need, replace=False).tolist()
            selected_indices.extend(take_extra)

        for lbl, n_take in base.items():
            if n_take <= 0:
                continue
            pool = wdf[wdf["__normalized_label"] == lbl].index.tolist()
            if len(pool) <= n_take:
                selected_indices.extend(pool)
            else:
                selected_indices.extend(rng_local.choice(pool, size=n_take, replace=False).tolist())

    selected_indices = sorted(set(selected_indices))
    out = known.loc[selected_indices].copy()
    return out


def balance_attribute(
    df_in: pd.DataFrame,
    normalized_col: str,
    classes: list[str],
    display_name: str,
    rng_local: np.random.Generator,
    announce: bool = True,
) -> pd.DataFrame:
    """Balance known attribute classes while retaining rows where the attribute is not applicable."""
    counts = df_in[normalized_col].value_counts()
    present_classes = [value for value in classes if int(counts.get(value, 0)) > 0]
    if len(present_classes) < 2:
        if announce:
            print(f"[{display_name}] fewer than two known classes found; skip balancing")
        return df_in

    target_per_class = min(int(counts[value]) for value in present_classes)
    count_text = ", ".join(f"{value}={int(counts[value])}" for value in present_classes)
    if announce:
        print(f"[{display_name}] target per class: {target_per_class} ({count_text})")

    known_mask = df_in[normalized_col].isin(present_classes)
    selected_indices = df_in.index[~known_mask].tolist()
    for value in present_classes:
        pool = df_in.index[df_in[normalized_col] == value].tolist()
        if len(pool) <= target_per_class:
            selected_indices.extend(pool)
        else:
            selected_indices.extend(rng_local.choice(pool, size=target_per_class, replace=False).tolist())

    return df_in.loc[sorted(selected_indices)].copy()


def print_attribute_distribution(df_in: pd.DataFrame, normalized_col: str, display_name: str) -> None:
    counts = df_in[normalized_col].value_counts().sort_values(ascending=False)
    print(f"[{display_name}] distribution:")
    for value, count in counts.items():
        print(f"  {value}: {count}")


before_counts = df["__normalized_label"].value_counts().sort_values(ascending=False)
print("\n[before] label distribution:")
for lbl, cnt in before_counts.items():
    print(f"  {lbl}: {cnt} ({cnt / data_len * 100:.2f}%)")

before_w = df["__normalized_weight"].value_counts().sort_values(ascending=False)
print("[before] weight distribution:")
for w, cnt in before_w.items():
    print(f"  {w}: {cnt} ({cnt / data_len * 100:.2f}%)")
print_attribute_distribution(df, "__normalized_interaction", "before interaction")
if BALANCE_PLACEMENT_OUTCOME:
    print_attribute_distribution(df, "__normalized_placement_outcome", "before placement_outcome")

# 1) label ratio cap (existing behavior)
df_thin = apply_label_ratio_cap(df, P, rng)

# 2) balance weight light/heavy (new)
df_thin = weight_balance(df_thin, rng)

# 3) Satisfy the label cap and both attribute balances together. Balancing one
# attribute can disturb another, so repeat the downsampling-only operations until
# a complete pass removes no rows.
df_final = df_thin
MAX_BALANCE_ROUNDS = 20
for balance_round in range(1, MAX_BALANCE_ROUNDS + 1):
    size_before_round = len(df_final)
    df_final = apply_label_ratio_cap(df_final, P, rng)
    df_final = balance_attribute(
        df_final,
        "__normalized_interaction",
        ["stable", "slow slip", "fast slip"],
        "interaction",
        rng,
        announce=balance_round == 1,
    )
    if BALANCE_PLACEMENT_OUTCOME:
        df_final = balance_attribute(
            df_final,
            "__normalized_placement_outcome",
            ["topple", "upright"],
            "placement_outcome",
            rng,
            announce=balance_round == 1,
        )
    removed_this_round = size_before_round - len(df_final)
    print(f"[balance] round {balance_round}: removed {removed_this_round}, remaining {len(df_final)}")
    if removed_this_round == 0:
        break
else:
    print(f"[WARN] balancing did not converge after {MAX_BALANCE_ROUNDS} rounds")

final_size = len(df_final)
print("\n[after] label distribution (by count desc):")
final_counts = df_final["__normalized_label"].value_counts().sort_values(ascending=False)
for lbl, cnt in final_counts.items():
    print(f"  {lbl}: {cnt} ({cnt / final_size * 100 if final_size else 0:.2f}%)")

print("[after] weight distribution:")
final_w = df_final["__normalized_weight"].value_counts().sort_values(ascending=False)
for w, cnt in final_w.items():
    print(f"  {w}: {cnt} ({cnt / final_size * 100 if final_size else 0:.2f}%)")
print_attribute_distribution(df_final, "__normalized_interaction", "after interaction")
if BALANCE_PLACEMENT_OUTCOME:
    print_attribute_distribution(df_final, "__normalized_placement_outcome", "after placement_outcome")

if final_size > 0:
    worst_lbl = max(final_counts.items(), key=lambda x: x[1] / final_size)[0]
    worst_ratio = final_counts[worst_lbl] / final_size * 100
    print(f"\nWorst label ratio: {worst_lbl} = {worst_ratio:.2f}% (limit {N:.2f}%)")

print(f"dropped: {data_len - final_size}")
print(f"final size: {final_size} (was {data_len})")

df_final.drop(columns=HELPER_COLUMNS, inplace=True)
out_path = f"{csv_path.replace('.csv', '')}_thin_{N}pct.csv"
df_final.to_csv(out_path, index=False)
print(f"saved -> {out_path}")
