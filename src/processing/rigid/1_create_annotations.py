import csv
import json
import os
import time
from multiprocessing import Process
from pathlib import Path

import numpy as np
import pandas as pd

BASE_PATH = Path(__file__).resolve().parent.parent.parent.parent
SEQUENCE_LENGTH = 80
DATASET = "eval_04072026"
WINDOW_STRIDE = 10
# WINDOW_STRIDE = 40 if "eval" in DATASET else 10
MAX_FORCE_ABS = 100.0
SLOW_SLIP_THRESHOLD = 0.002
FAST_SLIP_THRESHOLD = 0.004


def detect_bugs(force_df: pd.DataFrame, start: int, length: int = SEQUENCE_LENGTH) -> bool:
    """
    Exclude segments containing extreme contact-force readings.
    """
    force_cols = ["left_fx", "left_fy", "left_fz", "right_fx", "right_fy", "right_fz"]
    end = start + length
    seg = force_df.iloc[start:end]
    if seg.empty:
        return True
    return bool(np.any(np.abs(seg[force_cols].to_numpy()) >= MAX_FORCE_ABS))


def split_for_model(force_df: pd.DataFrame) -> list[dict]:
    """
    Generate fixed-length windows for annotation.
    """
    start = 0
    windows = []
    while start + SEQUENCE_LENGTH <= len(force_df):
        if not detect_bugs(force_df, start):
            windows.append({"start": start})
        start += WINDOW_STRIDE
    return windows


def _load_metadata(csv_path: str, obj_name: str, material: str, deformation: str) -> dict:
    metadata_path = os.path.join(csv_path, f"{obj_name}_{material}_metadata_{deformation}.json")
    if not os.path.exists(metadata_path):
        return {}
    with open(metadata_path, encoding="utf-8") as f:
        return json.load(f)


def _first_step(steps_df: pd.DataFrame, candidates: set[str]) -> int | None:
    rows = steps_df[steps_df["action"].astype(str).isin(candidates)]
    if rows.empty:
        return None
    return int(rows["step"].min())


def _first_plane_contact_index(force_df: pd.DataFrame) -> int | None:
    if "obj_plane" not in force_df.columns or force_df.empty:
        return None
    contact = force_df["obj_plane"].to_numpy(dtype=bool)
    if not np.any(contact):
        return None
    return int(np.argmax(contact))


def _first_release_index(force_df: pd.DataFrame, start_step: int | None = None) -> int | None:
    if force_df.empty:
        return None
    if "obj_left_finger" not in force_df.columns or "obj_right_finger" not in force_df.columns:
        return None

    left = force_df["obj_left_finger"].to_numpy(dtype=bool)
    right = force_df["obj_right_finger"].to_numpy(dtype=bool)
    contact_either = np.logical_or(left, right)

    begin = max(1, int(start_step) if start_step is not None else 1)
    begin = min(begin, len(contact_either) - 1)

    for i in range(begin, len(contact_either)):
        if contact_either[i - 1] and not contact_either[i]:
            return i
    return None


def _first_grasp_contact_index(force_df: pd.DataFrame, start_step: int | None = None) -> int | None:
    if force_df.empty:
        return None
    if "obj_left_finger" not in force_df.columns or "obj_right_finger" not in force_df.columns:
        return None

    left = force_df["obj_left_finger"].to_numpy(dtype=bool)
    right = force_df["obj_right_finger"].to_numpy(dtype=bool)
    contact_either = np.logical_or(left, right)

    begin = max(0, int(start_step) if start_step is not None else 0)
    begin = min(begin, len(contact_either) - 1)
    if contact_either[begin]:
        return begin

    for i in range(begin + 1, len(contact_either)):
        if (not contact_either[i - 1]) and contact_either[i]:
            return i
    return None


def _infer_place_mode(deformation: str, metadata: dict, steps_df: pd.DataFrame) -> str:
    place_mode = str(metadata.get("place_mode") or "").strip()
    if place_mode in {"drop", "gentle", "firm"}:
        return place_mode

    if deformation.startswith("early_"):
        tail = deformation.replace("early_", "", 1)
        if tail in {"drop", "gentle", "firm"}:
            return tail

    actions = set(steps_df["action"].astype(str).tolist())
    if "drop" in actions:
        return "drop"
    if "gentle_place" in actions:
        return "gentle"
    if "press_place" in actions:
        return "firm"
    return "gentle"


def _is_early_drop_case(deformation: str, metadata: dict, steps_df: pd.DataFrame) -> bool:
    if deformation.startswith("early_"):
        return True
    if bool(metadata.get("terminated_early_drop", False)):
        return True
    return bool(steps_df["action"].astype(str).str.contains("early_drop").any())


def _hold_condition(force_segment: pd.DataFrame) -> bool:
    left = force_segment["obj_left_finger"].to_numpy(dtype=bool)
    right = force_segment["obj_right_finger"].to_numpy(dtype=bool)
    plane = force_segment["obj_plane"].to_numpy(dtype=bool)
    contact_either = np.logical_or(left, right)
    return bool(np.all(contact_either) and not np.any(plane))


def _estimate_interaction(force_segment: pd.DataFrame) -> str:
    com_pos = force_segment[["obj_COM_x", "obj_COM_y", "obj_COM_z"]].to_numpy()
    right_finger_pos = force_segment[["right_finger_x", "right_finger_y", "right_finger_z"]].to_numpy()
    left_finger_pos = force_segment[["left_finger_x", "left_finger_y", "left_finger_z"]].to_numpy()
    grasp_pos = 0.5 * (right_finger_pos + left_finger_pos)

    left = force_segment["obj_left_finger"].to_numpy(dtype=bool)
    right = force_segment["obj_right_finger"].to_numpy(dtype=bool)
    contact_either = np.logical_or(left, right)
    if np.any(contact_either):
        grasp_pos = grasp_pos[contact_either]
        com_pos = com_pos[contact_either]

    if len(grasp_pos) == 0:
        return "stable"

    distances = np.linalg.norm(grasp_pos - com_pos, axis=1)
    slip_magnitude = float(np.max(distances) - np.min(distances))
    if slip_magnitude > FAST_SLIP_THRESHOLD:
        return "fast slip"
    if slip_magnitude > SLOW_SLIP_THRESHOLD:
        return "slow slip"
    return "stable"


def _determine_action(
    start: int,
    force_segment: pd.DataFrame,
    hold_start_step: int | None,
    place_start_step: int | None,
    first_plane_contact_idx: int | None,
    first_grasp_contact_idx: int | None,
    first_release_idx: int | None,
    early_drop_case: bool,
    place_mode: str,
) -> str:
    seg_end = start + len(force_segment) - 1
    includes_grasp_contact = first_grasp_contact_idx is not None and (start <= first_grasp_contact_idx <= seg_end)
    includes_release = first_release_idx is not None and (start <= first_release_idx <= seg_end)

    # Placement phase: explicit terminal labels.
    if place_start_step is not None and start >= place_start_step:
        if early_drop_case:
            return "accidental drop" if includes_release else "hold"
        if place_mode == "drop":
            return "drop" if includes_release else "hold"
        # For gentle/firm place, only label as "place" after object-plane contact appears.
        if first_plane_contact_idx is None or seg_end < first_plane_contact_idx:
            return "hold"
        if place_mode == "gentle":
            return "place gently"
        return "place firmly"

    # Pre-placement phase.
    if includes_grasp_contact:
        return "grasp"
    if _hold_condition(force_segment):
        return "hold"
    if early_drop_case:
        return "accidental drop" if includes_release else "hold"
    return "hold"


def _weight_descriptor(force_segment: pd.DataFrame) -> str:
    if "obj_mass" not in force_segment.columns or force_segment.empty:
        return "an object"
    mass = float(force_segment["obj_mass"].iloc[0])
    if mass >= 0.2:
        return "a heavy object"
    return "a light object"


def _build_annotation(action: str, interaction: str | None, force_segment: pd.DataFrame) -> str:
    obj_phrase = _weight_descriptor(force_segment)
    slow_slip = interaction == "slow slip"
    fast_slip = interaction == "fast slip"

    if action == "grasp":
        if fast_slip:
            return f"Grasping {obj_phrase}, letting it slip quickly."
        if slow_slip:
            return f"Grasping {obj_phrase}, letting it slip slowly."
        return f"Grasping {obj_phrase} with stable contact."

    if action == "hold":
        if fast_slip:
            return f"Holding {obj_phrase}, letting it slip quickly."
        if slow_slip:
            return f"Holding {obj_phrase}, letting it slip slowly."
        return f"Holding {obj_phrase} steadily."

    if action == "accidental drop":
        return f"Accidentally dropped {obj_phrase}."

    if action == "drop":
        return f"Released {obj_phrase}, dropping it."

    if action == "place gently":
        if fast_slip:
            return f"Placing {obj_phrase} gently, letting it slip quickly."
        if slow_slip:
            return f"Placing {obj_phrase} gently, letting it slip slowly."
        return f"Placing {obj_phrase} gently with stable contact."

    if action == "place firmly":
        if fast_slip:
            return f"Placing {obj_phrase} firmly, letting it slip quickly."
        if slow_slip:
            return f"Placing {obj_phrase} firmly, letting it slip slowly."
        return f"Placing {obj_phrase} firmly with stable contact."

    # Fallback for unexpected labels.
    if interaction:
        return f"Performs '{action}' with {interaction} contact."
    return f"Performs '{action}'."


def list_annotation_tasks(all_objects: list[str], material: str = "Rigid") -> list[tuple[str, str]]:
    tasks = []
    for obj_name in all_objects:
        if obj_name == ".DS_Store":
            continue
        object_root = os.path.join(BASE_PATH, "data", DATASET, "csv", obj_name, material)
        if not os.path.isdir(object_root):
            continue
        for deformation in sorted(os.listdir(object_root)):
            run_dir = os.path.join(object_root, deformation)
            if not os.path.isdir(run_dir):
                continue
            steps_csv = os.path.join(run_dir, f"{obj_name}_{material}_steps_{deformation}.csv")
            force_csv = os.path.join(run_dir, f"{obj_name}_{material}_{deformation}.csv")
            if not (os.path.exists(steps_csv) and os.path.exists(force_csv)):
                continue
            with open(force_csv, newline="") as f:
                reader = csv.reader(f)
                _ = next(reader, None)
                first_data_row = next(reader, None)
            if first_data_row is None:
                continue
            tasks.append((obj_name, deformation))
    return tasks


def main(obj_name: str, csv_path: str, deformation: str, material: str = "Rigid"):
    annotations_df = pd.DataFrame(columns=["action", "interaction", "start", "annotation"])

    steps_csv = os.path.join(csv_path, f"{obj_name}_{material}_steps_{deformation}.csv")
    force_csv = os.path.join(csv_path, f"{obj_name}_{material}_{deformation}.csv")
    if not os.path.exists(steps_csv) or not os.path.exists(force_csv):
        print(f"Skipping {obj_name}/{deformation}: missing steps or force CSV.")
        return

    steps_df = pd.read_csv(steps_csv)
    force_df = pd.read_csv(force_csv)
    metadata = _load_metadata(csv_path, obj_name, material, deformation)

    windows = split_for_model(force_df)
    hold_start_step = _first_step(steps_df, {"hold"})
    place_start_step = _first_step(steps_df, {"place", "drop", "gentle_place", "press_place"})
    first_plane_contact_idx = _first_plane_contact_index(force_df)
    first_grasp_contact_idx = _first_grasp_contact_index(force_df)
    release_search_start = hold_start_step if hold_start_step is not None else place_start_step
    first_release_idx = _first_release_index(force_df, release_search_start)
    early_drop_case = _is_early_drop_case(deformation, metadata, steps_df)
    place_mode = _infer_place_mode(deformation, metadata, steps_df)

    for row in windows:
        start = int(row["start"])
        if start + SEQUENCE_LENGTH > len(force_df):
            continue

        force_segment = force_df.iloc[start : start + SEQUENCE_LENGTH].reset_index(drop=True)
        if force_segment.isnull().values.any():
            print(f"Skipping {obj_name}/{deformation} segment @ {start}: NaN values.")
            continue
        if detect_bugs(force_df, start):
            continue

        action = _determine_action(
            start=start,
            force_segment=force_segment,
            hold_start_step=hold_start_step,
            place_start_step=place_start_step,
            first_plane_contact_idx=first_plane_contact_idx,
            first_grasp_contact_idx=first_grasp_contact_idx,
            first_release_idx=first_release_idx,
            early_drop_case=early_drop_case,
            place_mode=place_mode,
        )
        interaction = None if action in {"accidental drop", "drop"} else _estimate_interaction(force_segment)
        annotation = _build_annotation(action, interaction, force_segment)

        annotations_df.loc[len(annotations_df)] = {
            "action": action,
            "interaction": interaction,
            "start": start,
            "annotation": annotation,
        }

    output_csv_path = os.path.join(
        BASE_PATH,
        "data",
        DATASET,
        "csv",
        obj_name,
        material,
        deformation,
        f"{obj_name}_{material}_{deformation}_annotations.csv",
    )
    annotations_df.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    folder_path = os.path.join(BASE_PATH, "data", DATASET, "csv")
    if not os.path.isdir(folder_path):
        raise SystemExit(f"Dataset folder not found: {folder_path}")

    all_objects = os.listdir(folder_path)
    selected_tasks = list_annotation_tasks(all_objects)

    material = "Rigid"
    processes = []
    for obj_name, deformation in selected_tasks:
        picked_up_path = os.path.join(folder_path, obj_name, material, deformation)
        print(f"Processing {obj_name} / {deformation} ...")

        while len(processes) >= 8:
            processes = [p for p in processes if p.is_alive()]
            time.sleep(0.1)

        p = Process(target=main, args=(obj_name, picked_up_path, deformation, material))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All processes completed.")
