import csv
import json
import os
import time
from multiprocessing import Process
from pathlib import Path

import numpy as np
import pandas as pd

from force_window_filter import SEQUENCE_LENGTH, force_window_is_all_zero

BASE_PATH = Path(__file__).resolve().parent.parent.parent.parent
DATASET = os.environ.get("DATASET", "train_21072026")
WINDOW_STRIDE = 10
# WINDOW_STRIDE = 40 if "eval" in DATASET else 10
MAX_FORCE_ABS = 100.0
CONTACT_FORCE_COLS = ["left_fx", "left_fy", "left_fz", "right_fx", "right_fy", "right_fz"]
CONTACT_CONFIRM_STEPS = int(os.environ.get("CONTACT_CONFIRM_STEPS", "5"))
MAX_COM_STEP_M = float(os.environ.get("MAX_COM_STEP_M", "0.05"))
MAX_COM_XY_RADIUS_M = float(os.environ.get("MAX_COM_XY_RADIUS_M", "2.0"))
MIN_COM_Z_M = float(os.environ.get("MIN_COM_Z_M", "-0.1"))
MAX_COM_Z_M = float(os.environ.get("MAX_COM_Z_M", "2.0"))
LEGACY_TILE_Z_TOLERANCE_M = float(os.environ.get("LEGACY_TILE_Z_TOLERANCE_M", "0.008"))
INCLUDE_PLACEMENT_OUTCOME = os.environ.get(
    "INCLUDE_PLACEMENT_OUTCOME", "0" if "04272026" in DATASET else "1"
).lower() in {"1", "true", "yes"}
SLOW_SLIP_THRESHOLD = 0.002
FAST_SLIP_THRESHOLD = 0.004
PRESS_FORCE_MIN_N = float(os.environ.get("PRESS_FORCE_MIN_N", "1.0"))
PRESS_DESCENT_EPS_M = float(os.environ.get("PRESS_DESCENT_EPS_M", "1e-5"))
PLACE_ACTION = "place"


def detect_bugs(force_df: pd.DataFrame, start: int, length: int = SEQUENCE_LENGTH) -> bool:
    """
    Exclude segments containing extreme contact-force readings.
    """
    end = start + length
    seg = force_df.iloc[start:end]
    if seg.empty:
        return True
    return bool(np.any(np.abs(seg[CONTACT_FORCE_COLS].to_numpy()) >= MAX_FORCE_ABS))


def _has_bimanual_contact(force_df: pd.DataFrame, start: int, length: int = SEQUENCE_LENGTH) -> bool:
    """
    Keep a window only when both fingers touch the object at least once in the window.
    """
    end = start + length
    seg = force_df.iloc[start:end]
    if seg.empty:
        return False
    if "obj_left_finger" not in seg.columns or "obj_right_finger" not in seg.columns:
        return False

    left = seg["obj_left_finger"].to_numpy(dtype=bool)
    right = seg["obj_right_finger"].to_numpy(dtype=bool)
    return bool(np.any(left) and np.any(right))


def detect_rollout_bug(force_df: pd.DataFrame) -> str | None:
    """Return a reason when the complete rollout is physically invalid."""
    available_force_cols = [column for column in CONTACT_FORCE_COLS if column in force_df.columns]
    if available_force_cols:
        forces = force_df[available_force_cols].to_numpy(dtype=float)
        if np.any(np.abs(forces) >= MAX_FORCE_ABS):
            return f"contact-force component >= {MAX_FORCE_ABS:g} N"

    required = ["obj_COM_x", "obj_COM_y", "obj_COM_z"]
    if any(column not in force_df.columns for column in required):
        return "missing COM columns"
    com = force_df[required].to_numpy(dtype=float)
    if not np.isfinite(com).all():
        return "non-finite COM"
    if len(com) > 1 and np.any(np.linalg.norm(np.diff(com, axis=0), axis=1) > MAX_COM_STEP_M):
        return f"COM step displacement > {MAX_COM_STEP_M:g} m"
    if np.any(np.linalg.norm(com[:, :2], axis=1) > MAX_COM_XY_RADIUS_M):
        return f"COM XY radius > {MAX_COM_XY_RADIUS_M:g} m"
    if np.any((com[:, 2] < MIN_COM_Z_M) | (com[:, 2] > MAX_COM_Z_M)):
        return f"COM z outside [{MIN_COM_Z_M:g}, {MAX_COM_Z_M:g}] m"
    return None


def split_for_model(force_df: pd.DataFrame) -> list[dict]:
    """
    Generate fixed-length windows for annotation.
    """
    if detect_rollout_bug(force_df) is not None:
        return []
    start = 0
    windows = []
    while start + SEQUENCE_LENGTH <= len(force_df):
        if (
            not detect_bugs(force_df, start)
            and not force_window_is_all_zero(force_df, start)
            and _has_bimanual_contact(force_df, start)
        ):
            windows.append({"start": start})
        start += WINDOW_STRIDE
    return windows


def _append_terminal_window(
    force_df: pd.DataFrame, windows: list[dict], event_idx: int | None
) -> None:
    """Include an episode-ending event that falls between regular stride starts."""
    if len(force_df) < SEQUENCE_LENGTH or event_idx is None:
        return
    if any(
        int(window["start"]) <= event_idx < int(window["start"]) + SEQUENCE_LENGTH
        for window in windows
    ):
        return
    start = len(force_df) - SEQUENCE_LENGTH
    if any(int(window["start"]) == start for window in windows):
        return
    if (
        not detect_bugs(force_df, start)
        and not force_window_is_all_zero(force_df, start)
        and _has_bimanual_contact(force_df, start)
    ):
        windows.append({"start": start})
        windows.sort(key=lambda window: int(window["start"]))


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


def _first_sustained_state(
    series: np.ndarray,
    value: bool,
    start_step: int | None,
    confirm_steps: int = CONTACT_CONFIRM_STEPS,
    require_opposite_before: bool = False,
) -> int | None:
    """Find the first index starting a confirmed run of the requested state."""
    series = np.asarray(series, dtype=bool)
    if series.size < confirm_steps:
        return None
    begin = max(0, int(start_step) if start_step is not None else 0)
    begin = min(begin, len(series) - confirm_steps)
    opposite_seen = bool(np.any(series[:begin] != value))
    for index in range(begin, len(series) - confirm_steps + 1):
        if np.all(series[index : index + confirm_steps] == value):
            if not require_opposite_before or opposite_seen:
                return index
        if series[index] != value:
            opposite_seen = True
    return None


def _legacy_target_contact_series(force_df: pd.DataFrame, metadata: dict) -> np.ndarray:
    """Estimate target-tile contact for datasets whose recorded tile-contact column is broken."""
    required = {"obj_min_x", "obj_min_y", "obj_min_z", "obj_max_x", "obj_max_y"}
    target = metadata.get("target_xy") or {}
    if not required.issubset(force_df.columns) or not {"x", "y"}.issubset(target):
        return np.zeros(len(force_df), dtype=bool)

    tile = metadata.get("target_tile") or {}
    size_xyz = tile.get("size_xyz", (0.20, 0.20, 0.004))
    half_x = 0.5 * float(size_xyz[0])
    half_y = 0.5 * float(size_xyz[1])
    surface_z = float(tile.get("surface_center_z", 0.004))
    target_x, target_y = float(target["x"]), float(target["y"])

    overlaps_xy = (
        (force_df["obj_max_x"].to_numpy(dtype=float) >= target_x - half_x)
        & (force_df["obj_min_x"].to_numpy(dtype=float) <= target_x + half_x)
        & (force_df["obj_max_y"].to_numpy(dtype=float) >= target_y - half_y)
        & (force_df["obj_min_y"].to_numpy(dtype=float) <= target_y + half_y)
    )
    bottom_z = force_df["obj_min_z"].to_numpy(dtype=float)
    near_surface = np.abs(bottom_z - surface_z) <= LEGACY_TILE_Z_TOLERANCE_M
    return overlaps_xy & near_surface


def _target_contact_series(force_df: pd.DataFrame, metadata: dict) -> np.ndarray:
    recorded = (
        force_df["obj_target_tile"].to_numpy(dtype=bool)
        if "obj_target_tile" in force_df.columns
        else np.zeros(len(force_df), dtype=bool)
    )
    if int(metadata.get("contact_logging_version", 0)) >= 2:
        return recorded
    return recorded | _legacy_target_contact_series(force_df, metadata)


def _support_contact_series(force_df: pd.DataFrame, metadata: dict | None = None) -> np.ndarray:
    if force_df.empty:
        return np.array([], dtype=bool)
    tile = _target_contact_series(force_df, metadata or {})
    plane = force_df["obj_plane"].to_numpy(dtype=bool) if "obj_plane" in force_df.columns else None
    if tile is None and plane is None:
        return np.array([], dtype=bool)
    if tile is None:
        return plane
    if plane is None:
        return tile
    return np.logical_or(tile, plane)


def _first_support_contact_after(
    force_df: pd.DataFrame, metadata: dict, start_step: int | None
) -> int | None:
    return _first_sustained_state(
        _support_contact_series(force_df, metadata),
        True,
        start_step,
        require_opposite_before=True,
    )


def _first_support_detach_after(
    force_df: pd.DataFrame, metadata: dict, start_step: int | None
) -> int | None:
    return _first_sustained_state(
        _support_contact_series(force_df, metadata),
        False,
        start_step,
        require_opposite_before=True,
    )


def _first_obstacle_contact_after(force_df: pd.DataFrame, start_step: int | None) -> int | None:
    if "obj_obstacle" not in force_df.columns:
        return None
    obstacle = force_df["obj_obstacle"].to_numpy(dtype=bool)
    return _first_sustained_state(obstacle, True, start_step)


def _first_release_index(force_df: pd.DataFrame, start_step: int | None = None) -> int | None:
    if force_df.empty:
        return None
    if "obj_left_finger" not in force_df.columns or "obj_right_finger" not in force_df.columns:
        return None

    left = force_df["obj_left_finger"].to_numpy(dtype=bool)
    right = force_df["obj_right_finger"].to_numpy(dtype=bool)
    contact_either = np.logical_or(left, right)

    return _first_sustained_state(
        contact_either,
        False,
        start_step,
        require_opposite_before=True,
    )


def _confirmed_mask(series: np.ndarray, confirm_steps: int = CONTACT_CONFIRM_STEPS) -> np.ndarray:
    """Keep complete runs that persist for at least ``confirm_steps``."""
    series = np.asarray(series, dtype=bool)
    confirmed = np.zeros(series.size, dtype=bool)
    run_start = None
    for index, value in enumerate(series):
        if value and run_start is None:
            run_start = index
        if run_start is not None and (not value or index == len(series) - 1):
            run_end = index if value and index == len(series) - 1 else index - 1
            if run_end - run_start + 1 >= confirm_steps:
                confirmed[run_start : run_end + 1] = True
            run_start = None
    return confirmed


def _pressing_series(
    force_df: pd.DataFrame,
    metadata: dict,
    place_idx: int | None,
    release_command_step: int | None,
) -> np.ndarray:
    """Detect supported, finger-driven downward pressing after the place event."""
    if force_df.empty or place_idx is None:
        return np.zeros(len(force_df), dtype=bool)

    support = _support_contact_series(force_df, metadata)
    finger = np.logical_or(
        force_df["obj_left_finger"].to_numpy(dtype=bool),
        force_df["obj_right_finger"].to_numpy(dtype=bool),
    )
    eef_z = force_df["eef_z"].to_numpy(dtype=float)
    descending = np.r_[False, np.diff(eef_z) < -PRESS_DESCENT_EPS_M]
    vertical_force = (
        np.abs(force_df["left_fz"].to_numpy(dtype=float))
        + np.abs(force_df["right_fz"].to_numpy(dtype=float))
    )
    pressing = support & finger & (descending | (vertical_force >= PRESS_FORCE_MIN_N))
    pressing[: place_idx + 1] = False
    if release_command_step is not None:
        pressing[max(0, int(release_command_step)) :] = False
    return _confirmed_mask(pressing)


def _first_grasp_contact_index(force_df: pd.DataFrame, start_step: int | None = None) -> int | None:
    if force_df.empty:
        return None
    if "obj_left_finger" not in force_df.columns or "obj_right_finger" not in force_df.columns:
        return None

    left = force_df["obj_left_finger"].to_numpy(dtype=bool)
    right = force_df["obj_right_finger"].to_numpy(dtype=bool)
    bimanual_contact = np.logical_and(left, right)

    begin = max(0, int(start_step) if start_step is not None else 0)
    begin = min(begin, len(bimanual_contact) - 1)
    if bimanual_contact[begin]:
        return begin

    for i in range(begin + 1, len(bimanual_contact)):
        if (not bimanual_contact[i - 1]) and bimanual_contact[i]:
            return i
    return None


def _infer_place_mode(deformation: str, metadata: dict, steps_df: pd.DataFrame) -> str:
    place_mode = str(metadata.get("place_mode") or "").strip()
    if place_mode.startswith("inclined_"):
        return "inclined"
    if place_mode.startswith("step_"):
        return "step"
    if place_mode in {"drop", "drop_simple", "gentle", "gentle_simple", "firm", "firm_simple", "bump", "inclined"}:
        return place_mode

    if deformation.startswith("early_"):
        tail = deformation.replace("early_", "", 1)
        if tail.startswith("inclined_"):
            return "inclined"
        if tail.startswith("step_"):
            return "step"
        if tail in {"drop", "gentle", "firm", "bump", "inclined"}:
            return tail

    actions = set(steps_df["action"].astype(str).tolist())
    if "drop" in actions:
        return "drop"
    if "bump_push" in actions:
        return "bump"
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


def _estimate_interaction(force_segment: pd.DataFrame) -> str:
    com_pos = force_segment[["obj_COM_x", "obj_COM_y", "obj_COM_z"]].to_numpy()
    right_finger_pos = force_segment[["right_finger_x", "right_finger_y", "right_finger_z"]].to_numpy()
    left_finger_pos = force_segment[["left_finger_x", "left_finger_y", "left_finger_z"]].to_numpy()
    grasp_pos = 0.5 * (right_finger_pos + left_finger_pos)

    left = force_segment["obj_left_finger"].to_numpy(dtype=bool)
    right = force_segment["obj_right_finger"].to_numpy(dtype=bool)
    bimanual_contact = np.logical_and(left, right)
    if np.any(bimanual_contact):
        grasp_pos = grasp_pos[bimanual_contact]
        com_pos = com_pos[bimanual_contact]

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
    first_support_contact_idx: int | None,
    first_grasp_contact_idx: int | None,
    first_lift_idx: int | None,
    accidental_release_idx: int | None,
    release_event_idx: int | None,
    release_midair: bool,
    first_obstacle_contact_idx: int | None,
    lift_command_step: int | None,
    early_drop_case: bool,
    place_mode: str,
    pressing: np.ndarray,
) -> str | None:
    seg_end = start + len(force_segment) - 1
    includes_grasp_contact = first_grasp_contact_idx is not None and (start <= first_grasp_contact_idx <= seg_end)
    includes_lift = first_lift_idx is not None and (start <= first_lift_idx <= seg_end)
    includes_accidental_release = accidental_release_idx is not None and start <= accidental_release_idx <= seg_end
    includes_release = release_event_idx is not None and start <= release_event_idx <= seg_end
    includes_place = first_support_contact_idx is not None and start <= first_support_contact_idx <= seg_end
    includes_obstacle_contact = (
        first_obstacle_contact_idx is not None and (start <= first_obstacle_contact_idx <= seg_end)
    )
    segment_pressing = pressing[start : seg_end + 1]
    includes_pressing = bool(np.any(segment_pressing))

    # A failed lift can contain the commanded lift and confirmed finger-contact loss
    # in the same model window. Keep both semantics in the annotation while mapping
    # the classification label to accidental drop downstream.
    includes_lift_command = lift_command_step is not None and start <= lift_command_step <= seg_end
    lift_was_attempted = (
        lift_command_step is not None
        and accidental_release_idx is not None
        and accidental_release_idx >= lift_command_step
    )
    if early_drop_case and includes_accidental_release:
        return (
            "lift then accidental drop"
            if lift_was_attempted or includes_lift_command or includes_lift
            else "accidental drop"
        )

    if place_mode == "bump" and includes_obstacle_contact:
        return "bump"

    if includes_release:
        if release_midair:
            return "release midair"
        if includes_place and first_support_contact_idx <= release_event_idx:
            return "place then release"
        if includes_pressing:
            return "press then release"
        return "release"

    # A window containing the initial support-contact event is always place,
    # even if downward pressing begins near its end.  This prevents the
    # deliberately unsupported "place then press" composite label.
    if includes_place:
        return PLACE_ACTION

    if includes_pressing:
        return "press"

    # Once placement or intentional release has happened, unsupported waiting,
    # settling and retreat have no action annotation.
    if first_support_contact_idx is not None and first_support_contact_idx < start:
        return None
    if release_event_idx is not None and release_event_idx < start:
        return None

    if includes_grasp_contact and includes_lift:
        return "grasp then lift"
    if includes_grasp_contact:
        return "grasp"
    if includes_lift:
        return "lift"
    return "hold"


def _weight_descriptor(force_segment: pd.DataFrame) -> str:
    if "obj_mass" not in force_segment.columns or force_segment.empty:
        return "an object"
    mass = float(force_segment["obj_mass"].iloc[0])
    if mass >= 0.2:
        return "a heavy object"
    return "a light object"


def _placement_outcome(action: str, metadata: dict) -> str | None:
    if not INCLUDE_PLACEMENT_OUTCOME or action not in {PLACE_ACTION, "place then release"}:
        return None
    toppled = metadata.get("toppled")
    if toppled is True:
        return "topple"
    if toppled is False:
        return "upright"
    return None


def _build_annotation(
    action: str,
    interaction: str | None,
    force_segment: pd.DataFrame,
    placement_outcome: str | None = None,
) -> str:
    obj_phrase = _weight_descriptor(force_segment)
    slow_slip = interaction == "slow slip"
    fast_slip = interaction == "fast slip"

    if action == "grasp":
        if fast_slip:
            return f"Grasping {obj_phrase}, letting it slip quickly."
        if slow_slip:
            return f"Grasping {obj_phrase}, letting it slip slowly."
        return f"Grasping {obj_phrase} with stable contact."

    if action == "lift":
        if fast_slip:
            return f"Lifting {obj_phrase}, letting it slip quickly."
        if slow_slip:
            return f"Lifting {obj_phrase}, letting it slip slowly."
        return f"Lifting {obj_phrase} with stable contact."

    if action == "grasp then lift":
        if fast_slip:
            return f"Grasping then lifting {obj_phrase}, letting it slip quickly."
        if slow_slip:
            return f"Grasping then lifting {obj_phrase}, letting it slip slowly."
        return f"Grasping then lifting {obj_phrase} with stable contact."

    if action == "hold":
        if fast_slip:
            return f"Holding {obj_phrase}, letting it slip quickly."
        if slow_slip:
            return f"Holding {obj_phrase}, letting it slip slowly."
        return f"Holding {obj_phrase} steadily."

    if action == "accidental drop":
        return f"Accidentally dropped {obj_phrase}."

    if action == "lift then accidental drop":
        return f"Attempted to lift {obj_phrase}, but accidentally dropped it."

    if action == "release":
        return f"Releasing {obj_phrase} from the supporting surface."

    if action == "release midair":
        return f"Releasing {obj_phrase} in midair."

    if action == "bump":
        if fast_slip:
            return f"Bumping {obj_phrase}, letting it slip quickly."
        if slow_slip:
            return f"Bumping {obj_phrase}, letting it slip slowly."
        return f"Bumping {obj_phrase} with stable contact."

    if action == PLACE_ACTION:
        if fast_slip:
            annotation = f"Placing {obj_phrase}, letting it slip quickly"
        elif slow_slip:
            annotation = f"Placing {obj_phrase}, letting it slip slowly"
        else:
            annotation = f"Placing {obj_phrase} with stable contact"
        if placement_outcome in {"topple", "topples after release"}:
            return f"{annotation}; it topples after release."
        if placement_outcome in {"upright", "remains upright"}:
            return f"{annotation}; it remains upright after release."
        return f"{annotation}."

    if action == "place then release":
        return f"Placing {obj_phrase} on the supporting surface and then releasing it."

    if action == "press":
        if fast_slip:
            return f"Pressing {obj_phrase}, letting it slip quickly."
        if slow_slip:
            return f"Pressing {obj_phrase}, letting it slip slowly."
        return f"Pressing {obj_phrase} with stable contact."

    if action == "press then release":
        if fast_slip:
            return f"Pressing {obj_phrase} and then releasing it, while it slips quickly."
        if slow_slip:
            return f"Pressing {obj_phrase} and then releasing it, while it slips slowly."
        return f"Pressing {obj_phrase} and then releasing it."

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
    annotations_df = pd.DataFrame(
        columns=["action", "interaction", "placement_outcome", "start", "annotation"]
    )

    steps_csv = os.path.join(csv_path, f"{obj_name}_{material}_steps_{deformation}.csv")
    force_csv = os.path.join(csv_path, f"{obj_name}_{material}_{deformation}.csv")
    if not os.path.exists(steps_csv) or not os.path.exists(force_csv):
        print(f"Skipping {obj_name}/{deformation}: missing steps or force CSV.")
        return

    steps_df = pd.read_csv(steps_csv)
    force_df = pd.read_csv(force_csv)
    metadata = _load_metadata(csv_path, obj_name, material, deformation)

    rollout_bug = detect_rollout_bug(force_df)
    windows = split_for_model(force_df)
    place_start_step = _first_step(steps_df, {"place", "drop", "gentle_place", "press_place", "bump_push"})
    release_command_step = _first_step(steps_df, {"release"})
    lift_command_step = _first_step(steps_df, {"lift"})
    first_support_contact_idx = _first_support_contact_after(force_df, metadata, start_step=place_start_step)
    first_obstacle_contact_idx = _first_obstacle_contact_after(force_df, start_step=place_start_step)
    first_grasp_contact_idx = _first_grasp_contact_index(force_df)
    first_lift_idx = _first_support_detach_after(force_df, metadata, start_step=lift_command_step)
    hold_start_step = _first_step(steps_df, {"hold"})
    release_search_start = hold_start_step if hold_start_step is not None else place_start_step
    early_drop_case = _is_early_drop_case(deformation, metadata, steps_df)
    place_mode = _infer_place_mode(deformation, metadata, steps_df)
    if early_drop_case:
        # A failed pickup/transport may hit the floor before the five-frame
        # finger-loss confirmation completes.  That impact belongs to the
        # accidental-drop event and must never be relabeled as intentional place.
        first_support_contact_idx = None
    accidental_release_idx = (
        _first_release_index(force_df, release_search_start) if early_drop_case else None
    )
    if early_drop_case:
        _append_terminal_window(force_df, windows, accidental_release_idx)

    # New rollouts log the beginning of hand opening explicitly.  Confirm that
    # object contact is subsequently lost, then use the command step so the
    # selected window actually contains the opening motion.  Keep a legacy
    # fallback for already-generated, non-early-drop rollouts.
    if release_command_step is not None:
        confirmed_release_idx = _first_release_index(force_df, release_command_step)
        release_event_idx = release_command_step if confirmed_release_idx is not None else None
    elif not early_drop_case:
        confirmed_release_idx = _first_release_index(force_df, place_start_step)
        release_event_idx = confirmed_release_idx
    else:
        confirmed_release_idx = None
        release_event_idx = None

    support_contact = _support_contact_series(force_df, metadata)
    release_midair = False
    if release_event_idx is not None:
        release_begin = max(0, int(release_event_idx))
        release_end = min(len(support_contact), release_begin + CONTACT_CONFIRM_STEPS)
        release_midair = release_end - release_begin < CONTACT_CONFIRM_STEPS or not bool(
            np.all(support_contact[release_begin:release_end])
        )

    pressing = (
        np.zeros(len(force_df), dtype=bool)
        if early_drop_case
        else _pressing_series(
            force_df,
            metadata,
            place_idx=first_support_contact_idx,
            release_command_step=release_event_idx,
        )
    )

    if rollout_bug is not None:
        print(f"Excluding invalid rollout {obj_name}/{deformation}: {rollout_bug}")

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
        if force_window_is_all_zero(force_df, start):
            continue

        action = _determine_action(
            start=start,
            force_segment=force_segment,
            first_support_contact_idx=first_support_contact_idx,
            first_grasp_contact_idx=first_grasp_contact_idx,
            first_lift_idx=first_lift_idx,
            accidental_release_idx=accidental_release_idx,
            release_event_idx=release_event_idx,
            release_midair=release_midair,
            first_obstacle_contact_idx=first_obstacle_contact_idx,
            lift_command_step=lift_command_step,
            early_drop_case=early_drop_case,
            place_mode=place_mode,
            pressing=pressing,
        )
        if action is None:
            continue
        interaction = (
            None
            if action in {
                "accidental drop",
                "lift then accidental drop",
                "release",
                "release midair",
            }
            else _estimate_interaction(force_segment)
        )
        placement_outcome = _placement_outcome(action, metadata)
        annotation = _build_annotation(action, interaction, force_segment, placement_outcome)

        annotations_df.loc[len(annotations_df)] = {
            "action": action,
            "interaction": interaction,
            "placement_outcome": placement_outcome,
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
