import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSING_DIR = ROOT / "src" / "processing" / "rigid"
sys.path.insert(0, str(PROCESSING_DIR))
SPEC = importlib.util.spec_from_file_location(
    "rigid_annotations", PROCESSING_DIR / "1_create_annotations.py"
)
annotations = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(annotations)


def test_first_sustained_state_rejects_short_contact_flicker():
    contact = np.array([True, True, False, False, False, False, True, False, False, False, False, False])

    assert annotations._first_sustained_state(contact, False, 0, confirm_steps=5) == 7


def test_rollout_bug_detects_com_teleport():
    force_df = pd.DataFrame(
        {
            "obj_COM_x": [0.4, 0.4, 0.51],
            "obj_COM_y": [0.4, 0.4, 0.4],
            "obj_COM_z": [0.1, 0.1, 0.1],
        }
    )

    assert annotations.detect_rollout_bug(force_df) == "COM step displacement > 0.05 m"


def test_rollout_bug_rejects_whole_episode_for_extreme_force():
    force_df = pd.DataFrame(
        {
            "obj_COM_x": [0.4, 0.4],
            "obj_COM_y": [0.4, 0.4],
            "obj_COM_z": [0.1, 0.1],
            "left_fx": [1.0, 100.0],
        }
    )

    assert annotations.detect_rollout_bug(force_df) == "contact-force component >= 100 N"


def test_gentle_wait_after_place_is_unannotated():
    force_segment = pd.DataFrame(
        {
            "obj_left_finger": np.ones(20, dtype=bool),
            "obj_right_finger": np.ones(20, dtype=bool),
        }
    )

    action = annotations._determine_action(
        start=100,
        force_segment=force_segment,
        first_support_contact_idx=80,
        first_grasp_contact_idx=None,
        first_lift_idx=None,
        accidental_release_idx=None,
        release_event_idx=None,
        release_midair=False,
        first_obstacle_contact_idx=None,
        lift_command_step=None,
        early_drop_case=False,
        place_mode="gentle",
        pressing=np.zeros(200, dtype=bool),
    )

    assert action is None


def _action(**overrides):
    values = {
        "start": 100,
        "force_segment": pd.DataFrame(
            {
                "obj_left_finger": np.ones(20, dtype=bool),
                "obj_right_finger": np.ones(20, dtype=bool),
            }
        ),
        "first_support_contact_idx": None,
        "first_grasp_contact_idx": None,
        "first_lift_idx": None,
        "accidental_release_idx": None,
        "release_event_idx": None,
        "release_midair": False,
        "first_obstacle_contact_idx": None,
        "lift_command_step": None,
        "early_drop_case": False,
        "place_mode": "gentle",
        "pressing": np.zeros(200, dtype=bool),
    }
    values.update(overrides)
    return annotations._determine_action(**values)


def test_place_event_takes_priority_over_pressing_in_same_window():
    pressing = np.zeros(200, dtype=bool)
    pressing[115:120] = True

    assert _action(first_support_contact_idx=110, pressing=pressing) == "place"


def test_place_then_release_is_distinct_from_press_then_release():
    pressing = np.zeros(200, dtype=bool)
    pressing[105:110] = True

    assert (
        _action(first_support_contact_idx=105, release_event_idx=115, pressing=pressing)
        == "place then release"
    )
    assert _action(release_event_idx=115, pressing=pressing) == "press then release"


def test_midair_release_has_its_own_action():
    assert _action(release_event_idx=115, release_midair=True) == "release midair"


def test_pressing_requires_supported_finger_contact_and_sustained_motion():
    size = 20
    force_df = pd.DataFrame(
        {
            "obj_plane": np.ones(size, dtype=bool),
            "obj_tile": np.zeros(size, dtype=bool),
            "obj_left_finger": np.ones(size, dtype=bool),
            "obj_right_finger": np.zeros(size, dtype=bool),
            "eef_z": np.r_[np.ones(5), 1.0 - np.arange(15) * 0.001],
            "left_fz": np.zeros(size),
            "right_fz": np.zeros(size),
        }
    )

    pressing = annotations._pressing_series(
        force_df, metadata={}, place_idx=4, release_command_step=None
    )
    assert pressing[6:20].all()

    force_df["obj_left_finger"] = False
    assert not annotations._pressing_series(
        force_df, metadata={}, place_idx=4, release_command_step=None
    ).any()


def test_terminal_window_captures_event_between_regular_stride_starts():
    force_df = pd.DataFrame(
        {
            "left_fx": np.ones(107),
            "left_fy": np.zeros(107),
            "left_fz": np.zeros(107),
            "right_fx": np.ones(107),
            "right_fy": np.zeros(107),
            "right_fz": np.zeros(107),
            "left_tx": np.zeros(107),
            "left_ty": np.zeros(107),
            "left_tz": np.zeros(107),
            "right_tx": np.zeros(107),
            "right_ty": np.zeros(107),
            "right_tz": np.zeros(107),
            "obj_left_finger": np.ones(107, dtype=bool),
            "obj_right_finger": np.ones(107, dtype=bool),
        }
    )
    windows = [{"start": 0}]

    annotations._append_terminal_window(force_df, windows, event_idx=106)

    assert windows == [{"start": 0}, {"start": 27}]


def test_legacy_target_contact_uses_tile_overlap_and_height():
    force_df = pd.DataFrame(
        {
            "obj_min_x": [0.43, 0.70],
            "obj_max_x": [0.47, 0.74],
            "obj_min_y": [0.43, 0.43],
            "obj_max_y": [0.47, 0.47],
            "obj_min_z": [0.004, 0.004],
        }
    )
    metadata = {"target_xy": {"x": 0.45, "y": 0.45}}

    np.testing.assert_array_equal(
        annotations._legacy_target_contact_series(force_df, metadata),
        [True, False],
    )
