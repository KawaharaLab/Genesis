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
        place_start_step=40,
        first_support_contact_idx=80,
        first_grasp_contact_idx=None,
        first_lift_idx=None,
        first_release_idx=None,
        first_obstacle_contact_idx=None,
        bump_push_step=None,
        press_place_step=None,
        lift_command_step=None,
        early_drop_case=False,
        place_mode="gentle",
        target_contact=np.ones(200, dtype=bool),
    )

    assert action is None


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
