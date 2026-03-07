import os
import pandas as pd
import numpy as np
from multiprocessing import Process
import time
from pathlib import Path
import csv

BASE_PATH = Path(__file__).resolve().parent.parent.parent.parent
SEQUENCE_LENGTH = 80
DATASET = "train_03062026" 
WINDOW_STRIDE = 40 if "eval" in DATASET else 10
#------------------- Generate labels -------------------#

def extract_floats_from_string(data_string: str) -> list[float]:

    if pd.isna(data_string):
        return []
    # Remove brackets/quotes and split on whitespace
    cleaned = data_string.replace("[", " ").replace("]", " ").replace("\n", " ")
    return [float(x) for x in cleaned.split() if x]


def action_at_step(steps_df: pd.DataFrame, step_value: int | float) -> str:
    """
    Find the action label whose 'step' is the largest value <= step_value.
    If all steps are greater than step_value, return the first row's action.

    Assumes steps_df has columns ['action','step', ...].
    """
    # Ensure sorted by step ascending
    sdf = steps_df.sort_values('step', kind='mergesort')  # stable
    steps = sdf['step'].to_numpy()
    # position of rightmost value <= step_value
    pos = np.searchsorted(steps, step_value, side='right') - 1
    if pos < 0:
        pos = 0

    return str(sdf.iloc[pos]['action'])

def detect_bugs(force_df: pd.DataFrame, start: int) -> bool:
    """
    Check for bugs in the force_df DataFrame at the given start step.
    For example, if the fingers penetrate each other, the joint angles would be smaller than zero.
    Returns True if a bug is detected, False otherwise.
    The finger joint angles are sometimes small negative values when fingers are closed tight (and holding nothing)
    Force sensors may also report small negative values when fingers are free and open.
    """
    # left_minus = np.any(force_df['dof_7'][start:start+SEQUENCE_LENGTH].to_numpy() < -0.0001)
    # force_left_minus = np.any(force_df['left_fy'][start:start+SEQUENCE_LENGTH].to_numpy() < -10)
    return False
    # return True if left_minus and force_left_minus else False

def split_for_model(step_df, force_df):
    """
    cut and merge the step_df so that each 'action' is the length of SEQUENCE_LENGTH.
    For actions that are longer than SEQUENCE_LENGTH, split them into multiple actions.
    For actions that are shorter than SEQUENCE_LENGTH, merge them with the next action. The new action will be the two actions combined. 
    """
    start = 0
    added_steps_dicts = []
    while start + SEQUENCE_LENGTH <= len(force_df):
        end = start + SEQUENCE_LENGTH
        action_label_start = action_at_step(step_df, start)
        action_label_end = action_at_step(step_df, end)
        if (action_label_start != action_label_end):
            action = action_label_start + " then " + action_label_end # TODO: memorize the timestep when the action changed
        else:
            action = action_label_start

        if not detect_bugs(force_df, start):
            added_steps_dicts.append({'action': action, 'start': start, 'mass': force_df['obj_mass'][start]})
        start += WINDOW_STRIDE

    return added_steps_dicts


def get_picked_up_objects(all_objects, material='Rigid'):
    to_do = []
    for obj_name in all_objects:
        picked_up_path = os.path.join(BASE_PATH, 'data', DATASET, 'csv' , obj_name, material, 'none' )
        print(f'pup {picked_up_path}')
        if obj_name == '.DS_Store':
            continue

        # if csv_path is empty, then do not include this object
        csv_path = os.path.join(picked_up_path, f'{obj_name}_{material}_none.csv')
        if not os.path.exists(csv_path):
            print(f'❌ {obj_name} (no csv)')
            continue
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # Read header (if present)
            first_data_row = next(reader, None)

            if first_data_row is None:
                print("File has no data rows")
            else:
                print(f'✅ {obj_name}')
                to_do.append((obj_name, 'none'))

    return to_do


def generate_sentence(action: str, force_df: pd.DataFrame, lost_contact: bool) -> str:
    """
    Generate a sentence using selected values.
    """
    contact_state = "init"
    
    contact_left = force_df['obj_left_finger'].to_numpy()
    contact_right = force_df['obj_right_finger'].to_numpy()
    contact_either = np.logical_or(contact_left, contact_right)
    contact_both = np.logical_and(contact_left, contact_right)
    touched_both = False
    touched_either = np.any(contact_either)
    touched_idx = -1
    released_idx = SEQUENCE_LENGTH - 1
    for i in range(len(contact_both)):
        if contact_both[i] and not touched_both:
            touched_both = True
            touched_idx = i
        if touched_both and not contact_both[i]:
            # if force_df['obj_min_z'].values[i] > 0.03:
            contact_state = "letting it drop"
            released_idx = i
            break
    if "wiggle" in action:
        action = action.replace("wiggle", "shake")
    if lost_contact or not touched_both:
        if touched_either:
            return "touch an object."
        else:
            return "hand is empty."

    if "stop" in action:
        action = action.replace("stop", "hold")

    contact_range = np.array([False] * len(force_df))
    contact_range[touched_idx:released_idx+1] = True
    annotation = ""

    mass = force_df['obj_mass'].values[0]
    mass_str = "heavy" if mass > 0.1 else "light" #TODO: Check whether 0.5 [kg] is a good threshold

    annotation += f"{action} a {mass_str} object, " # explain the movement very simply

    com_pos = force_df[['obj_COM_x', 'obj_COM_y', 'obj_COM_z']].to_numpy()
    right_finger_pos = force_df[['right_finger_x', 'right_finger_y', 'right_finger_z']].to_numpy()
    left_finger_pos = force_df[['left_finger_x', 'left_finger_y', 'left_finger_z']].to_numpy()
    grasp_pos = (right_finger_pos + left_finger_pos)/2

    grasp_pos = grasp_pos[contact_range]
    com_pos = com_pos[contact_range]
    distances = np.linalg.norm(grasp_pos - com_pos, axis=1)
    # slip_velocities = np.diff(distances)
    if contact_state == "init":
        # if np.any(slip_velocities > 0.0001):
        if np.max(distances) - np.min(distances) > 0.001:
            contact_state = "letting it slip"
        else:
            contact_state = "keeping it stable"
    annotation += f"{contact_state}."
    return annotation


def main(obj_name, csv_path, deformation, material='Rigid'):
    annotations_df = pd.DataFrame(columns=['action','start', 'annotation'])

    # steps_csv: action, step, hand_coordinate, bounding_box [need to convert from np and pd]
    steps_csv = os.path.join(csv_path, f"{obj_name}_{material}_steps_{deformation}.csv")
    # force_csv: step, left_fx, left_fy, left_fz, left_tx, left_ty, left_tz, right_fx, right_fy, right_fz, right_tx, right_ty, right_tz, dof_0, dof_1, dof_2, dof_3, dof_4, dof_5, dof_6, dof_7, dof_8
    force_csv = os.path.join(csv_path, f"{obj_name}_{material}_{deformation}.csv")

    steps_df = pd.read_csv(steps_csv)
    force_df = pd.read_csv(force_csv)

    added_steps_dicts = split_for_model(steps_df, force_df)

    lost_contact = False
    for row in added_steps_dicts:
        if detect_bugs(force_df, row['start']):
            lost_contact = True
            print("Fingers phased through each other, skipping...")
            continue
        if row['start'] + SEQUENCE_LENGTH > len(force_df):
           continue
        force_csv_segment = force_df.iloc[row['start']:row['start']+SEQUENCE_LENGTH].reset_index(drop=True)
        if force_csv_segment.isnull().values.any():
            print("Skipping segment with NaN values")
            continue
        annotation = generate_sentence(row['action'], force_csv_segment, lost_contact)
        annotations_df.loc[len(annotations_df)] = {'action': row['action'], 'start': row['start'], 'annotation': annotation}
        if row["start"] >= 300 and annotation == "hand is empty.":
            lost_contact = True


    # ------------------- Save the annotations to a CSV file -------------------#
    output_csv_path = os.path.join(BASE_PATH, 'data', DATASET, f'csv', obj_name, material, deformation, f"{obj_name}_{material}_{deformation}_annotations.csv")
    annotations_df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    folder_path = os.path.join(BASE_PATH, "data", DATASET, "csv")
    all_objects = os.listdir(folder_path)
    selected_objects = get_picked_up_objects(all_objects)
    # selected_objects = [('Crayola_Bonus_64_Crayons', 'medium')]

    material = 'Rigid'
    processes = []

    for task in selected_objects:
        obj_name, deformation = task
        picked_up_path = os.path.join(folder_path, obj_name, material, deformation)
        print(f"Processing {obj_name} with target {deformation}...")

        while len(processes) >= 8:
            processes = [p for p in processes if p.is_alive()]
            time.sleep(0.1)  # Wait for some processes to finish

        p = Process(target=main, args=(obj_name, picked_up_path, deformation))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All processes completed.")
