import os
import pandas as pd
import numpy as np
from multiprocessing import Process, Manager  # Import Manager
import time
from pathlib import Path
import csv

from simple_annotation_bank import RobotLabelTemplate

BASE_PATH = Path(__file__).resolve().parent.parent.parent.parent
SEQUENCE_LENGTH = 80
WINDOW_STRIDE = 10
DATA_TYPE = "eval"

#------------------- Generate labels -------------------#

def extract_floats_from_string(data_string: str) -> list[float]:

    if pd.isna(data_string):
        return []
    # Remove brackets/quotes and split on whitespace
    cleaned = data_string.replace("[", " ").replace("]", " ").replace("\n", " ")
    return [float(x) for x in cleaned.split() if x]


def detect_bugs(force_df: pd.DataFrame, start: int) -> bool:
    """
    Check for bugs in the force_df DataFrame at the given start step.
    For example, if the fingers penetrate each other, the joint angles would be smaller than zero.
    Returns True if a bug is detected, False otherwise.
    The finger joint angles are sometimes small negative values when fingers are closed tight (and holding nothing)
    Force sensors may also report small negative values when fingers are free and open.
    """
    left_minus = np.any(force_df['dof_7'][start:start+80].to_numpy() < -0.0001)
    force_left_minus = np.any(force_df['left_fy'][start:start+80].to_numpy() < -0.01)
    return True if left_minus and force_left_minus else False


def get_picked_up_objects(all_objects, material='Rigid'):
    to_do = []
    for obj_name in all_objects:
        picked_up_path = os.path.join(BASE_PATH, 'data', DATA_TYPE, 'csv' , obj_name, material, 'none' )
        print(f'pup {picked_up_path}')
        if obj_name == '.DS_Store':
            continue
        
        csv_path = os.path.join(picked_up_path, f'{obj_name}_{material}_none.csv')
        if not os.path.exists(csv_path):
            print(f'❌ {obj_name} (no csv)')
            continue
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            first_data_row = next(reader, None)

            if first_data_row is None:
                print("File has no data rows")
            else:
                print(f'✅ {obj_name}')
                to_do.append((obj_name, 'none'))
        
    return to_do

def main(obj_name, csv_path, deformation, mass, material='Rigid'):
    annotations_df = pd.DataFrame(columns=['action','start', 'annotation', 'mass'])

    steps_csv = os.path.join(csv_path, f"{obj_name}_{material}_steps_{deformation}.csv")
    force_csv = os.path.join(csv_path, f"{obj_name}_{material}_{deformation}.csv")

    steps_df = pd.read_csv(steps_csv)
    force_df = pd.read_csv(force_csv)

    print(force_df['obj_mass'].values[0])
    mass.append(force_df['obj_mass'].values[0]) # Use append for the shared list

if __name__ == "__main__":
    folder_path = os.path.join(BASE_PATH, "data", DATA_TYPE, "csv")
    all_objects = os.listdir(folder_path)
    selected_objects = get_picked_up_objects(all_objects)

    material = 'Rigid'
    processes = []
    
    with Manager() as manager:
        mass = manager.list() # Create a shared list managed by the Manager

        for task in selected_objects:
            obj_name, deformation = task
            picked_up_path = os.path.join(folder_path, obj_name, material, deformation)
            print(f"Processing {obj_name} with target {deformation}...")

            while len(processes) >= 8:
                processes = [p for p in processes if p.is_alive()]
                time.sleep(0.1)

            p = Process(target=main, args=(obj_name, picked_up_path, deformation, mass))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
        
        # Convert the shared list to a numpy array for calculations
        final_mass_array = np.array(list(mass))

        #print(final_mass_array)
        print("mean:", np.mean(final_mass_array))
        print("median:", np.median(final_mass_array))
        print("All processes completed.")