import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os
import math

# Load your full CSV
# df = pd.read_csv("/Users/no.166/Documents/Azka's Workspace/Genesis/main/data/picked_up_csv/ASICS_GELBlur33_20_GS_BlackWhiteSafety_Orange/Elastic/soft/ASICS_GELBlur33_20_GS_BlackWhiteSafety_Orange_Elastic_soft.csv")
# df_steps = pd.read_csv("/Users/no.166/Documents/Azka's Workspace/Genesis/main/data/annotations/ASICS_GELBlur33_20_GS_BlackWhiteSafety_Orange_Elastic_soft_annotations.csv")

BASE_PATH = "/home/mdxuser/Genesis/main/data/picked_up"

DATA_DIR = os.path.join(BASE_PATH, "csv")
os.makedirs(os.path.join(BASE_PATH, "picked_up_upsampled_IB_2"), exist_ok=True)  # exists_ok=True to avoid errors if the directory already exists
OUTPUT_DIR = os.path.join(BASE_PATH, "picked_up_upsampled_IB_2")
os.makedirs(OUTPUT_DIR, exist_ok=True) # exists_ok=True to avoid errors if the directory already exists
LENGTH = 200  # Number of steps to interpolate to
MATERIAL = "Elastic"

def main(df, df_steps):
    # Partition boundaries
    partitions = df_steps['step start'].values.tolist()
    partitions.append(df.iloc[-1,0])  # Extend the last partition to include the end
    # partitions = [0, 50, 200, 350, 550, 730, 910, 1150, 1420, 1585, 1750, 1850]
    print(partitions)

    # Prepare list to collect each interpolated partition
    interpolated_parts = []

    # For each partition:
    for i in range(len(partitions) - 1):
        if (partitions[i+1] - partitions[i]) % 2 == 0:
            start, middle, end = partitions[i]+1, (partitions[i]+partitions[i+1])/2, partitions[i+1]   
        else:
            start, middle, end = partitions[i]+1, math.floor((partitions[i]+partitions[i+1])/2), partitions[i+1]
        
        # Extract the subset
        segment_1 = df[(df['step'] >= start) & (df['step'] <= middle)]
        segment_2 = df[(df['step'] >= middle) & (df['step'] <= end)]
        # print(f"Segment {i}: step range {start} to {end} → {len(segment)} rows")
        
        x1, x2, x3 = LENGTH*i+1, LENGTH*(i+0.5), LENGTH*(i+1)

        # Original x (step) and new x (270 evenly spaced steps from start to end)
        original_x1 = segment_1['step'].values
        original_x2 = segment_2['step'].values
        # new_x1 = np.linspace(LENGTH*i+1, math.floor(LENGTH*(i+1)/2))
        new_x1 = np.round(np.linspace(x1, x2, int(LENGTH/2))).astype(int)
        new_x2 = np.round(np.linspace(x2, x3, int(LENGTH/2))).astype(int)
        
        # Dictionary to store interpolated data
        interp_segment_1 = {'step': new_x1}
        interp_segment_2 = {'step': new_x2}
        
        # Interpolate each column
        for col in df.columns:
            if col == 'step':
                continue
            f1 = interp1d(original_x1, segment_1[col].values, kind='linear', fill_value="extrapolate")
            interp_segment_1[col] = f1(new_x1)
            f2 = interp1d(original_x2, segment_2[col].values, kind='linear', fill_value="extrapolate")
            interp_segment_2[col] = f2(new_x2)

            
        
        # Convert to DataFrame and append
        interpolated_parts.append(pd.DataFrame(interp_segment_1))
        interpolated_parts.append(pd.DataFrame(interp_segment_2))

    # Concatenate all interpolated segments
    df_upsampled = pd.concat(interpolated_parts, ignore_index=True)

    # Save or use
    # df_upsampled.to_csv(os.path.join(OUTPUT_DIR,f"test_run_upsampled.csv"), index=False)
    return df_upsampled

for object in os.listdir(DATA_DIR):
    if os.path.isdir(os.path.join(DATA_DIR, object)):
        for deformation in os.listdir(os.path.join(DATA_DIR, object, f"{MATERIAL}")):
            if deformation == '.DS_Store':
                continue
            print(f"Processing {object} and deformation {deformation}")
            df = pd.read_csv(os.path.join(DATA_DIR, object, f"{MATERIAL}", f"{deformation}", f"{object}_{MATERIAL}_{deformation}.csv"))
            df_steps = pd.read_csv(os.path.join(BASE_PATH, "simple_annotations_2", f"{object}_{MATERIAL}_{deformation}_annotations.csv"))
            df_upsampled = main(df, df_steps)
            df_upsampled.to_csv(os.path.join(OUTPUT_DIR,f"{object}_{deformation}_upsampled_IB_2.csv"), index=False)
