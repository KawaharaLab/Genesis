import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# Load your full CSV
df = pd.read_csv("/Users/no.166/Documents/Azka's Workspace/Genesis/main/data/picked_up_3/csv/ASICS_GELBlur33_20_GS_BlackWhiteSafety_Orange/Elastic/soft/ASICS_GELBlur33_20_GS_BlackWhiteSafety_Orange_Elastic_soft.csv")
df_steps = pd.read_csv("/Users/no.166/Documents/Azka's Workspace/Genesis/main/data/picked_up_3/annotations/ASICS_GELBlur33_20_GS_BlackWhiteSafety_Orange_Elastic_soft_annotations.csv")
# Partition boundaries
partitions = df_steps['step start'].values.tolist()
partitions.append(partitions[-1] + 100)  # Extend the last partition to include the end
# partitions = [0, 50, 200, 350, 550, 730, 910, 1150, 1420, 1585, 1750, 1850]
print(partitions)

# Prepare list to collect each interpolated partition
interpolated_parts = []

# For each partition:
for i in range(len(partitions) - 1):
    start, end = partitions[i], partitions[i+1]
    
    # Extract the subset
    segment = df[(df['step'] >= start) & (df['step'] <= end)]
    print(f"Segment {i}: step range {start} to {end} → {len(segment)} rows")
    
    # Original x (step) and new x (270 evenly spaced steps from start to end)
    original_x = segment['step'].values
    new_x = np.linspace(start, end, 270)
    
    # Dictionary to store interpolated data
    interp_segment = {'step': new_x}
    
    # Interpolate each column
    for col in df.columns:
        if col == 'step':
            continue
        f = interp1d(original_x, segment[col].values, kind='linear', fill_value="extrapolate")
        interp_segment[col] = f(new_x)
    
    # Convert to DataFrame and append
    interpolated_parts.append(pd.DataFrame(interp_segment))

# Concatenate all interpolated segments
df_upsampled = pd.concat(interpolated_parts, ignore_index=True)

# Save or use
df_upsampled.to_csv("upsampled_all_partitions.csv", index=False)
