import os
import pandas as pd
ANNOTATION_PATH = '/Users/no.166/Documents/Azka\'s Workspace/Genesis/main/data/picked_up_4/annotations'
CSV_PATH = '/Users/no.166/Documents/Azka\'s Workspace/Genesis/main/data/picked_up_4/csv'
MATERIAL = "Elastic"

# Directory containing all FORMAT 2 CSVs
input_dir = ANNOTATION_PATH
# Output CSV in FORMAT 1
output_csv = "/Users/no.166/Documents/Azka\'s Workspace/ImageBind/data/train.csv"


# List to collect rows for FORMAT 1
format1_rows = []

# Traverse all CSVs in the directory
for filename in os.listdir(input_dir):
    
    if filename.endswith(".csv"):
        core = filename.removesuffix("_annotations.csv")
        parts = core.split('_')
        deformation = parts[-1]
        name = '_'.join(parts[:-2])

        filepath = os.path.join(input_dir, filename)
        csv_path = os.path.join(CSV_PATH, name, MATERIAL, deformation, f"{name}_{MATERIAL}_{deformation}.csv")
        df = pd.read_csv(filepath)

        for _, row in df.iterrows():
            # Get the starting timestep and annotation
            timestep_start = row["step start"]
            annotation = row["annotation"]
            format1_rows.append({
                "csv_path": csv_path,
                "timestep_start": timestep_start,
                "annotation": annotation
            })

# Create a new DataFrame and save to CSV
format1_df = pd.DataFrame(format1_rows)
format1_df.to_csv(output_csv, index=False)

print(f"Converted {len(format1_rows)} annotations to {output_csv}")