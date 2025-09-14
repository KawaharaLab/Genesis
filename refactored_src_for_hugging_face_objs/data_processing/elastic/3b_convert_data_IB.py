import os
import pandas as pd

COMPLEXITY = 'simple'  # or 'normal' or 'simple'

BASE_PATH = "/home/mdxuser/Genesis/main/data/picked_up"

ANNOTATION_PATH = os.path.join(BASE_PATH, f"{COMPLEXITY}_annotations_2")
CSV_PATH = os.path.join(BASE_PATH, "picked_up_upsampled_IB_2")

MATERIAL = "Elastic"

# Directory containing all FORMAT 2 CSVs
# Output CSV in FORMAT 1
output_csv = os.path.join(BASE_PATH, f"{COMPLEXITY}_formatted_training_data_IB_2.csv")
LENGTH = 200  # Number of steps to interpolate to


# List to collect rows for FORMAT 1
format1_rows = []

# Traverse all CSVs in the directory
for annotation_file_name in os.listdir(ANNOTATION_PATH):
    
    if annotation_file_name.endswith(".csv"):
        core = annotation_file_name.removesuffix("_annotations.csv")
        parts = core.split('_')
        deformation = parts[-1]
        name = '_'.join(parts[:-2])

        annotation_file_path = os.path.join(ANNOTATION_PATH, annotation_file_name)
        csv_path = os.path.join(CSV_PATH, f"{name}_{deformation}_upsampled_IB_2.csv")
        df = pd.read_csv(annotation_file_path)

        for _, row in df.iterrows():
            # Get the starting timestep and annotation
            timestep_start = _*LENGTH
            annotation = row["annotation"]
            format1_rows.append({
                "csv_path": csv_path,
                "timestep_start": timestep_start,
                "annotation": annotation
            })
            timestep_start = int((_ + 0.5)*LENGTH)
            format1_rows.append({
                "csv_path": csv_path,
                "timestep_start": timestep_start,
                "annotation": annotation
            })

# Create a new DataFrame and save to CSV
format1_df = pd.DataFrame(format1_rows)
format1_df.to_csv(output_csv, index=False)

print(f"Converted {len(format1_rows)} annotations to {output_csv}")