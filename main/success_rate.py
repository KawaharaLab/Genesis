# This code is to run through picked_up_3 and not_picked_up_3 directories and check if the object was picked up successfully.

import os
import pandas as pd
import imageio.v3 as iio
import numpy as np

BASE_PATH = "/Users/no.166/Documents/Azka\'s Workspace/Genesis"


def check_success_rate(base_path, names, df, material="Elastic", targets=['soft', 'medium', 'hard']):
    total_objects = 0
    success_count = 0
    fail_count = 0
    total_count = 0

    for name in names:
        csv_path = os.path.join(base_path, 'main', 'data', 'picked_up_3', 'csv', name, material)
        total_objects += 1

        # Check if this object is already in the DataFrame
        if name not in df['object'].values:
            # Add a new row with just the object name
            df.loc[len(df)] = {'object': name, 'soft': None, 'medium': None, 'hard': None}

        for target in targets:
            target_csv_path = os.path.join(csv_path, target)
            total_count += 1
            if not os.path.isdir(target_csv_path) or not os.listdir(target_csv_path):
                result = '❌'
                fail_count += 1
            else:
                result = '✅'
                success_count += 1

            # Update the specific cell
            df.loc[df['object'] == name, target] = result

    print(f"Total objects: {total_objects}")
    print(f"Success count: {success_count}")
    print(f"Fail count: {fail_count}")
    print(f"Total count: {total_count}")

    return df  # Important: return the modified DataFrame

def delete_empty_directories(base_path, names, material="Elastic", targets=['soft', 'medium', 'hard']):
    for name in names:
        photo_path = os.path.join(base_path, 'main', 'data', 'picked_up_3', 'photos', name, material)
        for target in targets:
            target_csv_path = os.path.join(photo_path, target)
            if not os.path.isdir(target_csv_path) or not os.listdir(target_csv_path):
                print(f"Deleting empty directory: {target_csv_path}")
                # os.rmdir(target_csv_path)  # Remove the empty directory
    


if __name__ == "__main__":
    # Create a DataFrame to store results
    dataframe_columns = ['object', 'soft', 'medium', 'hard']
    df = pd.DataFrame(columns=dataframe_columns)

    folder_path = os.path.join(BASE_PATH, "data", "mujoco_scanned_objects", "models")
    all_files = os.listdir(folder_path)
    # delete_empty_directories(BASE_PATH, all_files)

    df = check_success_rate(BASE_PATH, all_files, df)
    df = df.sort_values(by='object').reset_index(drop=True)


    # Save the DataFrame to a CSV file
    output_csv_path = os.path.join(BASE_PATH, 'main', 'data', 'picked_up_3_success_rate.csv')
    df.to_csv(output_csv_path, index=False)

