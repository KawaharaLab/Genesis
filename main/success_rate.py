# This code is to run through picked_up_4 and not_picked_up_4 directories and check if the object was picked up successfully.

import os
import pandas as pd
import imageio.v3 as iio
import numpy as np

BASE_PATH = "/Users/no.166/Documents/Azka\'s Workspace/Genesis"


def check_success_rate(base_path, objects, df, material="Elastic", targets=['soft', 'medium', 'hard']):
    in_progress_list = []
    total_objects = 0
    success_count = 0
    in_progress_count = 0
    fail_count = 0
    total_count = 0
    missing_count = 0

    for name in objects:
        csv_path = os.path.join(base_path, 'main', 'data', 'picked_up_4', 'csv', name, material)
        csv_np_path = os.path.join(base_path, 'main', 'data', 'not_picked_up_4', 'csv', name, material)
        total_objects += 1

        # Check if this object is already in the DataFrame
        if name not in df['object'].values:
            # Add a new row with just the object name
            df.loc[len(df)] = {'object': name, 'soft': None, 'medium': None, 'hard': None}

        # for target in targets:
        #     target_csv_path = os.path.join(csv_path, target)
        #     target_csv_np_path = os.path.join(csv_np_path, target)
        #     total_count += 1

        #     if not os.path.exists(target_csv_path):
        #         if not os.path.exists(target_csv_np_path):
        #             result = '⏳'  # Both directories missing
        #             missing_count += 1
        #         else:
        #             result = '❌'  # CSV missing, NP present
        #             fail_count += 1
        #     elif os.path.isdir(target_csv_path) and not os.listdir(target_csv_path):
        #         result = '🚧'  # CSV directory empty
        #         in_progress_count += 1
        #     else:
        #         result = '✅'  # CSV directory exists and not empty
        #         success_count += 1
            
        #     # Update the specific cell
        #     df.loc[df['object'] == name, target] = result
        

        for target in targets:
            target_csv_path = os.path.join(csv_path, target)
            target_csv_np_path = os.path.join(csv_np_path, target)
            total_count += 1
            if not os.path.isdir(target_csv_path) or not os.listdir(target_csv_path):
                if not os.path.isdir(target_csv_np_path) or not os.listdir(target_csv_np_path):
                    result = '🚧'
                    in_progress_count += 1
                    in_progress_list.append(name)
                else:
                    result = '❌'
                    fail_count += 1
            else:
                result = '✅'
                success_count += 1

            # Update the specific cell
            df.loc[df['object'] == name, target] = result

    print(f"Total objects: {total_objects}")
    # print(f'Total completed: {success_count + in_progress_count + fail_count}')
    print(f"Success count: {success_count}")
    print(f"In progress count: {in_progress_count}")
    print(f"Fail count: {fail_count}")
    print(f"Total count: {total_count}")
    print(f"Missing count: {missing_count}")
    print(f"Success rate: {success_count / (success_count + fail_count) * 100:.2f}%")

    return df, in_progress_list  # Important: return the modified DataFrame

def delete_empty_directories(base_path, names, material="Elastic", targets=['soft', 'medium', 'hard']):
    for name in names:
        photo_path = os.path.join(base_path, 'main', 'data', 'picked_up_4', 'photos', name, material)
        for target in targets:
            target_csv_path = os.path.join(photo_path, target)
            if not os.path.isdir(target_csv_path) or not os.listdir(target_csv_path):
                print(f"Deleting empty directory: {target_csv_path}")
                # os.rmdir(target_csv_path)  # Remove the empty directory
    


if __name__ == "__main__":
    # Create a DataFrame to store results
    dataframe_columns = ['object', 'soft', 'medium', 'hard']
    df = pd.DataFrame(columns=dataframe_columns)

    folder_path = os.path.join(BASE_PATH, 'main', 'data', 'picked_up_4', 'csv')
    all_files = os.listdir(folder_path)
    # delete_empty_directories(BASE_PATH, all_files)

    df, in_progress_list = check_success_rate(BASE_PATH, all_files, df)
    df = df.sort_values(by='object').reset_index(drop=True)

    print("In progress objects:", in_progress_list)

    # Save the DataFrame to a CSV file
    output_csv_path = os.path.join(BASE_PATH, 'main', 'data', 'picked_up_4_success_rate.csv')
    df.to_csv(output_csv_path, index=False)

