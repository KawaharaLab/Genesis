from annotation_bank import RobotLabelTemplate
import os
import pandas as pd
import numpy as np
from multiprocessing import Process
import time

BASE_PATH = '/Users/no.166/Documents/Azka\'s Workspace/Genesis'

#------------------- Generate labels -------------------#

def extract_floats_from_string(data_string: str) -> list[float]:

    arr = eval(data_string)  # Evaluates the string as Python code
    arr = np.array(arr)  # Convert to NumPy array if not already

    return arr

def split_for_model(df):
    unique_array = np.sort(df.iloc[:,1].unique())
    # print(f"Unique steps: {unique_array}")
    expanded = []
    expanded_for_bbox = []
    to_return = []
    max_jump = 270
    inserted_points = []
    for i in range(len(unique_array) - 1):
        start = unique_array[i]
        end = unique_array[i + 1]
        expanded.append(start)
        expanded_for_bbox.append(start)
        
        jump = end - start
        if jump > max_jump:
            midpoint = (start + end) / 2
            expanded.append(midpoint)
            expanded_for_bbox.append(start)
            inserted_points.append((i, midpoint))
            
    expanded.append(unique_array[-1])
    expanded_for_bbox.append(unique_array[-1])
    expanded_array = np.round(expanded).astype(int)
    expanded_for_bbox = np.round(expanded_for_bbox).astype(int)
    # print(f"Expanded array: {expanded_array}, Inserted points: {inserted_points}")
    # print("\nInserted midpoints:")
    for idx, value in inserted_points:
        to_return.append(idx)
    pairings = [[int(a), int(b)] for a, b in zip(expanded_array[:-1], expanded_array[1:])]
    return pairings, to_return, expanded_for_bbox



#-------------------- Create annotations based on steps -------------------#
def logical(i, deform_csv, force_csv, steps_csv, deformation, start_step, end_step, insertions, exp_bbox_idx, drop_logic=None):
    actions = ['start', 'grasp', 'lift', 'rotation 1', 'buffer 1', 'rotation 2', 'buffer 2', 'wind_down']
    replacement = {1:['grasp pt1','grasp pt2'],
                   3:['rotation 1 pt1', 'rotation 1 pt2'],
                   5:['rotation 2 pt1', 'rotation 2 pt2'],
                   4:['buffer 1 pt1', 'buffer 1 pt2'],
                   6:['buffer 2 pt1', 'buffer 2 pt2']}
    angle = None
    dropped = None
    if drop_logic is not None:
        dropped = 'dropped'
    offset = 0
    for idx in insertions:
        actions = actions[:idx+offset] + replacement[idx] + actions[idx+1+offset:]
        offset += 1
    action = actions[i]

    # deformation level: ['none', 'soft', 'medium', 'hard']
    if action in ['start', 'wind_down']:
        deformation_level = 'none'
        force_level = 'none'
        add_trends = False
    elif action in ['grasp', 'grasp pt1', 'grasp pt2', 'lift', 'rotation 1', 'rotation 1 pt1', 'rotation 1 pt2', 'buffer 1', 'buffer 1 pt1', 'buffer 1 pt2', 'rotation 2', 'rotation 2 pt1', 'rotation 2 pt2', 'buffer 2', 'buffer 2 pt1', 'buffer 2 pt2']:
        deformation_level = deformation
        if deformation == 'soft':
            force_level = 'low'
        elif deformation == 'medium':
            force_level = 'medium'
        elif deformation == 'hard':
            force_level = 'high'
        if action in ['grasp', 'grasp pt1', 'grasp pt2', 'lift']:
            add_trends = 'increasing'
        elif action in ['rotation 1', 'rotation 1 pt1', 'rotation 1 pt2', 'rotation 2', 'rotation 2 pt1', 'rotation 2 pt2']:
            angle = (end_step - start_step)/6
            add_trends = 'constant'
        else:
            add_trends = False

    steps_df = pd.read_csv(steps_csv)
    if action not in ['start', 'grasp', 'grasp pt1', 'grasp pt2', 'lift']:
        ibbox = extract_floats_from_string(steps_df.loc[steps_df['step'] == 50].iloc[0, 3]) # initial bounding box
        # find length of initial bounding box from centre to outer xy corner
        ibbline = ((ibbox[1] - ibbox[0])**2 + (ibbox[3] - ibbox[2])**2)/2
        bbox = (extract_floats_from_string(steps_df.loc[steps_df['step'] == exp_bbox_idx].iloc[0, 3]))
        # print(f"Bounding box for action {action} at step {exp_bbox_idx}: {bbox}")
        bbox_center = np.mean(np.array(bbox).reshape(3, 2), axis=1)
        bbox_center_top = bbox_center + np.array([0, 0, (bbox[5]-bbox[4])/2])  # Adjusting the center to the top of the bounding box
        arm_center = steps_df.loc[steps_df['step'] == exp_bbox_idx].iloc[0, 2]
        arm_center = eval(arm_center.replace("tensor", ""))
        arm_center -= np.array([0, 0, 0.07])  # Adjusting the hand center to match the finger center
        # print(f"Arm center: {arm_center}, Bounding box center: {bbox_center}")
        xy_distance = np.linalg.norm(bbox_center_top[:2] - arm_center[:2])
        z_distance = bbox_center_top[2] - arm_center[2] # Only considering the z-coordinate for distance
        # print(f"Action: {action} at step {start_step}| Distance between arm center and bounding box center top: {distance}")
        if bbox[4] <= 0.01: # If min z value is less than or equal to 0.01, then the object is not picked up
            dropped = 'dropped'
            # print("failed at bbox min z value")
        elif z_distance > 0.01:
            dropped = 'dropped'
            # print("failed at z distance")
        # elif xy_distance > ibbline:
        #     dropped = 'dropped'
        #     print("failed at xy distance")
        # print(f"Action: {action} at step {start_step}| Distance = {xy_distance} | bbox center top: {bbox_center_top} | arm center: {arm_center} | ibbox: {ibbox} - {dropped if 'dropped' in locals() else 'not dropped'}")
        # print(f"ibbline: {ibbline} | xy distance: {xy_distance} |  bbox: {bbox} |")
        # print(f"Action: {action} at step {start_step}| Distance = {distance} | bbox center top: {bbox_center_top} | arm center: {arm_center} | bbox: {bbox} - {dropped if 'dropped' in locals() else 'not dropped'}")
        # print(f"Bounding box for action {action} at step {exp_bbox_idx}: {bbox} - {dropped if 'dropped' in locals() else 'not dropped'}")

    # bbox_center = np.mean(np.array(bbox).reshape(3, 2), axis=1)
    # return (action, deformation_level, force_level)
    return labeler.generate_sentence(action, deformation_level, force_level, stability = None, add_trend = add_trends, angle=angle, dropped=dropped), action, dropped

def get_picked_up_objects(all_objects, material='Elastic'):
    to_do = []
    for obj_name in all_objects:
        for target in ['hard', 'medium', 'soft']:
            picked_up_path = os.path.join(BASE_PATH, 'main', 'data', 'picked_up_4', 'csv', obj_name)
            test_path = os.path.join(picked_up_path, material, target)
            # if csv_path is empty, then do not include this object
            if not os.path.exists(test_path) or not os.listdir(test_path):
                print(f'❌ {obj_name}, {target} is not picked up.')
                continue
            else:
                to_do.append((obj_name, target))
                print(f'✅ {obj_name}, {target} is picked up.')
    return to_do


def main(obj_name, picked_up_path, deformation, material='Elastic'):
    #------------------ Set up dataframe ------------------#
    annotations_df = pd.DataFrame(columns=['action','step start', 'step end', 'annotation'])
    #------------------ Choose an object and deformation level ------------------#
    name = obj_name
    csv_path = picked_up_path
    drop_logic = None

    # #------------------- Check if paths contain files -------------------#
    # # If the picked up path does not exist or is empty
    # if not os.path.isdir(picked_up_path) or not os.listdir(picked_up_path):
    #     # If the not picked up path does not exist or is empty, then the object is invalid
    #     if not os.listdir(not_picked_up_path):
    #         raise ValueError(f"Invalid object: {object}. No data available for the specified material and deformation level.")
    #     else:
    #         # If the not picked up path exists and has files, use it instead
    #         csv_path = not_picked_up_path
    #     print(f"not picked up path: {object}")
    #     exit()
    # else:
    #     csv_path = picked_up_path

    #------------------- Load the CSV files -------------------#
    # deform_csv: step, deformations, grip_force
    deform_csv = os.path.join(csv_path, f"{obj_name}_{material}_deform_{deformation}.csv")
    # steps_csv: action, step, hand_coordinate, bounding_box [need to convert from np and pd]
    steps_csv = os.path.join(csv_path, f"{obj_name}_{material}_steps_{deformation}.csv")
    # force_csv: step, left_fx, left_fy, left_fz, left_tx, left_ty, left_tz, right_fx, right_fy, right_fz, right_tx, right_ty, right_tz, dof_0, dof_1, dof_2, dof_3, dof_4, dof_5, dof_6, dof_7, dof_8
    force_csv = os.path.join(csv_path, f"{obj_name}_{material}_{deformation}.csv")

    #------------------- Load the dataframes -------------------#
    # deform_df = pd.read_csv(deform_csv)
    steps_df = pd.read_csv(steps_csv)
    # force_df = pd.read_csv(force_csv)

    pairings, insertions, exp_bbox = split_for_model(steps_df)
    # print(f"Pairings: {pairings}, Insertions: {insertions}, Expanded BBox: {exp_bbox}")
    for i, (start, end) in enumerate(pairings):
        annotation, action, drop_logic = logical(i, deform_csv, force_csv, steps_csv, deformation, start, end, insertions, exp_bbox[i], drop_logic)
        annotations_df.loc[len(annotations_df)] = {'action': action, 'step start': start, 'step end': end, 'annotation': annotation}


    # ------------------- Save the annotations to a CSV file -------------------#
    os.makedirs(os.path.join(BASE_PATH, 'main', 'data', 'picked_up_4', 'annotations'), exist_ok=True)
    output_csv_path = os.path.join(BASE_PATH, 'main', 'data', 'picked_up_4', 'annotations', f"{obj_name}_{material}_{deformation}_annotations.csv")
    annotations_df.to_csv(output_csv_path, index=False)

labeler = RobotLabelTemplate()

if __name__ == "__main__":
    folder_path = os.path.join(BASE_PATH, "main", "data", "picked_up_4", "csv")
    all_objects = os.listdir(folder_path)
    selected_objects = get_picked_up_objects(all_objects)
    # selected_objects = [('Crayola_Bonus_64_Crayons', 'medium')]
    
    material = 'Elastic'
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