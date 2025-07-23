from annotation_bank import RobotLabelTemplate
import os
import pandas as pd
import re
import numpy as np

labeler = RobotLabelTemplate()

#------------------ Set up dataframe ------------------#
annotations_df = pd.DataFrame(columns=['action','step start', 'step end', 'annotation'])

#------------------ Choose an object and deformation level ------------------#
object = '3D_Dollhouse_Sofa'
material = 'Elastic'
deformation = 'hard'  # Options: 'soft', 'medium', 'hard'

#------------------ Set up paths of csv files ------------------#
base_path = '/Users/no.166/Documents/Azka\'s Workspace/Genesis'
picked_up_path = os.path.join(base_path, 'main', 'data', 'picked_up_3', 'csv', object, material, deformation)
not_picked_up_path = os.path.join(base_path, 'main', 'data', 'not_picked_up_3', 'csv', object, material, deformation)

#------------------- Check if paths contain files -------------------#
# If the picked up path does not exist or is empty
if not os.path.isdir(picked_up_path) or not os.listdir(picked_up_path):
    # If the not picked up path does not exist or is empty, then the object is invalid
    if not os.listdir(not_picked_up_path):
        raise ValueError(f"Invalid object: {object}. No data available for the specified material and deformation level.")
    else:
        # If the not picked up path exists and has files, use it instead
        csv_path = not_picked_up_path
    print(f"not picked up path: {object}")
    exit()
else:
    csv_path = picked_up_path

#------------------- Load the CSV files -------------------#
# deform_csv: step, deformations, grip_force
deform_csv = os.path.join(csv_path, f"{object}_{material}_deform_{deformation}.csv")
# steps_csv: action, step, hand_coordinate, bounding_box [need to convert from np and pd]
steps_csv = os.path.join(csv_path, f"{object}_{material}_steps_{deformation}.csv")
# force_csv: step, left_fx, left_fy, left_fz, left_tx, left_ty, left_tz, right_fx, right_fy, right_fz, right_tx, right_ty, right_tz, dof_0, dof_1, dof_2, dof_3, dof_4, dof_5, dof_6, dof_7, dof_8
force_csv = os.path.join(csv_path, f"{object}_{material}_{deformation}.csv")

def convert(df):
    out = []

    for step, i in zip(df['hand_coordinate'], df['object bounding box']):
        stripped = i[8:-2]
        
        x, y, z = stripped.split(',')

        out.append([float(x), float(y), float(z)])
    return out


#------------------- Load the dataframes -------------------#
deform_df = pd.read_csv(deform_csv)
steps_df = pd.read_csv(steps_csv)
force_df = pd.read_csv(force_csv)

#------------------- Generate labels -------------------#
# annotations: ['step start', 'step end', 'annotation']
# annotation = labeler.generate_sentence(
#     action='lift',
#     deformation_level='moderate',
#     force_level='medium',
#     stability='stable',
#     add_trend='increasing'
# )

# def extract_floats_from_string(data_string: str) -> list[float]:

#     string_numbers = re.findall(r'\d+\.\d+', data_string)

#     # A list comprehension is used to convert each number string into a float.
#     float_numbers = [float(num) for num in string_numbers]

#     return float_numbers

def split_for_model(df):
    unique_array = np.sort(df.iloc[:, 1].unique())
    # print(f"Unique steps: {unique_array}")
    expanded = []
    to_return = []
    max_jump = 270
    inserted_points = []
    for i in range(len(unique_array) - 1):
        start = unique_array[i]
        end = unique_array[i + 1]
        expanded.append(start)
        
        jump = end - start
        if jump > max_jump:
            midpoint = (start + end) / 2
            expanded.append(midpoint)
            inserted_points.append((i, midpoint))
            
    expanded.append(unique_array[-1])
    expanded_array = np.round(expanded).astype(int)
    # print(f"Expanded array: {expanded_array}, Inserted points: {inserted_points}")
    # print("\nInserted midpoints:")
    for idx, value in inserted_points:
        to_return.append(idx)
    pairings = [[int(a), int(b)] for a, b in zip(expanded_array[:-1], expanded_array[1:])]
    return pairings, to_return



#-------------------- Create annotations based on steps -------------------#
def logical(i, deform_csv, force_csv, steps_csv, deformation, start_step, end_step, insertions):

    actions = ['start', 'grasp', 'lift', 'rotation 1', 'buffer 1', 'rotation 2', 'buffer 2', 'wind down']
    if 1 in insertions:
        actions.pop(1)
        actions.insert(1, 'grasp pt1')
        actions.insert(2, 'grasp pt2')
        if 3 in insertions:
            actions.pop(4)
            actions.insert(4, 'rotation 1 pt1')
            actions.insert(5, 'rotation 1 pt2')
            if 5 in insertions:
                actions.pop(7)
                actions.insert(7, 'rotation 2 pt1')
                actions.insert(8, 'rotation 2 pt2')
        elif 5 in insertions:
            actions.pop(6)
            actions.insert(6, 'rotation 2 pt1')
            actions.insert(7, 'rotation 2 pt2')
    action = actions[i]


    # deformation level: ['none', 'soft', 'medium', 'hard']
    if action in ['start', 'wind down']:
        deformation_level = 'none'
        force_level = 'none'
    elif action in ['grasp', 'grasp pt1', 'grasp pt2' 'lift', 'rotation 1', 'rotation 1 pt1', 'rotation 1 pt2', 'buffer 1', 'rotation 2', 'rotation 2 pt1', 'rotation 2 pt2', 'buffer 2']:
        deformation_level = deformation
        if deformation == 'soft':
            force_level = 'low'
        elif deformation == 'medium':
            force_level = 'medium'
        elif deformation == 'hard':
            force_level = 'high'

    # bbox = (extract_floats_from_string(steps_df.iloc[3, i]))
    # bbox_center = np.mean(np.array(bbox).reshape(3, 2), axis=1)
    # return (action, deformation_level, force_level)
    return labeler.generate_sentence(action='start', deformation_level='soft', force_level='low',
                                     stability='stable',
                                     add_trend='increasing'), action

pairings, insertions = split_for_model(steps_df)
for i, (start, end) in enumerate(pairings):
    annotation, action = logical(i, deform_csv, force_csv, steps_csv, deformation, start, end, insertions)
    annotations_df.loc[len(annotations_df)] = {'action': action, 'step start': start, 'step end': end, 'annotation': annotation}

# annotations_df.loc[len(annotations_df)] = {'step start': steps_df.iloc[1,1], 'step end': steps_df.iloc[2,1], 'annotation':annotation} # Assuming the first step is the start

# ------------------- Save the annotations to a CSV file -------------------#
os.makedirs(os.path.join(base_path, 'main', 'data', 'picked_up_3', 'annotations'), exist_ok=True)
output_csv_path = os.path.join(base_path, 'main', 'data', 'picked_up_3', 'annotations', f"{object}_{material}_{deformation}_annotations.csv")
annotations_df.to_csv(output_csv_path, index=False)
