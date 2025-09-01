import random

import numpy as np
import pandas as pd

SEQUENCE_LENGTH = 80

class RobotLabelTemplate:
    def __init__(self):

        self.force_descriptors = {
            'none': 'zero-force',
            'low': 'gentle force',
            'medium': 'moderate force',
            'high': 'strong force'

        }

        self.stability_descriptors = {
            'stable': ['stable grasp'],
            'unstable': ['unstable grasp']
        }

        self.add_trends = {
            'increasing': 'increasing force',
            'decreasing':'decreasing force',
            'constant': 'constant force',
            'deformation': 'progressive deformation'
        }

        self.object_refs = 'the object'
        self.transitions = ['while', 'as', 'during', 'throughout', 'simultaneously', 'then', 'followed by']

        self.dropped = {
            'dropped': ['dropped']
        }

    def dist_from_COM(self, com: np.ndarray, pos: np.ndarray, contact_range: np.ndarray) -> str:
        """
        Calculate the distance between the grasp position and the center of mass (COM), and return whether it is far or near.
        Considers only the first timestep of the timeseries.
        Args:
            com [t, 3]: A time series of the center of mass position.
            pos [t, 3]: A time series of the grasp position.
        TODO:
            This code means that it annotates 'far' when the distance is greater than 0.1[meters]. 
            We need to check whether this is a good threshold value.
            Maybe we need to change this to be relative to the size of the object being grasped.
            Also, the [x, y] distance may be more important than the [z] distance, because the gravity is acting downwards.
        """
        com = com[contact_range]
        pos = pos[contact_range]
        distance = np.linalg.norm((com - pos)[0]) # TODO: change to average
        return "far from" if distance > 0.15 else "near"

    def slip_detection(self, grasp_pos: np.ndarray, com: np.ndarray, contact_range: np.ndarray) -> str:
        """
        Detect whether a slip has occurred based on the distance between the grasp position and the center of mass (COM).
        Args:
            com [t, 3]: A time series of the center of mass position.
            grasp_pos [t, 3]: A time series of the grasp position.
        TODO:
            First of all we need to check if the code works correctly.
            The code now uses the diff of the distances to determine slip velocity.
            Since each timestep is 0.01 seconds for rigid objects, if the distance changes by more than 0.0005 meters in 1 timestep, the slip velocity is 5cm/s.
            Is this a good threshold to separate 'slipping quickly' from 'slipping slowly'?
            Remember that VLAs can re-generate action chunks once in 0.8s, and the size of the Franka finger.
            Maybe it's better to use bounding box information rather than the COM position.
        """
        grasp_pos = grasp_pos[contact_range]
        com = com[contact_range]
        distances = np.linalg.norm(grasp_pos - com, axis=1)
        # Decide the slip velocity from the time series of distances
        slip_velocities = np.diff(distances)
        return "letting it slip quickly" if np.any(slip_velocities > 0.0005) else "letting it slip slowly" if np.any(slip_velocities > 0.0001) else "keeping it stable" 
    
    def torque_annotation(self, force_df, contact_range):
        lf = force_df[['left_fx', 'left_fy', 'left_fz']].to_numpy()[contact_range]
        lp = force_df[['left_finger_x', 'left_finger_y', 'left_finger_z']].to_numpy()[contact_range]
        rf = force_df[['right_fx', 'right_fy', 'right_fz']].to_numpy()[contact_range]
        rp = force_df[['right_finger_x', 'right_finger_y', 'right_finger_z']].to_numpy()[contact_range]
        com = force_df[['obj_COM_x', 'obj_COM_y', 'obj_COM_z']].to_numpy()[contact_range]

        tau = np.linalg.norm(np.cross(lp - com, lf) + np.cross(rp - com, rf), axis=1)

        if np.any(tau >= 1.0):
            return "high"
        elif np.any(tau >= 0.1):
            return "moderate"
        else:
            return "none"

    

    def generate_sentence(self, action: str, force_df: pd.DataFrame) -> str:
        """
        Generate a sentence using selected values.
        """

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
                if force_df['obj_min_z'].values[i] > 0.03:
                    if "place" in action:
                        action = action.replace("place", "drop")
                    else:
                        action = "dropping when trying to " + action
                released_idx = i
                break
        

        if not touched_both:
            if touched_either:
                return "touched an object."
            else:
                return "moving with empty hands."

        contact_range = np.array([False] * len(force_df))
        contact_range[touched_idx:released_idx+1] = True
        annotation = ""

        mass = force_df['obj_mass'].values[0]
        mass_str = "heavy" if mass > 0.5 else "light" #TODO: Check whether 0.5 [kg] is a good threshold

        annotation += f"{action} a {mass_str} object " # explain the movement very simply

        com_pos = force_df[['obj_COM_x', 'obj_COM_y', 'obj_COM_z']].to_numpy()
        right_finger_pos = force_df[['right_finger_x', 'right_finger_y', 'right_finger_z']].to_numpy()
        left_finger_pos = force_df[['left_finger_x', 'left_finger_y', 'left_finger_z']].to_numpy()
        grasp_pos = (right_finger_pos + left_finger_pos)/2
        annotation += f"{self.dist_from_COM(com_pos, grasp_pos, contact_range)} the center of mass, "

        annotation += f"{self.slip_detection(grasp_pos, com_pos, contact_range)}"
        annotation += f" under {self.torque_annotation(force_df, contact_range)} torque stress"

        return annotation
