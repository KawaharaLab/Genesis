import random

import numpy as np
import pandas as pd

class RobotLabelTemplate:
    def __init__(self):
        self.actions = {
            'start': 'start', # TODO: make it more simple
            'lift': 'lifting',
            'grasp': 'grasping',
            'grasp pt1': 'grasping',
            'grasp pt2': 'grasping',
            'rotation 1': 'rotating',
            'rotation 1 pt1': 'rotating',
            'rotation 1 pt2': 'rotating',
            'buffer 1':   'holding position of',
            'buffer 1 pt1':   'holding position of',
            'buffer 1 pt2':   'holding position of',
            'buffer 2':   'holding position of',
            'buffer 2 pt1':   'holding position of',
            'buffer 2 pt2':   'holding position of',
            'rotation 2': 'rotating',
            'rotation 2 pt1': 'rotating',
            'rotation 2 pt2': 'rotating',
            'wind_down':  'stopping'
        }

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

    def dist_from_COM(self, com: np.ndarray, pos: np.ndarray) -> str:
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
        distance = np.linalg.norm((com - pos)[0])
        return "far from" if distance > 0.1 else "near"

    def slip_detection(self, grasp_pos: np.ndarray, com: np.ndarray) -> str:
        """
        Detect whether a slip has occurred based on the distance between the grasp position and the center of mass (COM).
        Args:
            com [t, 3]: A time series of the center of mass position.
            grasp_pos [t, 3]: A time series of the grasp position.
        TODO:
            First of all we need to check if the code works correctly.
            The code now uses the diff of the distances to determine slip velocity.
            Since each timestep is 0.01 seconds for rigid objects, if the distance changes by more than 0.005 meters in 1 timestep, the slip velocity is 0.5m/s.
            Is this a good threshold to separate 'slipping quickly' from 'slipping slowly'?
            Remember that VLAs can re-generate action chunks once in 0.8s, and the size of the Franka finger.
            Maybe it's better to use bounding box information rather than the COM position.
        """
        distances = np.linalg.norm(grasp_pos - com, axis=1)
        # Decide the slip velocity from the time series of distances
        slip_velocities = np.diff(distances, prepend=0)
        return "slipping quickly" if np.any(slip_velocities > 0.005) else "slipping slowly" if np.any(slip_velocities > 0.001) else "no slip" # TODO: something's wrong with the calculation?

    def drop_detection(self, bbox: np.ndarray, grasp_pos: np.ndarray) -> str:
        return False
        if bbox[4] <= 0.01: # If min z value is less than or equal to 0.01, then the object is not picked up
            dropped = 'dropped' # TODO: it might have been placed successfully
            # print("failed at bbox min z value")
        elif z_distance > 0.01:
            dropped = 'dropped'

    def generate_sentence(self, action: str, start: int, force_df: pd.DataFrame) -> str:
        """
        Generate a sentence using selected values.
        """
        annotation = ""

        mass = force_df['obj_mass'].values[0]
        mass_str = "heavy" if mass > 1.0 else "light" #TODO: Check whether 1 [kg] is a good threshold

        annotation += f"{action} a {mass_str} object " # explain the movement very simply

        com_pos = force_df[['obj_COM_x', 'obj_COM_y', 'obj_COM_z']].values
        right_finger_pos = force_df[['right_finger_x', 'right_finger_y', 'right_finger_z']].values
        left_finger_pos = force_df[['left_finger_x', 'left_finger_y', 'left_finger_z']].values
        grasp_pos = (right_finger_pos + left_finger_pos)/2
        annotation += f"{self.dist_from_COM(com_pos, grasp_pos)} the center of mass, "

        annotation += f"{self.slip_detection(grasp_pos, com_pos)}."

        if self.drop_detection(com_pos, grasp_pos):
            sentence = f"a {mass_str} object has been {random.choice(self.dropped.get('dropped', []))}." #TODO: Describe what was happening before the drop, and why it dropped.
            return sentence[0].upper() + sentence[1:]
        #TODO: add the case of "successfully placed object"

        return annotation
