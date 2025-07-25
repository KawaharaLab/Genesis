import random

class RobotLabelTemplate:
    def __init__(self):
        self.actions = {
            'start': ['starting movement', 'initiating', 'advancing', 'aligning', 'reaching', 'positioning', 'preparing', 'approaching'],
            
            'grasp': ['grasping', 'gripping', 'holding', 'securing', 'clasping', 'clamping', 'capturing', 'pinching', 'squeezing'],
            'grasp pt1': ['grasping', 'gripping', 'holding', 'securing', 'clasping', 'clamping', 'capturing', 'pinching', 'squeezing'],
            'grasp pt2': ['grasping', 'gripping', 'holding', 'securing', 'clasping', 'clamping', 'capturing', 'pinching', 'squeezing'],
            
            'lift': ['lifting', 'raising', 'elevating', 'picking up'],
            
            'rotation 1': ['rotating', 'turning', 'twisting'],
            'rotation 1 pt1': ['rotating', 'turning', 'twisting'],
            'rotation 1 pt2': ['rotating', 'turning', 'twisting'],

            'buffer 1':   ['holding position of', 'maintaining hold of', 'pausing motion of', 'steadying'],
            'buffer 1 pt1': ['holding position of', 'maintaining hold of', 'pausing motion of', 'steadying'],
            'buffer 1 pt2': ['holding position of', 'maintaining hold of', 'pausing motion of', 'steadying'],

            'rotation 2': ['rotating', 'turning', 'twisting'],
            'rotation 2 pt1': ['rotating', 'turning', 'twisting'],
            'rotation 2 pt2': ['rotating', 'turning', 'twisting'],

            'buffer 2':   ['holding position of', 'maintaining hold of', 'pausing motion of', 'steadying'],
            'buffer 2 pt1': ['holding position of', 'maintaining hold of', 'pausing motion of', 'steadying'],
            'buffer 2 pt2': ['holding position of', 'maintaining hold of', 'pausing motion of', 'steadying'],

            'wind_down':  ['slowing down motion of', 'decelerating motion of', 'coming to rest with']
        }

        self.deformation_levels = {
            'none': ['undeformed shape', 'maintaining shape', 'contactless', 'form-conservation',
                     'shape-conservation', 'unyielding'],
            'soft': ['soft-deformation', 'gentle-compression', 'lightly-yielding', 'minimal-resistance',
                     'slight-yielding', 'cushioned-contact'],
            'medium': ['medium-deformation', 'moderate-compression', 'controlled-yielding', 'steady-resistance',
                       'balanced-yielding', 'firm-contact'],
            'hard': ['hard-deformation', 'significant-compression', 'firm-yielding', 'strong-resistance',
                     'substantial-yielding', 'rigid-contact'],
        }

        self.force_descriptors = {
            'none': ['forceless methods', 'zero-pressure', 'zero-force', 'contactless method'],
            'low': ['gentle force', 'light pressure', 'minimal force', 'soft contact',
                    'cushioned pressure', 'slight force', 'subtle force'],
            'medium': ['moderate force', 'controlled pressure', 'steady force', 'balanced pressure',
                       'firm contact', 'moderate force'],
            'high': ['strong force', 'firm pressure', 'significant force', 'intense pressure',
                     'heavy force', 'strong contact'],
        }

        self.stability_descriptors = {
            'stable': ['stable grasp', 'secure hold', 'firm grip', 'controlled grasp', 'steady contact'],
            'unstable': ['unstable grasp', 'loose grip', 'slipping contact', 'precarious hold', 'uncertain grip']
        }

        self.add_trends = {
            'increasing': ['increasing force', 'growing pressure', 'rising force'],
            'decreasing': ['decreasing force', 'reducing pressure', 'diminishing force', 'lessening pressure'],
            'constant': ['constant force', 'steady pressure', 'unchanging force'],
            'deformation': ['progressive deformation', 'gradual yielding', 'increasing compression']
        }

        self.positional_terms = ['end-effector', 'gripper', 'fingers', 'gripping mechanism']
        self.object_refs = ['the object', 'target object', 'item']
        self.transitions = ['while', 'as', 'during', 'throughout', 'simultaneously', 'then', 'followed by']

        self.droppped = {
            'dropped': ['dropped', 'released', 'let go of'],
        }

    def generate_sentence(self, action: str, deformation_level: str = None,
                          force_level: str = None, stability: str = None,
                          add_trend: str = None, angle: int = None, dropped: str = None) -> str:
        """
        Generate a sentence using selected values.
        """
        parts = []

        effector = random.choice(self.positional_terms)
        object_ref = random.choice(self.object_refs)
        action_phrase = random.choice(self.actions.get(action))
        if action in ['grasp pt2', 'rotation 1 pt2', 'rotation 2 pt2', 'buffer 1 pt2', 'buffer 2 pt2']:
            base = f"{effector} continue {action_phrase} {object_ref}"
        else:
            base = f"{effector} {action_phrase} {object_ref}"
        parts.append(base)

        if action == 'start':
            sentence = f"{effector} {action_phrase} towards {object_ref}, to begin the task."
            return sentence[0].upper() + sentence[1:]
        
        if angle:
            angle_phrase = f"by an angle of {angle} degrees"
            parts.append(angle_phrase)
        
        if action in ['buffer 1', 'buffer 1 pt1', 'buffer 1 pt2', 'buffer 2', 'buffer 2 pt1', 'buffer 2 pt2']:
            parts.append("maintaining a stable hold")

        if force_level:
            force_phrase = random.choice(self.force_descriptors.get(force_level, []))
            parts.append(f"using {force_phrase}")

        if deformation_level:
            deform_phrase = random.choice(self.deformation_levels.get(deformation_level, []))
            parts.append(f"causing {deform_phrase}")

        if stability:
            stability_phrase = random.choice(self.stability_descriptors.get(stability, []))
            parts.append(f"maintaining {stability_phrase}")

        if add_trend:
            add_trend_phrase = random.choice(self.add_trends.get(add_trend, []))
            parts.append(f"with {add_trend_phrase}")

        if dropped:
            sentence = f"{object_ref} has been {random.choice(self.droppped.get(dropped, []))}."
            return sentence[0].upper() + sentence[1:]
            

        # Join sentence with transitions or commas
        if len(parts) == 1:
            sentence = parts[0]
        elif len(parts) == 2:
            connector = random.choice([' ', ', ', ' while '])
            sentence = connector.join(parts)
        else:
            sentence = parts[0] + ", " + ", ".join(parts[1:])

        sentence = sentence.strip().rstrip(',') # remove trailing comma
        if not sentence.endswith('.'):
            sentence += '.'

        return sentence[0].upper() + sentence[1:] 






