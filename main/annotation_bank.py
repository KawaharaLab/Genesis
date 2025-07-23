import random

class RobotLabelTemplate:
    def __init__(self):
        self.actions = {
            'start': ['starting', 'initiating', 'beginning', 'commencing', 'launching'],
            
            'grasp': ['grasping', 'gripping', 'holding', 'securing', 'clasping'],
            'grasp pt1': ['grasping', 'gripping', 'holding', 'securing', 'clasping'],
            'grasp pt2': ['grasping', 'gripping', 'holding', 'securing', 'clasping'],
            
            'lift': ['lifting', 'raising', 'elevating', 'picking up'],
            
            'rotation 1': ['rotating', 'turning', 'pivoting', 'swiveling', 'twisting to finish'],
            'rotation 1 pt1': ['rotating', 'turning', 'pivoting', 'swiveling', 'twisting to finish'],
            'rotation 1 pt2': ['rotating', 'turning', 'pivoting', 'swiveling', 'twisting to finish'],

            'buffer 1':   ['holding position', 'maintaining hold', 'stabilizing grip', 'pausing motion', 'steadying'],
            'buffer 1 pt1': ['holding position', 'maintaining hold', 'stabilizing grip', 'pausing motion', 'steadying'],
            'buffer 1 pt2': ['holding position', 'maintaining hold', 'stabilizing grip', 'pausing motion', 'steadying'],

            'rotation 2': ['rotating', 'turning', 'pivoting', 'swiveling', 'twisting to finish'],
            'rotation 2 pt1': ['rotating', 'turning', 'pivoting', 'swiveling', 'twisting to finish'],
            'rotation 2 pt2': ['rotating¥', 'turning', 'pivoting', 'swiveling', 'twisting to finish'],

            'buffer 2':   ['holding position', 'maintaining hold', 'stabilizing grip', 'pausing motion', 'steadying'],
            'buffer 2 pt1': ['holding position', 'maintaining hold', 'stabilizing grip', 'pausing motion', 'steadying'],
            'buffer 2 pt2': ['holding position', 'maintaining hold', 'stabilizing grip', 'pausing motion', 'steadying'],

            'wind down':  ['slowing down', 'decelerating motion', 'winding down', 'easing off', 'coming to rest']
        }

        self.deformation_levels = {
            'none': ['no deformation', 'maintaining shape', 'rigid contact', 'no compression'],
            'slight': ['minimal deformation', 'barely compressing', 'light contact', 'subtle compression'],
            'moderate': ['moderate deformation', 'visible compression', 'controlled pressure', 'steady compression'],
            'heavy': ['significant deformation', 'substantial compression', 'strong pressure', 'deep compression'],
            'extreme': ['extreme deformation', 'maximum compression', 'intense pressure', 'severe compression'],
            'soft': ['soft deformation', 'gentle compression', 'lightly yielding', 'minimal resistance'],
            'medium': ['medium deformation', 'moderate compression', 'controlled yielding', 'steady resistance'],
            'hard': ['hard deformation', 'significant compression', 'firm yielding', 'strong resistance']
        }

        self.force_descriptors = {
            'none': ['no force', 'no pressure', 'zero force', 'no contact'],
            'low': ['gentle force', 'light pressure', 'minimal force', 'soft contact'],
            'medium': ['moderate force', 'controlled pressure', 'steady force', 'balanced pressure'],
            'high': ['strong force', 'firm pressure', 'significant force', 'intense pressure'],
            'variable': ['varying force', 'adjusting pressure', 'dynamic force', 'changing pressure']
        }

        self.stability_descriptors = {
            'stable': ['stable grasp', 'secure hold', 'firm grip', 'controlled grasp', 'steady contact'],
            'unstable': ['unstable grasp', 'loose grip', 'slipping contact', 'precarious hold', 'uncertain grip']
        }

        self.positional_terms = ['end-effector', 'gripper', 'fingers', 'gripping mechanism']
        self.object_refs = ['the object', 'target object', 'item']
        self.transitions = ['while', 'as', 'during', 'throughout', 'simultaneously', 'then', 'followed by']

    def generate_sentence(self, action: str, deformation_level: str = None,
                          force_level: str = None, stability: str = None,
                          add_trend: str = None) -> str:
        """
        Generate a sentence using selected values.
        """
        parts = []

        effector = random.choice(self.positional_terms)
        object_ref = random.choice(self.object_refs)
        action_phrase = random.choice(self.actions.get(action))
        base = f"{effector} {action_phrase} {object_ref}"
        parts.append(base)

        if deformation_level:
            deform_phrase = random.choice(self.deformation_levels.get(deformation_level, []))
            parts.append(f"with {deform_phrase}")

        if force_level:
            force_phrase = random.choice(self.force_descriptors.get(force_level, []))
            parts.append(f"using {force_phrase}")

        if stability:
            stability_phrase = random.choice(self.stability_descriptors.get(stability, []))
            parts.append(f"maintaining {stability_phrase}")

        if add_trend:
            if add_trend == 'increasing':
                parts.append("with increasing force")
            elif add_trend == 'decreasing':
                parts.append("while reducing pressure")
            elif add_trend == 'deformation':
                parts.append("causing progressive deformation")

        # Join sentence with transitions or commas
        if len(parts) == 1:
            sentence = parts[0]
        elif len(parts) == 2:
            connector = random.choice([' ', ', ', ' while '])
            sentence = connector.join(parts)
        else:
            sentence = parts[0] + ", " + ", ".join(parts[1:])

        sentence = sentence.strip().rstrip(',')
        if not sentence.endswith('.'):
            sentence += '.'

        return sentence[0].upper() + sentence[1:]






