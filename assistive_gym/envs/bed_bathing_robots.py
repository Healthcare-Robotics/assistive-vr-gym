from .bed_bathing import BedBathingEnv

class BedBathingPR2Env(BedBathingEnv):
    def __init__(self):
        super(BedBathingPR2Env, self).__init__(robot_type='pr2', human_control=False)

class BedBathingJacoEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingJacoEnv, self).__init__(robot_type='jaco', human_control=False)

class BedBathingPR2HumanEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingPR2HumanEnv, self).__init__(robot_type='pr2', human_control=True)

class BedBathingJacoHumanEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingJacoHumanEnv, self).__init__(robot_type='jaco', human_control=True)

class BedBathingPR2NewEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingPR2NewEnv, self).__init__(robot_type='pr2', human_control=False, new=True)

class BedBathingJacoNewEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingJacoNewEnv, self).__init__(robot_type='jaco', human_control=False, new=True)

# VR
class BedBathingVRPR2Env(BedBathingEnv):
    def __init__(self):
        super(BedBathingVRPR2Env, self).__init__(robot_type='pr2', human_control=False, vr=True, new=False)

class BedBathingVRJacoEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingVRJacoEnv, self).__init__(robot_type='jaco', human_control=False, vr=True, new=False)

class BedBathingVRPR2HumanEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingVRPR2HumanEnv, self).__init__(robot_type='pr2', human_control=True, vr=True, new=False)

class BedBathingVRJacoHumanEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingVRJacoHumanEnv, self).__init__(robot_type='jaco', human_control=True, vr=True, new=False)

class BedBathingVRPR2NewEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingVRPR2NewEnv, self).__init__(robot_type='pr2', human_control=False, vr=True, new=True)

class BedBathingVRJacoNewEnv(BedBathingEnv):
    def __init__(self):
        super(BedBathingVRJacoNewEnv, self).__init__(robot_type='jaco', human_control=False, vr=True, new=True)