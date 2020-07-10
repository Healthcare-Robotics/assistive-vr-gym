from .drinking import DrinkingEnv

class DrinkingPR2Env(DrinkingEnv):
    def __init__(self):
        super(DrinkingPR2Env, self).__init__(robot_type='pr2', human_control=False)

class DrinkingJacoEnv(DrinkingEnv):
    def __init__(self):
       super(DrinkingJacoEnv, self).__init__(robot_type='jaco', human_control=False)

class DrinkingPR2HumanEnv(DrinkingEnv):
    def __init__(self):
        super(DrinkingPR2HumanEnv, self).__init__(robot_type='pr2', human_control=True)

class DrinkingJacoHumanEnv(DrinkingEnv):
    def __init__(self):
        super(DrinkingJacoHumanEnv, self).__init__(robot_type='jaco', human_control=True)

class DrinkingPR2NewEnv(DrinkingEnv):
    def __init__(self):
        super(DrinkingPR2NewEnv, self).__init__(robot_type='pr2', human_control=False, new=True)

class DrinkingJacoNewEnv(DrinkingEnv):
    def __init__(self):
        super(DrinkingJacoNewEnv, self).__init__(robot_type='jaco', human_control=False, new=True)

# VR
class DrinkingVRPR2Env(DrinkingEnv):
    def __init__(self):
        super(DrinkingVRPR2Env, self).__init__(robot_type='pr2', human_control=False, vr=True, new=False)

class DrinkingVRJacoEnv(DrinkingEnv):
    def __init__(self):
        super(DrinkingVRJacoEnv, self).__init__(robot_type='jaco', human_control=False, vr=True, new=False)

class DrinkingVRPR2HumanEnv(DrinkingEnv):
    def __init__(self):
        super(DrinkingVRPR2HumanEnv, self).__init__(robot_type='pr2', human_control=True, vr=True, new=False)

class DrinkingVRJacoHumanEnv(DrinkingEnv):
    def __init__(self):
        super(DrinkingVRJacoHumanEnv, self).__init__(robot_type='jaco', human_control=True, vr=True, new=False)

class DrinkingVRPR2NewEnv(DrinkingEnv):
    def __init__(self):
        super(DrinkingVRPR2NewEnv, self).__init__(robot_type='pr2', human_control=False, vr=True, new=True)

class DrinkingVRJacoNewEnv(DrinkingEnv):
    def __init__(self):
        super(DrinkingVRJacoNewEnv, self).__init__(robot_type='jaco', human_control=False, vr=True, new=True)