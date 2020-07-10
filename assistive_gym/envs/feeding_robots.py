from .feeding import FeedingEnv

class FeedingPR2Env(FeedingEnv):
    def __init__(self):
        super(FeedingPR2Env, self).__init__(robot_type='pr2', human_control=False)

class FeedingJacoEnv(FeedingEnv):
    def __init__(self):
        super(FeedingJacoEnv, self).__init__(robot_type='jaco', human_control=False)

class FeedingPR2HumanEnv(FeedingEnv):
    def __init__(self):
        super(FeedingPR2HumanEnv, self).__init__(robot_type='pr2', human_control=True)

class FeedingJacoHumanEnv(FeedingEnv):
    def __init__(self):
        super(FeedingJacoHumanEnv, self).__init__(robot_type='jaco', human_control=True)

class FeedingPR2NewEnv(FeedingEnv):
    def __init__(self):
        super(FeedingPR2NewEnv, self).__init__(robot_type='pr2', human_control=False, new=True)

class FeedingJacoNewEnv(FeedingEnv):
    def __init__(self):
        super(FeedingJacoNewEnv, self).__init__(robot_type='jaco', human_control=False, new=True)

# VR
class FeedingVRPR2Env(FeedingEnv):
    def __init__(self):
        super(FeedingVRPR2Env, self).__init__(robot_type='pr2', human_control=False, vr=True, new=False)

class FeedingVRJacoEnv(FeedingEnv):
    def __init__(self):
        super(FeedingVRJacoEnv, self).__init__(robot_type='jaco', human_control=False, vr=True, new=False)

class FeedingVRPR2HumanEnv(FeedingEnv):
    def __init__(self):
        super(FeedingVRPR2HumanEnv, self).__init__(robot_type='pr2', human_control=True, vr=True, new=False)

class FeedingVRJacoHumanEnv(FeedingEnv):
    def __init__(self):
        super(FeedingVRJacoHumanEnv, self).__init__(robot_type='jaco', human_control=True, vr=True, new=False)

class FeedingVRPR2NewEnv(FeedingEnv):
    def __init__(self):
        super(FeedingVRPR2NewEnv, self).__init__(robot_type='pr2', human_control=False, vr=True, new=True)

class FeedingVRJacoNewEnv(FeedingEnv):
    def __init__(self):
        super(FeedingVRJacoNewEnv, self).__init__(robot_type='jaco', human_control=False, vr=True, new=True)