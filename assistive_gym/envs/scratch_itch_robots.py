from .scratch_itch import ScratchItchEnv

class ScratchItchPR2Env(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchPR2Env, self).__init__(robot_type='pr2', human_control=False)

class ScratchItchJacoEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchJacoEnv, self).__init__(robot_type='jaco', human_control=False)

class ScratchItchPR2HumanEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchPR2HumanEnv, self).__init__(robot_type='pr2', human_control=True)

class ScratchItchJacoHumanEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchJacoHumanEnv, self).__init__(robot_type='jaco', human_control=True)

class ScratchItchPR2NewEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchPR2NewEnv, self).__init__(robot_type='pr2', human_control=False, new=True)

class ScratchItchJacoNewEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchJacoNewEnv, self).__init__(robot_type='jaco', human_control=False, new=True)

# VR
class ScratchItchVRPR2Env(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchVRPR2Env, self).__init__(robot_type='pr2', human_control=False, vr=True, new=False)

class ScratchItchVRJacoEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchVRJacoEnv, self).__init__(robot_type='jaco', human_control=False, vr=True, new=False)

class ScratchItchVRPR2HumanEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchVRPR2HumanEnv, self).__init__(robot_type='pr2', human_control=True, vr=True, new=False)

class ScratchItchVRJacoHumanEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchVRJacoHumanEnv, self).__init__(robot_type='jaco', human_control=True, vr=True, new=False)

class ScratchItchVRPR2NewEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchVRPR2NewEnv, self).__init__(robot_type='pr2', human_control=False, vr=True, new=True)

class ScratchItchVRJacoNewEnv(ScratchItchEnv):
    def __init__(self):
        super(ScratchItchVRJacoNewEnv, self).__init__(robot_type='jaco', human_control=False, vr=True, new=True)