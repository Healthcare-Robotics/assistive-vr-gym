import os
import pybullet as p
import numpy as np

# -- Joint Legend --
# 0-2 waist x,y,z
# 3 chest
# 4-6 right_shoulder x,y,z
# 7-9 right_shoulder_socket x,y,z
# 10 right_elbow x
# 11 right_forearm_roll z
# 12-13 right_hand x,y
# 14-16 left_shoulder x,y,z
# 17-19 left_shoulder_socket x,y,z
# 20 left_elbow x
# 21 left_forearm_roll z
# 22-23 left_hand x,y
# 24 neck x
# 25-27 head x,y,z
# 28-30 right_hip x,y,z
# 31 right_knee x
# 32-34 right_ankle x,y,z
# 35-37 left_hip x,y,z
# 38 left_knee x
# 39-41 left_ankle x,y,z

# -- Limb (link) Legend --
# 2 waist
# 3 chest
# 6 right_shoulder
# 9 right_upperarm
# 11 right_forearm
# 13 right_hand
# 16 left_shoulder
# 19 left_upperarm
# 21 left_forearm
# 23 left_hand
# 24 neck
# 27 head
# 30 right_thigh
# 31 right_shin
# 34 right_foot
# 37 left_thigh
# 38 left_shin
# 41 left_foot

class HumanCreationVR:
    def __init__(self, pid=None, np_random=None, cloth=False):
        self.directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets')
        self.cloth = cloth
        self.np_random = np_random
        self.hand_radius = 0.0
        self.elbow_radius = 0.0
        self.shoulder_radius = 0.0
        self.id = pid

    def create_human(self, static=True, limit_scale=1.0, specular_color=[0.1, 0.1, 0.1], gender='random', config=None, hipbone_to_mouth_height=None, visible=0):
        if gender not in ['male', 'female']:
            gender = self.np_random.choice(['male', 'female'])
        if gender == 'male':
            hmhs = hipbone_to_mouth_height / 0.6
        else:
            hmhs = hipbone_to_mouth_height / 0.54
        def create_body(shape=p.GEOM_CAPSULE, radius=0, length=0, position_offset=[0, 0, 0], orientation=[0, 0, 0, 1], visible=1):
            visual_shape = p.createVisualShape(shape, radius=radius, length=length, rgbaColor=[0.8, 0.6, 0.4, visible], specularColor=specular_color, visualFramePosition=position_offset, visualFrameOrientation=orientation, physicsClientId=self.id)
            collision_shape = p.createCollisionShape(shape, radius=radius, height=length, collisionFramePosition=position_offset, collisionFrameOrientation=orientation, physicsClientId=self.id)
            return collision_shape, visual_shape

        joint_c, joint_v = -1, -1
        if gender == 'male':
            c = lambda tag: config(tag, 'human_male')
            m = c('mass') # mass of 50% male in kg
            rs = c('radius_scale')
            hs = c('height_scale')
            hs *= hmhs
            chest_c, chest_v = create_body(shape=p.GEOM_CAPSULE, radius=0.127*rs, length=0.056, orientation=p.getQuaternionFromEuler([0, np.pi/2.0, 0], physicsClientId=self.id))
            right_shoulders_c, right_shoulders_v = create_body(shape=p.GEOM_CAPSULE, radius=0.106*rs, length=0.253/8, position_offset=[-0.253/2.5 + 0.253/16, 0, 0], orientation=p.getQuaternionFromEuler([0, np.pi/2.0, 0], physicsClientId=self.id))
            left_shoulders_c, left_shoulders_v = create_body(shape=p.GEOM_CAPSULE, radius=0.106*rs, length=0.253/8, position_offset=[0.253/2.5 - 0.253/16, 0, 0], orientation=p.getQuaternionFromEuler([0, np.pi/2.0, 0], physicsClientId=self.id))
            neck_c, neck_v = create_body(shape=p.GEOM_CAPSULE, radius=0.06*rs, length=0.124*hs, position_offset=[0, 0, (0.2565 - 0.1415 - 0.025)*hs], visible=visible)
            upperarm_c, upperarm_v = create_body(shape=p.GEOM_CAPSULE, radius=0.043*rs, length=0.279*hs, position_offset=[0, 0, -0.279/2.0*hs])
            forearm_c, forearm_v = create_body(shape=p.GEOM_CAPSULE, radius=0.033*rs, length=0.257*hs, position_offset=[0, 0, -0.257/2.0*hs])
            hand_c, hand_v = create_body(shape=p.GEOM_SPHERE, radius=0.043*rs, length=0, position_offset=[0, 0, -0.043*rs])
            _, upperarm_invisible_v = create_body(shape=p.GEOM_CAPSULE, radius=0.0001*rs, length=0.0001*hs, position_offset=[0, 0, -0.279/2.0*hs], visible=0)
            _, forearm_invisible_v = create_body(shape=p.GEOM_CAPSULE, radius=0.0001*rs, length=0.0001*hs, position_offset=[0, 0, -0.257/2.0*hs], visible=0)
            _, hand_invisible_v = create_body(shape=p.GEOM_SPHERE, radius=0.0001*rs, length=0, position_offset=[0, 0, -0.043*rs], visible=0)
            
            self.hand_radius, self.elbow_radius, self.shoulder_radius = 0.043*rs, 0.043*rs, 0.043*rs
            waist_c, waist_v = create_body(shape=p.GEOM_CAPSULE, radius=0.1205*rs, length=0.049, orientation=p.getQuaternionFromEuler([0, np.pi/2.0, 0], physicsClientId=self.id))
            hips_c, hips_v = create_body(shape=p.GEOM_CAPSULE, radius=0.1335*rs, length=0.094, position_offset=[0, 0, -0.08125*hs], orientation=p.getQuaternionFromEuler([0, np.pi/2.0, 0], physicsClientId=self.id))
            thigh_c, thigh_v = create_body(shape=p.GEOM_CAPSULE, radius=0.08*rs, length=0.424*hs, position_offset=[0, 0, -0.424/2.0*hs])
            shin_c, shin_v = create_body(shape=p.GEOM_CAPSULE, radius=0.05*rs, length=0.403*hs, position_offset=[0, 0, -0.403/2.0*hs])
            foot_c, foot_v = create_body(shape=p.GEOM_CAPSULE, radius=0.05*rs, length=0.215*hs, position_offset=[0, -0.1, -0.025*rs], orientation=p.getQuaternionFromEuler([np.pi/2.0, 0, 0], physicsClientId=self.id))
            elbow_v = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=(0.043+0.033)/2*rs, length=0, rgbaColor=[0.8, 0.6, 0.4, 1], visualFramePosition=[0, 0.01, 0], physicsClientId=self.id)
            elbow_invisible_v = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.0001*rs, length=0, rgbaColor=[0, 0, 0, 0], specularColor=[0, 0, 0], visualFramePosition=[0, 0.01, 0], physicsClientId=self.id)
            if self.cloth:
                # Cloth penetrates the spheres at the end of each capsule, so we create physical spheres at the joints
                invisible_v = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01*rs, length=0, rgbaColor=[0, 0, 0, 0], physicsClientId=self.id)
                shoulder_cloth_c, _ = create_body(shape=p.GEOM_SPHERE, radius=0.043*rs, length=0)
                elbow_cloth_c, _ = create_body(shape=p.GEOM_SPHERE, radius=0.043*rs, length=0)
                wrist_cloth_c, _ = create_body(shape=p.GEOM_SPHERE, radius=0.033*rs, length=0)

            head_scale = [0.89]*3
            head_pos = [0.09, 0.08, -0.07 + 0.01]
            head_c = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=os.path.join(self.directory, 'head_female_male', 'BaseHeadMeshes_v5_male_cropped_reduced_compressed_vhacd.obj'), collisionFramePosition=head_pos, collisionFrameOrientation=p.getQuaternionFromEuler([np.pi/2.0, 0, 0], physicsClientId=self.id), meshScale=head_scale, physicsClientId=self.id)
            head_v = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=os.path.join(self.directory, 'head_female_male', 'BaseHeadMeshes_v5_male_cropped_reduced_compressed.obj'), rgbaColor=[0.8, 0.6, 0.4, visible], specularColor=specular_color, visualFramePosition=head_pos, visualFrameOrientation=p.getQuaternionFromEuler([np.pi/2.0, 0, 0], physicsClientId=self.id), meshScale=head_scale, physicsClientId=self.id)

            joint_p, joint_o = [0, 0, 0], [0, 0, 0, 1]
            chest_p = [0, 0, 0.156*hs]
            shoulders_p = [0, 0, 0.1415/2*hs]
            neck_p = [0, 0, 0.1515*hs]
            head_p = [0, 0, (0.399 - 0.1415 - 0.1205)*hs]
            right_upperarm_p = [-0.106*rs - 0.073, 0, 0]
            left_upperarm_p = [0.106*rs + 0.073, 0, 0]
            forearm_p = [0, 0, -0.279*hs]
            hand_p = [0, 0, -(0.033*rs + 0.257*hs)]
            waist_p = [0, 0, 0.08125*hs]
            hips_p = [0, 0, 1.00825*hs]
            right_thigh_p = [-0.08*rs - 0.009, 0, -0.08125*hs]
            left_thigh_p = [0.08*rs + 0.009, 0, -0.08125*hs]
            shin_p = [0, 0, -0.424*hs]
            foot_p = [0, 0, -0.403*hs - 0.025]
        else:
            c = lambda tag: config(tag, 'human_female')
            m = c('mass') # mass of 50% female in kg
            rs = c('radius_scale')
            hs = c('height_scale')
            hs *= hmhs
            chest_c, chest_v = create_body(shape=p.GEOM_CAPSULE, radius=0.127*rs, length=0.01, orientation=p.getQuaternionFromEuler([0, np.pi/2.0, 0], physicsClientId=self.id)) #
            right_shoulders_c, right_shoulders_v = create_body(shape=p.GEOM_CAPSULE, radius=0.092*rs, length=0.225/8, position_offset=[-0.225/2.5 + 0.225/16, 0, 0], orientation=p.getQuaternionFromEuler([0, np.pi/2.0, 0], physicsClientId=self.id)) #
            left_shoulders_c, left_shoulders_v = create_body(shape=p.GEOM_CAPSULE, radius=0.092*rs, length=0.225/8, position_offset=[0.225/2.5 - 0.225/16, 0, 0], orientation=p.getQuaternionFromEuler([0, np.pi/2.0, 0], physicsClientId=self.id)) #
            neck_c, neck_v = create_body(shape=p.GEOM_CAPSULE, radius=0.05*rs, length=0.121*hs, position_offset=[0, 0, (0.2565 - 0.1415 - 0.025)*hs], visible=visible) # not position
            upperarm_c, upperarm_v = create_body(shape=p.GEOM_CAPSULE, radius=0.0355*rs, length=0.264*hs, position_offset=[0, 0, -0.264/2.0*hs]) #
            forearm_c, forearm_v = create_body(shape=p.GEOM_CAPSULE, radius=0.027*rs, length=0.234*hs, position_offset=[0, 0, -0.234/2.0*hs]) #
            hand_c, hand_v = create_body(shape=p.GEOM_SPHERE, radius=0.0355*rs, length=0, position_offset=[0, 0, -0.0355*rs]) #
            _, upperarm_invisible_v = create_body(shape=p.GEOM_CAPSULE, radius=0.0001*rs, length=0.0001*hs, position_offset=[0, 0, -0.264/2.0*hs], visible=0) #
            _, forearm_invisible_v = create_body(shape=p.GEOM_CAPSULE, radius=0.0001*rs, length=0.0001*hs, position_offset=[0, 0, -0.234/2.0*hs], visible=0) #
            _, hand_invisible_v = create_body(shape=p.GEOM_SPHERE, radius=0.0001*rs, length=0, position_offset=[0, 0, -0.0355*rs], visible=0) #

            self.hand_radius, self.elbow_radius, self.shoulder_radius = 0.0355*rs, 0.0355*rs, 0.0355*rs
            waist_c, waist_v = create_body(shape=p.GEOM_CAPSULE, radius=0.11*rs, length=0.009, orientation=p.getQuaternionFromEuler([0, np.pi/2.0, 0], physicsClientId=self.id)) #
            hips_c, hips_v = create_body(shape=p.GEOM_CAPSULE, radius=0.127*rs, length=0.117, position_offset=[0, 0, -0.15/2*hs], orientation=p.getQuaternionFromEuler([0, np.pi/2.0, 0], physicsClientId=self.id)) #
            thigh_c, thigh_v = create_body(shape=p.GEOM_CAPSULE, radius=0.0775*rs, length=0.391*hs, position_offset=[0, 0, -0.391/2.0*hs]) #
            shin_c, shin_v = create_body(shape=p.GEOM_CAPSULE, radius=0.045*rs, length=0.367*hs, position_offset=[0, 0, -0.367/2.0*hs]) #
            foot_c, foot_v = create_body(shape=p.GEOM_CAPSULE, radius=0.045*rs, length=0.195*hs, position_offset=[0, -0.09, -0.0225*rs], orientation=p.getQuaternionFromEuler([np.pi/2.0, 0, 0], physicsClientId=self.id)) #
            elbow_v = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=(0.0355+0.027)/2*rs, length=0, rgbaColor=[0.8, 0.6, 0.4, 1], visualFramePosition=[0, 0.01, 0], physicsClientId=self.id)
            elbow_invisible_v = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.0001*rs, length=0, rgbaColor=[0, 0, 0, 0], specularColor=[0, 0, 0], visualFramePosition=[0, 0.01, 0], physicsClientId=self.id)
            
            if self.cloth:
                # Cloth penetrates the spheres at the end of each capsule, so we create physical spheres at the joints
                invisible_v = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01*rs, length=0, rgbaColor=[0.8, 0.6, 0.4, 0], physicsClientId=self.id)
                shoulder_cloth_c, _ = create_body(shape=p.GEOM_SPHERE, radius=0.0355*rs, length=0)
                elbow_cloth_c, _ = create_body(shape=p.GEOM_SPHERE, radius=0.0355*rs, length=0)
                wrist_cloth_c, _ = create_body(shape=p.GEOM_SPHERE, radius=0.027*rs, length=0)

            head_scale = [0.89]*3
            head_pos = [-0.089, -0.09, -0.07]
            head_c = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=os.path.join(self.directory, 'head_female_male', 'BaseHeadMeshes_v5_female_cropped_reduced_compressed_vhacd.obj'), collisionFramePosition=head_pos, collisionFrameOrientation=p.getQuaternionFromEuler([np.pi/2.0, 0, 0], physicsClientId=self.id), meshScale=head_scale, physicsClientId=self.id)
            head_v = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=os.path.join(self.directory, 'head_female_male', 'BaseHeadMeshes_v5_female_cropped_reduced_compressed.obj'), rgbaColor=[0.8, 0.6, 0.4, visible], specularColor=specular_color, visualFramePosition=head_pos, visualFrameOrientation=p.getQuaternionFromEuler([np.pi/2.0, 0, 0], physicsClientId=self.id), meshScale=head_scale, physicsClientId=self.id)

            joint_p, joint_o = [0, 0, 0], [0, 0, 0, 1]
            chest_p = [0, 0, 0.15*hs]
            shoulders_p = [0, 0, 0.132/2*hs]
            neck_p = [0, 0, 0.132*hs]
            head_p = [0, 0, 0.12*hs]
            right_upperarm_p = [-0.092*rs - 0.067, 0, 0]
            left_upperarm_p = [0.092*rs + 0.067, 0, 0]
            forearm_p = [0, 0, -0.264*hs]
            hand_p = [0, 0, -(0.027*rs + 0.234*hs)]
            waist_p = [0, 0, 0.15/2*hs]
            hips_p = [0, 0, 0.923*hs]
            right_thigh_p = [-0.0775*rs - 0.0145, 0, -0.15/2*hs]
            left_thigh_p = [0.0775*rs + 0.0145, 0, -0.15/2*hs]
            shin_p = [0, 0, -0.391*hs]
            foot_p = [0, 0, -0.367*hs - 0.045/2]

        linkMasses = []
        linkCollisionShapeIndices = []
        linkVisualShapeIndices = []
        linkPositions = []
        linkOrientations = []
        linkInertialFramePositions = []
        linkInertialFrameOrientations = []
        linkParentIndices = []
        linkJointTypes = []
        linkJointAxis = []
        linkLowerLimits = []
        linkUpperLimits = []

        # NOTE: Waist and chest
        linkMasses.extend(m*np.array([0, 0, 0.13, 0.1]))
        linkCollisionShapeIndices.extend([joint_c, joint_c, waist_c, chest_c])
        linkVisualShapeIndices.extend([joint_v, joint_v, waist_v, chest_v])
        linkPositions.extend([waist_p, joint_p, joint_p, chest_p])
        linkOrientations.extend([joint_o]*4)
        linkInertialFramePositions.extend([[0, 0, 0]]*4)
        linkInertialFrameOrientations.extend([[0, 0, 0, 1]]*4)
        linkParentIndices.extend([0, 1, 2, 3])
        linkJointTypes.extend([p.JOINT_REVOLUTE]*3 + [p.JOINT_FIXED])
        linkJointAxis.extend([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
        linkLowerLimits.extend(np.array([np.deg2rad(-180), np.deg2rad(-180), np.deg2rad(-180), 0]))
        linkUpperLimits.extend(np.array([np.deg2rad(180), np.deg2rad(180), np.deg2rad(180), 0]))

        # NOTE: Shoulders, neck, and head
        linkMasses.extend(m*np.array([0, 0, 0.05, 0, 0, 0.05, 0.01, 0, 0, 0.07]))
        linkCollisionShapeIndices.extend([joint_c, joint_c, right_shoulders_c, joint_c, joint_c, left_shoulders_c, neck_c, joint_c, joint_c, head_c])
        linkVisualShapeIndices.extend([joint_v, joint_v, right_shoulders_v, joint_v, joint_v, left_shoulders_v, neck_v, joint_v, joint_v, head_v])
        linkPositions.extend([shoulders_p, shoulders_p, joint_p, shoulders_p, shoulders_p, joint_p, neck_p, head_p, joint_p, joint_p])
        linkOrientations.extend([joint_o]*10)
        linkInertialFramePositions.extend([[0, 0, 0]]*10)
        linkInertialFrameOrientations.extend([[0, 0, 0, 1]]*10)
        linkParentIndices.extend([4, 5, 6, 4, 8, 9, 4, 11, 12, 13])
        linkJointTypes.extend([p.JOINT_FIXED]*7+ [p.JOINT_REVOLUTE]*3)
        linkJointAxis.extend([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        linkLowerLimits.extend(np.array([0, 0, 0, 0, 0, 0, 0, np.deg2rad(-180), np.deg2rad(-180), np.deg2rad(-180)])*limit_scale)
        linkUpperLimits.extend(np.array([0, 0, 0, 0, 0, 0, 0, np.deg2rad(180), np.deg2rad(180), np.deg2rad(180)])*limit_scale)

        # NOTE: Right arm
        linkMasses.extend(m*np.array([0, 0, 0.033, 0, 0.019, 0, 0.0065]))
        if not self.cloth:
            linkCollisionShapeIndices.extend([joint_c, joint_c, upperarm_c, joint_c, forearm_c, joint_c, hand_c])
            linkVisualShapeIndices.extend([joint_v, joint_v, upperarm_v, elbow_v, forearm_v, joint_v, hand_v])
        else:
            linkCollisionShapeIndices.extend([joint_c, shoulder_cloth_c, upperarm_c, elbow_cloth_c, forearm_c, wrist_cloth_c, hand_c])
            linkVisualShapeIndices.extend([joint_v, invisible_v, upperarm_v, elbow_v, forearm_v, invisible_v, hand_v])
        linkPositions.extend([right_upperarm_p, joint_p, joint_p, forearm_p, joint_p, hand_p, joint_p])
        linkOrientations.extend([joint_o]*7)
        linkInertialFramePositions.extend([[0, 0, 0]]*7)
        linkInertialFrameOrientations.extend([[0, 0, 0, 1]]*7)
        linkParentIndices.extend([7, 15, 16, 17, 18, 19, 20])
        linkJointTypes.extend([p.JOINT_REVOLUTE]*7)
        linkJointAxis.extend([[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
        linkLowerLimits.extend(np.array([np.deg2rad(-180), np.deg2rad(-188), np.deg2rad(-90), np.deg2rad(-128), np.deg2rad(-90), np.deg2rad(-81), np.deg2rad(-27)])*limit_scale)
        linkUpperLimits.extend(np.array([np.deg2rad(198), np.deg2rad(61), np.deg2rad(90), np.deg2rad(0), np.deg2rad(90), np.deg2rad(90), np.deg2rad(47)])*limit_scale)

        # NOTE: Left arm
        linkMasses.extend(m*np.array([0, 0, 0.033, 0, 0.019, 0, 0.0065]))
        if not self.cloth:
            linkCollisionShapeIndices.extend([joint_c, joint_c, upperarm_c, joint_c, forearm_c, joint_c, hand_c])
            linkVisualShapeIndices.extend([joint_v, joint_v, upperarm_v, elbow_v, forearm_v, joint_v, hand_v])
        else:
            linkCollisionShapeIndices.extend([joint_c, shoulder_cloth_c, upperarm_c, elbow_cloth_c, forearm_c, wrist_cloth_c, hand_c])
            linkVisualShapeIndices.extend([joint_v, invisible_v, upperarm_v, elbow_v, forearm_v, invisible_v, hand_v])
        linkPositions.extend([left_upperarm_p, joint_p, joint_p, forearm_p, joint_p, hand_p, joint_p])
        linkOrientations.extend([joint_o]*7)
        linkInertialFramePositions.extend([[0, 0, 0]]*7)
        linkInertialFrameOrientations.extend([[0, 0, 0, 1]]*7)
        linkParentIndices.extend([10, 22, 23, 24, 25, 26, 27])
        linkJointTypes.extend([p.JOINT_REVOLUTE]*7)
        linkJointAxis.extend([[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
        linkLowerLimits.extend(np.array([np.deg2rad(-198), np.deg2rad(-188), np.deg2rad(-90), np.deg2rad(-128), np.deg2rad(-90), np.deg2rad(-90), np.deg2rad(-47)])*limit_scale)
        linkUpperLimits.extend(np.array([np.deg2rad(180), np.deg2rad(61), np.deg2rad(90), np.deg2rad(0), np.deg2rad(90), np.deg2rad(81), np.deg2rad(27)])*limit_scale)

        # NOTE: Right leg
        linkMasses.extend(m*np.array([0, 0, 0.105, 0.0475, 0, 0, 0.014]))
        linkCollisionShapeIndices.extend([joint_c, joint_c, thigh_c, shin_c, joint_c, joint_c, foot_c])
        linkVisualShapeIndices.extend([joint_v, joint_v, thigh_v, shin_v, joint_v, joint_v, foot_v])
        linkPositions.extend([right_thigh_p, joint_p, joint_p, shin_p, foot_p, joint_p, joint_p])
        linkOrientations.extend([joint_o]*7)
        linkInertialFramePositions.extend([[0, 0, 0]]*7)
        linkInertialFrameOrientations.extend([[0, 0, 0, 1]]*7)
        linkParentIndices.extend([0, 29, 30, 31, 32, 33, 34])
        linkJointTypes.extend([p.JOINT_REVOLUTE]*7)
        linkJointAxis.extend([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        linkLowerLimits.extend(np.array([np.deg2rad(-127), np.deg2rad(-40), np.deg2rad(-45), 0, np.deg2rad(-35), np.deg2rad(-23), np.deg2rad(-43)]))
        linkUpperLimits.extend(np.array([np.deg2rad(30), np.deg2rad(45), np.deg2rad(40), np.deg2rad(130), np.deg2rad(38), np.deg2rad(24), np.deg2rad(35)]))

        # NOTE: Left leg
        linkMasses.extend(m*np.array([0, 0, 0.105, 0.0475, 0, 0, 0.014]))
        linkCollisionShapeIndices.extend([joint_c, joint_c, thigh_c, shin_c, joint_c, joint_c, foot_c])
        linkVisualShapeIndices.extend([joint_v, joint_v, thigh_v, shin_v, joint_v, joint_v, foot_v])
        linkPositions.extend([left_thigh_p, joint_p, joint_p, shin_p, foot_p, joint_p, joint_p])
        linkOrientations.extend([joint_o]*7)
        linkInertialFramePositions.extend([[0, 0, 0]]*7)
        linkInertialFrameOrientations.extend([[0, 0, 0, 1]]*7)
        linkParentIndices.extend([0, 36, 37, 38, 39, 40, 41])
        linkJointTypes.extend([p.JOINT_REVOLUTE]*7)
        linkJointAxis.extend([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        linkLowerLimits.extend(np.array([np.deg2rad(-127), np.deg2rad(-45), np.deg2rad(-40), 0, np.deg2rad(-35), np.deg2rad(-24), np.deg2rad(-35)]))
        linkUpperLimits.extend(np.array([np.deg2rad(30), np.deg2rad(40), np.deg2rad(45), np.deg2rad(130), np.deg2rad(38), np.deg2rad(23), np.deg2rad(43)]))

        human = p.createMultiBody(baseMass=0 if static else m*0.14, baseCollisionShapeIndex=hips_c, baseVisualShapeIndex=hips_v, basePosition=[10, 10, 10], baseOrientation=[0, 0, 0, 1], linkMasses=linkMasses, linkCollisionShapeIndices=linkCollisionShapeIndices, linkVisualShapeIndices=linkVisualShapeIndices, linkPositions=linkPositions, linkOrientations=linkOrientations, linkInertialFramePositions=linkInertialFramePositions, linkInertialFrameOrientations=linkInertialFrameOrientations, linkParentIndices=linkParentIndices, linkJointTypes=linkJointTypes, linkJointAxis=linkJointAxis, linkLowerLimits=linkLowerLimits, linkUpperLimits=linkUpperLimits, useMaximalCoordinates=False, flags=p.URDF_USE_SELF_COLLISION, physicsClientId=self.id)
        # Self collision has been enabled for the person
        # For stability: Remove all collisions except between the arms/legs and the other body parts
        num_joints = p.getNumJoints(human, physicsClientId=self.id)
        for i in range(-1, num_joints):
            for j in range(-1, num_joints):
                p.setCollisionFilterPair(human, human, i, j, 0, physicsClientId=self.id)
        for i in range(7, 14): # Right arm
            for j in list(range(-1, 4)) + list(range(14, num_joints)):
                p.setCollisionFilterPair(human, human, i, j, 1, physicsClientId=self.id)
        for i in range(17, 24): # Left arm
            for j in list(range(-1, 14)) + list(range(24, num_joints)):
                p.setCollisionFilterPair(human, human, i, j, 1, physicsClientId=self.id)
        for i in range(28, 35): # Right leg
            for j in [-1] + list(range(4, 28)) + list(range(35, num_joints)):
                p.setCollisionFilterPair(human, human, i, j, 1, physicsClientId=self.id)
        for i in range(35, num_joints): # Left leg
            for j in [-1] + list(range(4, 35)):
                p.setCollisionFilterPair(human, human, i, j, 1, physicsClientId=self.id)

        # Enforce joint limits
        human_joint_states = p.getJointStates(human, jointIndices=list(range(p.getNumJoints(human, physicsClientId=self.id))), physicsClientId=self.id)
        human_joint_positions = np.array([x[0] for x in human_joint_states])
        for j in range(p.getNumJoints(human, physicsClientId=self.id)):
            p.resetJointState(human, jointIndex=j, targetValue=0, targetVelocity=0, physicsClientId=self.id)

        # NOTE: Invisible right arm
        linkMasses = []
        linkCollisionShapeIndices = []
        linkVisualShapeIndices = []
        linkPositions = []
        linkOrientations = []
        linkInertialFramePositions = []
        linkInertialFrameOrientations = []
        linkParentIndices = []
        linkJointTypes = []
        linkJointAxis = []
        linkLowerLimits = []
        linkUpperLimits = []

        linkMasses.extend(m*np.array([0, 0, 0.033, 0, 0.019, 0, 0.0065]))
        if not self.cloth:
            linkCollisionShapeIndices.extend([joint_c, joint_c, upperarm_c, joint_c, forearm_c, joint_c, hand_c])
            linkVisualShapeIndices.extend([joint_v, joint_v, upperarm_invisible_v, elbow_invisible_v, forearm_invisible_v, joint_v, hand_invisible_v])
        else:
            linkCollisionShapeIndices.extend([joint_c, shoulder_cloth_c, upperarm_c, elbow_cloth_c, forearm_c, wrist_cloth_c, hand_c])
            linkVisualShapeIndices.extend([joint_v, invisible_v, upperarm_v, elbow_v, forearm_v, invisible_v, hand_v])
        linkPositions.extend([right_upperarm_p, joint_p, joint_p, forearm_p, joint_p, hand_p, joint_p])
        linkOrientations.extend([joint_o]*7)
        linkInertialFramePositions.extend([[0, 0, 0]]*7)
        linkInertialFrameOrientations.extend([[0, 0, 0, 1]]*7)
        linkParentIndices.extend([0, 1, 2, 3, 4, 5, 6])
        linkJointTypes.extend([p.JOINT_REVOLUTE]*7)
        linkJointAxis.extend([[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
        linkLowerLimits.extend(np.array([np.deg2rad(-180), np.deg2rad(-188), np.deg2rad(-90), np.deg2rad(-128), np.deg2rad(-90), np.deg2rad(-81), np.deg2rad(-27)])*limit_scale)
        linkUpperLimits.extend(np.array([np.deg2rad(198), np.deg2rad(61), np.deg2rad(90), np.deg2rad(0), np.deg2rad(90), np.deg2rad(90), np.deg2rad(47)])*limit_scale)
        right_arm = p.createMultiBody(baseMass=0 if static else m*0.14, baseCollisionShapeIndex=joint_c, baseVisualShapeIndex=joint_v, basePosition=[8, 10, 10], baseOrientation=[0, 0, 0, 1], linkMasses=linkMasses, linkCollisionShapeIndices=linkCollisionShapeIndices, linkVisualShapeIndices=linkVisualShapeIndices, linkPositions=linkPositions, linkOrientations=linkOrientations, linkInertialFramePositions=linkInertialFramePositions, linkInertialFrameOrientations=linkInertialFrameOrientations, linkParentIndices=linkParentIndices, linkJointTypes=linkJointTypes, linkJointAxis=linkJointAxis, linkLowerLimits=linkLowerLimits, linkUpperLimits=linkUpperLimits, useMaximalCoordinates=False, flags=p.URDF_USE_SELF_COLLISION, physicsClientId=self.id)

        num_joints_arm = p.getNumJoints(right_arm, physicsClientId=self.id)
        for i in range(-1, num_joints_arm):
            for j in range(-1, num_joints_arm):
                p.setCollisionFilterPair(right_arm, right_arm, i, j, 0, physicsClientId=self.id)
        for i in range(-1, num_joints_arm):
            for j in range(-1, num_joints):
                p.setCollisionFilterPair(right_arm, human, i, j, 0, physicsClientId=self.id)

        # NOTE: Invisible left arm
        linkMasses = []
        linkCollisionShapeIndices = []
        linkVisualShapeIndices = []
        linkPositions = []
        linkOrientations = []
        linkInertialFramePositions = []
        linkInertialFrameOrientations = []
        linkParentIndices = []
        linkJointTypes = []
        linkJointAxis = []
        linkLowerLimits = []
        linkUpperLimits = []

        linkMasses.extend(m*np.array([0, 0, 0.033, 0, 0.019, 0, 0.0065]))
        if not self.cloth:
            linkCollisionShapeIndices.extend([joint_c, joint_c, upperarm_c, joint_c, forearm_c, joint_c, hand_c])
            linkVisualShapeIndices.extend([joint_v, joint_v, upperarm_invisible_v, elbow_invisible_v, forearm_invisible_v, joint_v, hand_invisible_v])
        else:
            linkCollisionShapeIndices.extend([joint_c, shoulder_cloth_c, upperarm_c, elbow_cloth_c, forearm_c, wrist_cloth_c, hand_c])
            linkVisualShapeIndices.extend([joint_v, invisible_v, upperarm_v, elbow_v, forearm_v, invisible_v, hand_v])
        linkPositions.extend([left_upperarm_p, joint_p, joint_p, forearm_p, joint_p, hand_p, joint_p])
        linkOrientations.extend([joint_o]*7)
        linkInertialFramePositions.extend([[0, 0, 0]]*7)
        linkInertialFrameOrientations.extend([[0, 0, 0, 1]]*7)
        linkParentIndices.extend([0, 1, 2, 3, 4, 5, 6])
        linkJointTypes.extend([p.JOINT_REVOLUTE]*7)
        linkJointAxis.extend([[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
        linkLowerLimits.extend(np.array([np.deg2rad(-198), np.deg2rad(-188), np.deg2rad(-90), np.deg2rad(-128), np.deg2rad(-90), np.deg2rad(-90), np.deg2rad(-47)])*limit_scale)
        linkUpperLimits.extend(np.array([np.deg2rad(180), np.deg2rad(61), np.deg2rad(90), np.deg2rad(0), np.deg2rad(90), np.deg2rad(81), np.deg2rad(27)])*limit_scale)
        left_arm = p.createMultiBody(baseMass=0 if static else m*0.14, baseCollisionShapeIndex=joint_c, baseVisualShapeIndex=joint_v, basePosition=[12, 10, 10], baseOrientation=[0, 0, 0, 1], linkMasses=linkMasses, linkCollisionShapeIndices=linkCollisionShapeIndices, linkVisualShapeIndices=linkVisualShapeIndices, linkPositions=linkPositions, linkOrientations=linkOrientations, linkInertialFramePositions=linkInertialFramePositions, linkInertialFrameOrientations=linkInertialFrameOrientations, linkParentIndices=linkParentIndices, linkJointTypes=linkJointTypes, linkJointAxis=linkJointAxis, linkLowerLimits=linkLowerLimits, linkUpperLimits=linkUpperLimits, useMaximalCoordinates=False, flags=p.URDF_USE_SELF_COLLISION, physicsClientId=self.id)

        num_joints_arm = p.getNumJoints(left_arm, physicsClientId=self.id)
        for i in range(-1, num_joints_arm):
            for j in range(-1, num_joints_arm):
                p.setCollisionFilterPair(left_arm, left_arm, i, j, 0, physicsClientId=self.id)
        for i in range(-1, num_joints_arm):
            for j in range(-1, num_joints):
                p.setCollisionFilterPair(left_arm, human, i, j, 0, physicsClientId=self.id)

        return human, left_arm, right_arm