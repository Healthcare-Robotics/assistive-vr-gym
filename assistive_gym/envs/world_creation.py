import os
import numpy as np
import pybullet as p
from .human_creation import HumanCreation
from .human_creation_vr import HumanCreationVR

class WorldCreation:
    def __init__(self, pid, robot_type='pr2', task='scratch_itch', vr=False, replay=False, new=False, time_step=0.02, np_random=None, config=None):
        self.id = pid
        self.robot_type = robot_type
        self.task = task
        self.time_step = time_step
        self.np_random = np_random
        self.config = config
        self.vr = vr
        self.replay = replay
        self.new = new
        self.directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets')
        if self.vr or self.replay:
            self.human_creation = HumanCreationVR(self.id, np_random=np_random, cloth=(task=='dressing'))
        else:
            self.human_creation = HumanCreation(self.id, np_random=np_random, cloth=(task=='dressing'))
        self.human_limit_scale = 1.0
        self.human_strength = 1.0
        self.human_tremors = np.zeros(10)

    def create_new_world(self, furniture_type='wheelchair', static_human_base=False, human_impairment='random', print_joints=False, gender='random', hipbone_to_mouth_height=None):
        p.resetSimulation(physicsClientId=self.id)

        # Configure camera position
        p.resetDebugVisualizerCamera(cameraDistance=1.75, cameraYaw=-25, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.4], physicsClientId=self.id)

        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.id)

        # Load all models off screen and then move them into place
        p.loadURDF(os.path.join(self.directory, 'plane', 'plane.urdf'), physicsClientId=self.id)

        # Disable rendering during creation
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.id)

        if furniture_type == 'wheelchair':
            # Create wheelchair
            if self.robot_type in ['jaco', 'kinova_gen3']:
                furniture = p.loadURDF(os.path.join(self.directory, 'wheelchair', 'wheelchair.urdf' if self.task not in ['dressing'] else 'wheelchair.urdf'), physicsClientId=self.id)
            else:
                furniture = p.loadURDF(os.path.join(self.directory, 'wheelchair', 'wheelchair.urdf'), physicsClientId=self.id)
            # Initialize chair position
            p.resetBasePositionAndOrientation(furniture, [0.0, 0.09, -0.01], p.getQuaternionFromEuler([np.pi/2.0, 0, -np.pi/2.0-0.05], physicsClientId=self.id), physicsClientId=self.id)
        elif furniture_type == 'bed':
            mesh_scale = [1.1]*3
            bed_visual = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=os.path.join(self.directory, 'bed', 'bed_single_reduced.obj'), rgbaColor=[1, 1, 1, 1], specularColor=[0.2, 0.2, 0.2], meshScale=mesh_scale, physicsClientId=self.id)
            bed_collision = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=os.path.join(self.directory, 'bed', 'bed_single_reduced_vhacd.obj'), meshScale=mesh_scale, flags=p.GEOM_FORCE_CONCAVE_TRIMESH, physicsClientId=self.id)
            furniture = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=bed_collision, baseVisualShapeIndex=bed_visual, basePosition=[0, 0, 0], useMaximalCoordinates=True, physicsClientId=self.id)
            # Initialize bed position
            p.resetBasePositionAndOrientation(furniture, [-0.1, 0, 0], p.getQuaternionFromEuler([np.pi/2.0, 0, 0], physicsClientId=self.id), physicsClientId=self.id)
        elif furniture_type == 'table':
            furniture = p.loadURDF(os.path.join(self.directory, 'table', 'table.urdf'), basePosition=[0, -0.35, 0], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2.0], physicsClientId=self.id), physicsClientId=self.id)
        else:
            furniture = None

        # Choose gender
        if gender not in ['male', 'female']:
            gender = self.np_random.choice(['male', 'female'])
        # Specify human impairments
        if human_impairment == 'random':
            human_impairment = self.np_random.choice(['none', 'limits', 'weakness', 'tremor'])
        elif human_impairment == 'no_tremor':
            human_impairment = self.np_random.choice(['none', 'limits', 'weakness'])
        self.human_impairment = human_impairment
        self.human_limit_scale = 1.0 if human_impairment != 'limits' else self.np_random.uniform(0.5, 1.0)
        self.human_strength = 1.0 if human_impairment != 'weakness' else self.np_random.uniform(0.25, 1.0)
        human, left_arm, right_arm, human_lower_limits, human_upper_limits = self.init_human(static_human_base, self.human_limit_scale, print_joints, gender=gender, hipbone_to_mouth_height=hipbone_to_mouth_height)

        p.setTimeStep(self.time_step, physicsClientId=self.id)
        if not self.vr and not self.replay:
            # Disable real time simulation so that the simulation only advances when we call stepSimulation
            p.setRealTimeSimulation(0, physicsClientId=self.id)

        if self.robot_type == 'pr2':
            robot, robot_lower_limits, robot_upper_limits, robot_right_arm_joint_indices, robot_left_arm_joint_indices = self.init_pr2(print_joints)
        elif self.robot_type == 'sawyer':
            robot, robot_lower_limits, robot_upper_limits, robot_right_arm_joint_indices, robot_left_arm_joint_indices = self.init_sawyer(print_joints)
        elif self.robot_type == 'baxter':
            robot, robot_lower_limits, robot_upper_limits, robot_right_arm_joint_indices, robot_left_arm_joint_indices = self.init_baxter(print_joints)
        elif self.robot_type == 'jaco':
            robot, robot_lower_limits, robot_upper_limits, robot_right_arm_joint_indices, robot_left_arm_joint_indices = self.init_jaco(print_joints)
        elif self.robot_type == 'kinova_gen3':
            robot, robot_lower_limits, robot_upper_limits, robot_right_arm_joint_indices, robot_left_arm_joint_indices = self.init_kinova_gen3(print_joints)
        else:
            robot, robot_lower_limits, robot_upper_limits, robot_right_arm_joint_indices, robot_left_arm_joint_indices = None, None, None, None, None

        return human, left_arm, right_arm, furniture, robot, robot_lower_limits, robot_upper_limits, human_lower_limits, human_upper_limits, robot_right_arm_joint_indices, robot_left_arm_joint_indices, gender


    def init_human(self, static_human_base=False, limit_scale=1.0, print_joints=False, gender='random', hipbone_to_mouth_height=None):
        if self.vr:
            human, left_arm, right_arm = self.human_creation.create_human(static=static_human_base, limit_scale=limit_scale, specular_color=[0.1, 0.1, 0.1], gender=gender, config=self.config, hipbone_to_mouth_height=hipbone_to_mouth_height, visible=0)
        elif self.replay:
            human, left_arm, right_arm = self.human_creation.create_human(static=static_human_base, limit_scale=limit_scale, specular_color=[0.1, 0.1, 0.1], gender=gender, config=self.config, hipbone_to_mouth_height=hipbone_to_mouth_height, visible=1)
        else:
            human = self.human_creation.create_human(static=static_human_base, limit_scale=limit_scale, specular_color=[0.1, 0.1, 0.1], gender=gender, config=self.config, hipbone_to_mouth_height=hipbone_to_mouth_height, new=self.new)
            left_arm, right_arm = None, None
        if print_joints:
            self.print_joint_info(human, show_fixed=True)

        lower_limits, upper_limits = self.enforce_joint_limits(human)
        return human, left_arm, right_arm, lower_limits, upper_limits

    def enforce_joint_limits(self, body):
        # Enforce joint limits
        joint_states = p.getJointStates(body, jointIndices=list(range(p.getNumJoints(body, physicsClientId=self.id))), physicsClientId=self.id)
        joint_positions = np.array([x[0] for x in joint_states])
        lower_limits = []
        upper_limits = []
        for j in range(p.getNumJoints(body, physicsClientId=self.id)):
            joint_info = p.getJointInfo(body, j, physicsClientId=self.id)
            joint_name = joint_info[1]
            joint_pos = joint_positions[j]
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            if lower_limit == 0 and upper_limit == -1:
                lower_limit = -1e10
                upper_limit = 1e10
            lower_limits.append(lower_limit)
            upper_limits.append(upper_limit)
            if joint_pos < lower_limit:
                p.resetJointState(body, jointIndex=j, targetValue=lower_limit, targetVelocity=0, physicsClientId=self.id)
            elif joint_pos > upper_limit:
                p.resetJointState(body, jointIndex=j, targetValue=upper_limit, targetVelocity=0, physicsClientId=self.id)
        lower_limits = np.array(lower_limits)
        upper_limits = np.array(upper_limits)
        return lower_limits, upper_limits

    def setup_human_joints(self, human, joints_positions, controllable_joints, use_static_joints=True, human_reactive_force=None, human_reactive_gain=0.05, non_static_joints=[], add_joint_positions=False):
        if self.human_impairment != 'tremor':
            self.human_tremors = np.zeros(len(controllable_joints))
        elif len(controllable_joints) == 4:
            self.human_tremors = self.np_random.uniform(np.deg2rad(-20), np.deg2rad(20), size=len(controllable_joints))
        else:
            self.human_tremors = self.np_random.uniform(np.deg2rad(-10), np.deg2rad(10), size=len(controllable_joints))
        # Set starting joint positions
        human_joint_states = p.getJointStates(human, jointIndices=list(range(p.getNumJoints(human, physicsClientId=self.id))), physicsClientId=self.id)
        human_joint_positions = np.array([x[0] for x in human_joint_states])
        for j in range(p.getNumJoints(human, physicsClientId=self.id)):
            set_position = None
            for j_index, j_angle in joints_positions:
                if j == j_index:
                    if self.vr or self.replay or self.new or self.task == 'bed_bathing':
                        set_position = j_angle if not add_joint_positions else (human_joint_positions[j] + j_angle)
                        p.resetJointState(human, jointIndex=j, targetValue=set_position, targetVelocity=0, physicsClientId=self.id)
                        break
                    else:
                        p.resetJointState(human, jointIndex=j, targetValue=j_angle, targetVelocity=0, physicsClientId=self.id)
                        set_position = j_angle
                        break
            if use_static_joints and j not in controllable_joints and j not in non_static_joints:
                # Make all non controllable joints on the person static by setting mass of each link (joint) to 0
                p.changeDynamics(human, j, mass=0, physicsClientId=self.id)
                # Set velocities to 0
                p.resetJointState(human, jointIndex=j, targetValue=human_joint_positions[j] if set_position is None else set_position, targetVelocity=0, physicsClientId=self.id)

        # By default, all joints have motors enabled by default that prevent free motion. Disable these motors in human
        if not self.vr:
            if self.replay or self.task == 'bed_bathing':
                for j in range(p.getNumJoints(human, physicsClientId=self.id)):
                    p.setJointMotorControl2(human, jointIndex=j, controlMode=p.VELOCITY_CONTROL, force=0.1, physicsClientId=self.id)
            elif not self.new:
                for j in range(p.getNumJoints(human, physicsClientId=self.id)):
                    p.setJointMotorControl2(human, jointIndex=j, controlMode=p.VELOCITY_CONTROL, force=0, physicsClientId=self.id)

        self.enforce_joint_limits(human)

        if human_reactive_force is not None:
            # NOTE: This runs a Position / Velocity PD controller for each joint motor on the human
            human_joint_states = p.getJointStates(human, jointIndices=controllable_joints, physicsClientId=self.id)
            target_human_joint_positions = np.array([x[0] for x in human_joint_states])
            forces = [human_reactive_force * self.human_strength] * len(target_human_joint_positions)
            p.setJointMotorControlArray(human, jointIndices=controllable_joints, controlMode=p.POSITION_CONTROL, targetPositions=target_human_joint_positions, positionGains=np.array([human_reactive_gain]*len(forces)), forces=forces, physicsClientId=self.id)

    def init_pr2(self, print_joints=False):
        if self.task == 'arm_manipulation':
            robot = p.loadURDF(os.path.join(self.directory, 'PR2', 'pr2_no_torso_lift_tall_arm_manipulation.urdf'), useFixedBase=True, basePosition=[0, 0, 0], physicsClientId=self.id)
            robot_right_arm_joint_indices = [42, 43, 44, 46, 47, 49, 50]
            robot_left_arm_joint_indices = [65, 66, 67, 69, 70, 72, 73]
        else:
            robot = p.loadURDF(os.path.join(self.directory, 'PR2', 'pr2_no_torso_lift_tall.urdf'), useFixedBase=True, basePosition=[0, 0, 0], flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=self.id)
            robot_right_arm_joint_indices = [42, 43, 44, 46, 47, 49, 50]
            robot_left_arm_joint_indices = [64, 65, 66, 68, 69, 71, 72]
        if print_joints:
            self.print_joint_info(robot, show_fixed=True)

        # Initialize and position PR2
        p.resetBasePositionAndOrientation(robot, [-2, -2, 0], [0, 0, 0, 1], physicsClientId=self.id)

        # Recolor PR2
        if self.task == 'arm_manipulation':
            for i in [19, 42, 65]:
                p.changeVisualShape(robot, i, rgbaColor=[1.0, 1.0, 1.0, 1.0], physicsClientId=self.id)
            for i in [43, 46, 49, 59, 61, 66, 69, 72, 82, 84]:
                p.changeVisualShape(robot, i, rgbaColor=[0.4, 0.4, 0.4, 1.0], physicsClientId=self.id)
            for i in [45, 51, 68, 74]:
                p.changeVisualShape(robot, i, rgbaColor=[0.7, 0.7, 0.7, 1.0], physicsClientId=self.id)
        else:
            for i in [19, 42, 64]:
                p.changeVisualShape(robot, i, rgbaColor=[1.0, 1.0, 1.0, 1.0], physicsClientId=self.id)
            for i in [43, 46, 49, 58, 60, 65, 68, 71, 80, 82]:
                p.changeVisualShape(robot, i, rgbaColor=[0.4, 0.4, 0.4, 1.0], physicsClientId=self.id)
            for i in [45, 51, 67, 73]:
                p.changeVisualShape(robot, i, rgbaColor=[0.7, 0.7, 0.7, 1.0], physicsClientId=self.id)
        p.changeVisualShape(robot, 20, rgbaColor=[0.8, 0.8, 0.8, 1.0], physicsClientId=self.id)
        p.changeVisualShape(robot, 40, rgbaColor=[0.6, 0.6, 0.6, 1.0], physicsClientId=self.id)

        # Grab and enforce robot arm joint limits
        lower_limits, upper_limits = self.enforce_joint_limits(robot)

        return robot, lower_limits, upper_limits, robot_right_arm_joint_indices, robot_left_arm_joint_indices

    def init_sawyer(self, print_joints=False):
        # Enable self collisions to prevent the arm from going through the torso
        if self.task == 'arm_manipulation':
            robot = p.loadURDF(os.path.join(self.directory, 'sawyer', 'sawyer_arm_manipulation.urdf'), useFixedBase=True, basePosition=[0, 0, 0], flags=p.URDF_USE_SELF_COLLISION, physicsClientId=self.id)
            # Disable collisions between the fingers and the tool
            for i in range(16, 24):
                p.setCollisionFilterPair(robot, robot, i, 24, 0, physicsClientId=self.id)
        else:
            robot = p.loadURDF(os.path.join(self.directory, 'sawyer', 'sawyer.urdf'), useFixedBase=True, basePosition=[0, 0, 0], flags=p.URDF_USE_SELF_COLLISION, physicsClientId=self.id)
        # Remove collisions between the various arm links for stability
        for i in range(3, 24):
            for j in range(3, 24):
                p.setCollisionFilterPair(robot, robot, i, j, 0, physicsClientId=self.id)
        for i in range(0, 3):
            for j in range(0, 9):
                p.setCollisionFilterPair(robot, robot, i, j, 0, physicsClientId=self.id)
        robot_arm_joint_indices = [3, 8, 9, 10, 11, 13, 16]
        if print_joints:
            self.print_joint_info(robot, show_fixed=True)

        # Initialize and position
        p.resetBasePositionAndOrientation(robot, [-2, -2, 0.975], [0, 0, 0, 1], physicsClientId=self.id)

        # Grab and enforce robot arm joint limits
        lower_limits, upper_limits = self.enforce_joint_limits(robot)

        return robot, lower_limits, upper_limits, robot_arm_joint_indices, robot_arm_joint_indices

    def init_baxter(self, print_joints=False):
        if self.task == 'arm_manipulation':
            robot = p.loadURDF(os.path.join(self.directory, 'baxter', 'baxter_custom_arm_manipulation.urdf'), useFixedBase=True, basePosition=[0, 0, 0], physicsClientId=self.id)
            robot_right_arm_joint_indices = [12, 13, 14, 15, 16, 18, 19]
            robot_left_arm_joint_indices = [35, 36, 37, 38, 39, 41, 42]
        else:
            robot = p.loadURDF(os.path.join(self.directory, 'baxter', 'baxter_custom.urdf'), useFixedBase=True, basePosition=[0, 0, 0], physicsClientId=self.id)
            robot_right_arm_joint_indices = [12, 13, 14, 15, 16, 18, 19]
            robot_left_arm_joint_indices = [34, 35, 36, 37, 38, 40, 41]
        if print_joints:
            self.print_joint_info(robot, show_fixed=True)

        # Initialize and position
        p.resetBasePositionAndOrientation(robot, [-2, -2, 0.975], [0, 0, 0, 1], physicsClientId=self.id)

        if self.task == 'arm_manipulation':
            for i in [20, 21, 23, 32, 33, 43, 44, 46, 55, 56]:
                p.changeVisualShape(robot, i, rgbaColor=[1.0, 1.0, 1.0, 0.0], physicsClientId=self.id)
        else:
            for i in [20, 21, 23, 31, 32, 42, 43, 45, 53, 54]:
                p.changeVisualShape(robot, i, rgbaColor=[1.0, 1.0, 1.0, 0.0], physicsClientId=self.id)

        # Grab and enforce robot arm joint limits
        lower_limits, upper_limits = self.enforce_joint_limits(robot)

        return robot, lower_limits, upper_limits, robot_right_arm_joint_indices, robot_left_arm_joint_indices

    def init_jaco(self, print_joints=False):
        # Enable self collisions to prevent the arm from going through the torso
        if self.task == 'arm_manipulation':
            robot = p.loadURDF(os.path.join(self.directory, 'jaco', 'j2s7s300_gym_arm_manipulation.urdf'), useFixedBase=True, basePosition=[0, 0, 0], flags=p.URDF_USE_SELF_COLLISION, physicsClientId=self.id)
            # Disable collisions between the fingers and the tool
            for i in range(10, 16):
                p.setCollisionFilterPair(robot, robot, i, 9, 0, physicsClientId=self.id)
        else:
            robot = p.loadURDF(os.path.join(self.directory, 'jaco', 'j2s7s300_gym.urdf'), useFixedBase=True, basePosition=[0, 0, 0], flags=p.URDF_USE_SELF_COLLISION, physicsClientId=self.id)
        robot_arm_joint_indices = [1, 2, 3, 4, 5, 6, 7]
        if print_joints:
            self.print_joint_info(robot, show_fixed=True)

        # Initialize and position
        p.resetBasePositionAndOrientation(robot, [-2, -2, 0.975], [0, 0, 0, 1], physicsClientId=self.id)

        # Grab and enforce robot arm joint limits
        lower_limits, upper_limits = self.enforce_joint_limits(robot)

        return robot, lower_limits, upper_limits, robot_arm_joint_indices, robot_arm_joint_indices

    def init_kinova_gen3(self, print_joints=False):
        robot = p.loadURDF(os.path.join(self.directory, 'kinova_gen3', 'GEN3_URDF_V12.urdf'), useFixedBase=True, basePosition=[0, 0, 0], flags=p.URDF_USE_SELF_COLLISION, physicsClientId=self.id)
        robot_arm_joint_indices = [0, 1, 2, 3, 4, 5, 6]
        if print_joints:
            self.print_joint_info(robot, show_fixed=True)

        # Initialize and position
        p.resetBasePositionAndOrientation(robot, [-0.95, -0.3, 0.975], [0, 0, 0, 1], physicsClientId=self.id)

        # Grab and enforce robot arm joint limits
        lower_limits, upper_limits = self.enforce_joint_limits(robot)

        return robot, lower_limits, upper_limits, robot_arm_joint_indices, robot_arm_joint_indices

    def set_gripper_open_position(self, robot, position=0, left=True, set_instantly=False, indices=None):
        if self.robot_type == 'pr2':
            indices_new = [79, 80, 81, 82] if left else [57, 58, 59, 60]
            positions = [position]*len(indices_new)
        elif self.robot_type == 'baxter':
            indices_new = [49, 51] if left else [27, 29]
            positions = [position, -position]
        elif self.robot_type == 'sawyer':
            indices_new = [20, 22]
            positions = [position, -position]
        elif self.robot_type == 'jaco':
            indices_new = [9, 11, 13]
            positions = [position, position, position]
        if indices is None:
            indices = indices_new

        if set_instantly:
            for i, j in enumerate(indices):
                p.resetJointState(robot, jointIndex=j, targetValue=positions[i], targetVelocity=0, physicsClientId=self.id)
        p.setJointMotorControlArray(robot, jointIndices=indices, controlMode=p.POSITION_CONTROL, targetPositions=positions, positionGains=np.array([0.05]*len(indices)), forces=[500]*len(indices), physicsClientId=self.id)

    def init_tool(self, robot, mesh_scale=[1]*3, pos_offset=[0]*3, orient_offset=[0, 0, 0, 1], left=True, maximal=False, alpha=1.0):
        if left:
            gripper_pos, gripper_orient = p.getLinkState(robot, 76 if self.robot_type=='pr2' else 18 if self.robot_type=='sawyer' else 47 if self.robot_type=='baxter' else 8, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        else:
            gripper_pos, gripper_orient = p.getLinkState(robot, 54 if self.robot_type=='pr2' else 18 if self.robot_type=='sawyer' else 25 if self.robot_type=='baxter' else 8, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        transform_pos, transform_orient = p.multiplyTransforms(positionA=gripper_pos, orientationA=gripper_orient, positionB=pos_offset, orientationB=orient_offset, physicsClientId=self.id)
        if self.task == 'scratch_itch':
            tool = p.loadURDF(os.path.join(self.directory, 'scratcher', 'tool_scratch.urdf'), basePosition=transform_pos, baseOrientation=transform_orient, physicsClientId=self.id)
        elif self.task == 'bed_bathing':
            tool = p.loadURDF(os.path.join(self.directory, 'bed_bathing', 'wiper.urdf'), basePosition=transform_pos, baseOrientation=transform_orient, physicsClientId=self.id)
        elif self.task == 'drinking':
            tool = p.loadURDF(os.path.join(self.directory, 'dinnerware', 'cup.urdf'), basePosition=transform_pos, baseOrientation=transform_orient, physicsClientId=self.id)
        elif self.task in ['scooping', 'feeding']:
            tool = p.loadURDF(os.path.join(self.directory, 'dinnerware', 'spoon.urdf'), basePosition=transform_pos, baseOrientation=transform_orient, useMaximalCoordinates=maximal, physicsClientId=self.id)
        elif self.task == 'arm_manipulation':
            visual_filename = os.path.join(self.directory, 'arm_manipulation', 'arm_manipulation_scooper.obj')
            collision_filename = os.path.join(self.directory, 'arm_manipulation', 'arm_manipulation_scooper_vhacd.obj')
            tool_visual = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=visual_filename, meshScale=mesh_scale, rgbaColor=[1, 1, 1, alpha], physicsClientId=self.id)
            tool_collision = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=collision_filename, meshScale=mesh_scale, physicsClientId=self.id)
            tool = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=tool_collision, baseVisualShapeIndex=tool_visual, basePosition=transform_pos, baseOrientation=transform_orient, useMaximalCoordinates=maximal, physicsClientId=self.id)
        if left:
            # Disable collisions between the tool and robot
            for j in (range(71, 86) if self.robot_type == 'pr2' else [18, 20, 21, 22, 23] if self.robot_type == 'sawyer' else [47, 49, 50, 51, 52] if self.robot_type == 'baxter' else [7, 8, 9, 10, 11, 12, 13, 14]):
                for tj in list(range(p.getNumJoints(tool, physicsClientId=self.id))) + [-1]:
                    p.setCollisionFilterPair(robot, tool, j, tj, False, physicsClientId=self.id)
            # Create constraint that keeps the tool in the gripper
            constraint = p.createConstraint(robot, 76 if self.robot_type=='pr2' else 18 if self.robot_type=='sawyer' else 47 if self.robot_type=='baxter' else 8, tool, -1, p.JOINT_FIXED, [0, 0, 0], parentFramePosition=pos_offset, childFramePosition=[0, 0, 0], parentFrameOrientation=orient_offset, physicsClientId=self.id)
        else:
            # Disable collisions between the tool and robot
            for j in (range(49, 64) if self.robot_type == 'pr2' else [18, 20, 21, 22, 23] if self.robot_type == 'sawyer' else [25, 27, 28, 29, 30] if self.robot_type == 'baxter' else [7, 8, 9, 10, 11, 12, 13, 14]):
                for tj in list(range(p.getNumJoints(tool, physicsClientId=self.id))) + [-1]:
                    p.setCollisionFilterPair(robot, tool, j, tj, False, physicsClientId=self.id)
            # Create constraint that keeps the tool in the gripper
            constraint = p.createConstraint(robot, 54 if self.robot_type=='pr2' else 18 if self.robot_type=='sawyer' else 25 if self.robot_type=='baxter' else 8, tool, -1, p.JOINT_FIXED, [0, 0, 0], parentFramePosition=pos_offset, childFramePosition=[0, 0, 0], parentFrameOrientation=orient_offset, physicsClientId=self.id)
        p.changeConstraint(constraint, maxForce=500, physicsClientId=self.id)
        return tool

    def setup_robot_joints(self, robot, robot_joint_indices, lower_limits, upper_limits, randomize_joint_positions=False, default_positions=[1, 1, 0, -1.75, 0, -1.1, -0.5], tool=None):
        if randomize_joint_positions:
            # Randomize arm joint positions
            # Keep trying random joint positions until the end effector is not colliding with anything
            retry = True
            while retry:
                for j, lower_limit, upper_limit in zip(robot_joint_indices, lower_limits, upper_limits):
                    if lower_limit == -1e10:
                        lower_limit = -np.pi
                        upper_limit = np.pi
                    joint_range = upper_limit - lower_limit
                    p.resetJointState(robot, jointIndex=j, targetValue=self.np_random.uniform(lower_limit + joint_range/6.0, upper_limit - joint_range/6.0), targetVelocity=0, physicsClientId=self.id)
                p.stepSimulation(physicsClientId=self.id)
                retry = len(p.getContactPoints(bodyA=robot, physicsClientId=self.id)) > 0
                if tool is not None:
                    retry = retry or (len(p.getContactPoints(bodyA=tool, physicsClientId=self.id)) > 0)
        else:
            default_positions[default_positions < lower_limits] = lower_limits[default_positions < lower_limits]
            default_positions[default_positions > upper_limits] = upper_limits[default_positions > upper_limits]
            for i, j in enumerate(robot_joint_indices):
                p.resetJointState(robot, jointIndex=j, targetValue=default_positions[i], targetVelocity=0, physicsClientId=self.id)

    def print_joint_info(self, body, show_fixed=True):
        joint_names = []
        for j in range(p.getNumJoints(body, physicsClientId=self.id)):
            if show_fixed or p.getJointInfo(body, j, physicsClientId=self.id)[2] != p.JOINT_FIXED:
                print(p.getJointInfo(body, j, physicsClientId=self.id))
                joint_names.append((j, p.getJointInfo(body, j, physicsClientId=self.id)[1]))
        print(joint_names)
