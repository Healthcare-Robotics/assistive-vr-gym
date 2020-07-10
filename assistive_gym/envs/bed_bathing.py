import os, sys, pickle, time, glob
from gym import spaces
import numpy as np
import pybullet as p
from datetime import datetime

from .env import AssistiveEnv

class BedBathingEnv(AssistiveEnv):
    def __init__(self, robot_type='pr2', human_control=False, vr=False, new=False):
        self.participant = -1
        self.gender = 'male'
        self.hipbone_to_mouth_height = 0.6
        self.policy_name = ''
        self.replay = False
        self.replay_dir = None
        self.human_gains, self.waist_gains, self.human_forces, self.waist_forces = 0.1, 0.1, 1.0, 4.0
        super(BedBathingEnv, self).__init__(robot_type=robot_type, task='bed_bathing', human_control=human_control, vr=vr, new=new, frame_skip=5, time_step=0.02, action_robot_len=7, action_human_len=(10 if human_control else 0), obs_robot_len=24, obs_human_len=(28 if human_control else 0))

    def setup(self, gender, participant, policy_name, hipbone_to_mouth_height):
        self.gender = gender
        self.participant = participant
        self.policy_name = policy_name
        if hipbone_to_mouth_height is None:
            self.hipbone_to_mouth_height = self.calc_hipbone_to_mouth_height()
            time.sleep(5)
        else:
            self.calc_hipbone_to_mouth_height()
            self.hipbone_to_mouth_height = hipbone_to_mouth_height

    def step(self, action):
        if self.replay:
            if self.last_sim_time is None:
                self.last_sim_time = time.time()
            for frame in range(self.frame_skip):
                p.restoreState(fileName=os.path.join(self.replay_dir, 'frame_%d.bullet' % (self.iteration*self.frame_skip + frame + 1)))
                # Slow down time so that the simulation matches real time
                self.slow_time()
            action = self.action_list[self.iteration]
            self.iteration += 1
        else:
            if len(action) < self.action_robot_len + self.action_human_len and self.participant >= 0:
                self.free_move(robot_arm='left', gains=self.config('robot_gains'), forces=self.config('robot_forces'))
                obs = self._get_obs([0], [0, 0])
                return obs, 0, False, dict()
            self.take_step(action, robot_arm='left', gains=self.config('robot_gains'), forces=self.config('robot_forces'), human_gains=0.05)
            self.action_list.append(action)
            if self.vr and self.participant >= 0:
                if self.iteration == 200:
                    # End of simulation, save action_list
                    with open(os.path.join(self.directory, 'actions.pkl'), 'wb') as f:
                        pickle.dump(self.action_list, f)

        total_force, tool_force, tool_force_on_human, total_force_on_human, new_contact_points = self.get_total_force()
        end_effector_velocity = np.linalg.norm(p.getLinkState(self.tool, 1, computeForwardKinematics=True, computeLinkVelocity=True, physicsClientId=self.id)[6])
        obs = self._get_obs([tool_force], [total_force_on_human, tool_force_on_human])

        # Get human preferences
        preferences_score = self.human_preferences(end_effector_velocity=end_effector_velocity, total_force_on_human=total_force_on_human, tool_force_at_target=tool_force_on_human)

        reward_distance = -min([c[8] for c in p.getClosestPoints(self.tool, self.human, distance=4.0, physicsClientId=self.id)])
        reward_action = -np.sum(np.square(action))  # Penalize actions
        reward_new_contact_points = new_contact_points  # Reward new contact points on a person

        reward = self.config('distance_weight')*reward_distance + self.config('action_weight')*reward_action + self.config('wiping_reward_weight')*reward_new_contact_points + preferences_score

        if self.gui and tool_force_on_human > 0 and not self.vr:
            print('Task success:', self.task_success, 'Force at tool on human:', tool_force_on_human, reward_new_contact_points)

        info = {'total_force_on_human': total_force_on_human, 'task_success': int(self.task_success >= (self.total_target_count*self.config('task_success_threshold'))), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = False

        if self.replay and self.iteration == 200:
            done = True
        return obs, reward, done, info

    def get_total_force(self):
        total_force = 0
        tool_force = 0
        tool_force_on_human = 0
        total_force_on_human = 0
        new_contact_points = 0
        for c in p.getContactPoints(bodyA=self.tool, physicsClientId=self.id):
            total_force += c[9]
            tool_force += c[9]
        for c in p.getContactPoints(bodyA=self.robot, physicsClientId=self.id):
            bodyB = c[2]
            if bodyB != self.tool:
                total_force += c[9]
        for c in p.getContactPoints(bodyA=self.robot, bodyB=self.human, physicsClientId=self.id):
            total_force_on_human += c[9]
        for c in p.getContactPoints(bodyA=self.tool, bodyB=self.human, physicsClientId=self.id):
            linkA = c[3]
            linkB = c[4]
            contact_position = np.array(c[6])
            total_force_on_human += c[9]
            if linkA in [1]:
                tool_force_on_human += c[9]
                # Contact with human upperarm, forearm, hand
                if linkB < 0 or linkB > p.getNumJoints(self.human, physicsClientId=self.id):
                    continue

                indices_to_delete = []
                for i, (target_pos_world, target) in enumerate(zip(self.targets_pos_upperarm_world, self.targets_upperarm)):
                    if np.linalg.norm(contact_position - target_pos_world) < 0.025:
                        # The robot made contact with a point on the person's arm
                        new_contact_points += 1
                        self.task_success += 1
                        p.resetBasePositionAndOrientation(target, [1000, 1000, 1000], [0, 0, 0, 1], physicsClientId=self.id)
                        indices_to_delete.append(i)
                self.targets_pos_on_upperarm = [t for i, t in enumerate(self.targets_pos_on_upperarm) if i not in indices_to_delete]
                self.targets_upperarm = [t for i, t in enumerate(self.targets_upperarm) if i not in indices_to_delete]
                self.targets_pos_upperarm_world = [t for i, t in enumerate(self.targets_pos_upperarm_world) if i not in indices_to_delete]

                indices_to_delete = []
                for i, (target_pos_world, target) in enumerate(zip(self.targets_pos_forearm_world, self.targets_forearm)):
                    if np.linalg.norm(contact_position - target_pos_world) < 0.025:
                        # The robot made contact with a point on the person's arm
                        new_contact_points += 1
                        self.task_success += 1
                        p.resetBasePositionAndOrientation(target, [1000, 1000, 1000], [0, 0, 0, 1], physicsClientId=self.id)
                        indices_to_delete.append(i)
                self.targets_pos_on_forearm = [t for i, t in enumerate(self.targets_pos_on_forearm) if i not in indices_to_delete]
                self.targets_forearm = [t for i, t in enumerate(self.targets_forearm) if i not in indices_to_delete]
                self.targets_pos_forearm_world = [t for i, t in enumerate(self.targets_pos_forearm_world) if i not in indices_to_delete]

        return total_force, tool_force, tool_force_on_human, total_force_on_human, new_contact_points

    def _get_obs(self, forces, forces_human):
        torso_pos = np.array(p.getLinkState(self.robot, 15 if self.robot_type == 'pr2' else 0, computeForwardKinematics=True, physicsClientId=self.id)[0])
        state = p.getLinkState(self.tool, 1, computeForwardKinematics=True, physicsClientId=self.id)
        tool_pos = np.array(state[0])
        tool_orient = np.array(state[1])  # Quaternions
        robot_joint_states = p.getJointStates(self.robot, jointIndices=self.robot_left_arm_joint_indices, physicsClientId=self.id)
        robot_joint_positions = np.array([x[0] for x in robot_joint_states])
        robot_pos, robot_orient = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.id)
        if self.human_control:
            human_pos = np.array(p.getLinkState(self.human, 3, computeForwardKinematics=True, physicsClientId=self.id)[:2][0])
            human_joint_states = p.getJointStates(self.human, jointIndices=list(range(4, 14)), physicsClientId=self.id)
            human_joint_positions = np.array([x[0] for x in human_joint_states])

        # Human shoulder, elbow, and wrist joint locations
        shoulder_pos, shoulder_orient = p.getLinkState(self.human, 9, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        elbow_pos, elbow_orient = p.getLinkState(self.human, 11, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        wrist_pos, wrist_orient = p.getLinkState(self.human, 13, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        
        robot_obs = np.concatenate([tool_pos-torso_pos, tool_orient, robot_joint_positions, shoulder_pos-torso_pos, elbow_pos-torso_pos, wrist_pos-torso_pos, forces]).ravel()
        if self.human_control:
            human_obs = np.concatenate([tool_pos-human_pos, tool_orient, human_joint_positions, shoulder_pos-human_pos, elbow_pos-human_pos, wrist_pos-human_pos, forces_human]).ravel()
        else:
            human_obs = []

        return np.concatenate([robot_obs, human_obs]).ravel()

    def reset(self):
        p.setRealTimeSimulation(enableRealTimeSimulation=0, physicsClientId=self.id)

        if self.vr and self.participant >= 0:
            self.directory = os.path.join(os.getcwd(), 'participant_%d' % self.participant, 'bed_bathing_vr_data_' + self.robot_type + '_' + self.policy_name + ('_participant_%d' % self.participant) + datetime.now().strftime('_%Y-%m-%d_%H-%M-%S'))
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)
            self.save_filename =  os.path.join(self.directory, 'frame_%d.bullet')

        self.action_list = []
        if self.replay:
            with open(os.path.join(self.replay_dir, 'setup.pkl'), 'rb') as f:
                self.robot_type, self.gender, self.hipbone_to_mouth_height = pickle.load(f)
            with open(os.path.join(self.replay_dir, 'actions.pkl'), 'rb') as f:
                self.action_list = pickle.load(f)

        self.setup_timing()
        self.task_success = 0
        self.contact_points_on_arm = {}

        if self.vr or self.replay:
            if self.vr:
                bed_to_hmd_height = self.hipbone_to_mouth_height + (0.068 + 0.117 if self.gender == 'male' else 0.058 + 0.094)
                p.setOriginCameraPositionAndOrientation(deviceTypeFilter=p.VR_DEVICE_HMD, pos_offset=[0, 0.46-bed_to_hmd_height*np.cos(np.pi/3.), -(0.7-(0.117 if self.gender == 'male' else 0.094) + bed_to_hmd_height*np.sin(np.pi/3.))], orn_offset=[0, 0, 0, 1])
            self.human, self.left_arm, self.right_arm, self.bed, self.robot, self.robot_lower_limits, self.robot_upper_limits, self.human_lower_limits, self.human_upper_limits, self.robot_right_arm_joint_indices, self.robot_left_arm_joint_indices, self.gender = self.world_creation.create_new_world(furniture_type='bed', static_human_base=True, human_impairment='random', print_joints=False, gender=self.gender, hipbone_to_mouth_height=self.hipbone_to_mouth_height)
        else:
            if self.participant < 0:
                self.gender = self.np_random.choice(['male', 'female'])
            if self.new:
                self.hipbone_to_mouth_height = self.np_random.uniform(0.6 - 0.1, 0.6 + 0.1) if self.gender == 'male' else self.np_random.uniform(0.54 - 0.1, 0.54 + 0.1)
                self.human, self.left_arm, self.right_arm, self.bed, self.robot, self.robot_lower_limits, self.robot_upper_limits, self.human_lower_limits, self.human_upper_limits, self.robot_right_arm_joint_indices, self.robot_left_arm_joint_indices, self.gender = self.world_creation.create_new_world(furniture_type='bed', static_human_base=True, human_impairment='none', print_joints=False, gender=self.gender, hipbone_to_mouth_height=self.hipbone_to_mouth_height)
            else:
                self.hipbone_to_mouth_height = 0.6 if self.gender == 'male' else 0.54          
                self.human, self.left_arm, self.right_arm, self.bed, self.robot, self.robot_lower_limits, self.robot_upper_limits, self.human_lower_limits, self.human_upper_limits, self.robot_right_arm_joint_indices, self.robot_left_arm_joint_indices, self.gender = self.world_creation.create_new_world(furniture_type='bed', static_human_base=True, human_impairment='none', print_joints=False, gender=self.gender, hipbone_to_mouth_height=self.hipbone_to_mouth_height)

        self.robot_lower_limits = self.robot_lower_limits[self.robot_left_arm_joint_indices]
        self.robot_upper_limits = self.robot_upper_limits[self.robot_left_arm_joint_indices]
        self.reset_robot_joints()

        p.resetBasePositionAndOrientation(self.human, [0, 0, 0.7], p.getQuaternionFromEuler([np.deg2rad(-30), 0, 0], physicsClientId=self.id), physicsClientId=self.id)
        if self.vr or self.replay:
            left_shoulder_pos, left_shoulder_orient = p.getLinkState(self.human, 16, computeForwardKinematics=True, physicsClientId=self.id)[:2]
            p.resetBasePositionAndOrientation(self.left_arm, left_shoulder_pos, left_shoulder_orient, physicsClientId=self.id)
            right_shoulder_pos, right_shoulder_orient = p.getLinkState(self.human, 6, computeForwardKinematics=True, physicsClientId=self.id)[:2]
            p.resetBasePositionAndOrientation(self.right_arm, right_shoulder_pos, right_shoulder_orient, physicsClientId=self.id)

        # Create the mattress
        p.removeBody(self.bed, physicsClientId=self.id)
        y_offset = -0.53
        self.bed_parts = []
        mattress_collision = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[0.88/2.0, 1.25/2.0, 0.15/2.0], collisionFramePosition=[0, 0, 0.15/2.0], physicsClientId=self.id)
        mattress_visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[0.88/2.0, 1.25/2.0, 0.15/2.0], visualFramePosition=[0, 0, 0.15/2.0], rgbaColor=[1, 1, 1, 1], physicsClientId=self.id)
        self.bed_parts.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=mattress_collision, baseVisualShapeIndex=mattress_visual, basePosition=[0, y_offset, 0.4], useMaximalCoordinates=False, physicsClientId=self.id))

        mattress_collision = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[0.88/2.0, 0.7/2.0, 0.15/2.0], collisionFramePosition=[0, 0.7/2.0, 0], physicsClientId=self.id)
        mattress_visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[0.88/2.0, 0.7/2.0, 0.15/2.0], visualFramePosition=[0, 0.7/2.0, 0], rgbaColor=[1, 1, 1, 1], physicsClientId=self.id)
        self.bed_parts.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=mattress_collision, baseVisualShapeIndex=mattress_visual, basePosition=[0, 1.25/2.0+y_offset, 0.4+0.15/2.0], baseOrientation=p.getQuaternionFromEuler([np.deg2rad(60), 0, 0], physicsClientId=self.id), useMaximalCoordinates=False, physicsClientId=self.id))

        # Create the bed frame
        visual_filename = os.path.join(self.world_creation.directory, 'bed', 'hospital_bed_frame_reduced.obj')
        collision_filename = os.path.join(self.world_creation.directory, 'bed', 'hospital_bed_frame_vhacd.obj')
        bed_collision = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=collision_filename, meshScale=[1, 1.2, 1], physicsClientId=self.id)
        bed_visual = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=visual_filename, meshScale=[1, 1.2, 1], rgbaColor=[1, 1, 1, 1], physicsClientId=self.id)
        self.bed_parts.append(p.createMultiBody(baseMass=0, baseCollisionShapeIndex=bed_collision, baseVisualShapeIndex=bed_visual, basePosition=[0, y_offset + 0.45, 0.42], baseOrientation=p.getQuaternionFromEuler([np.pi/2.0, 0, -np.pi/2.0], physicsClientId=self.id), useMaximalCoordinates=False, physicsClientId=self.id))

        p.resetDebugVisualizerCamera(cameraDistance=1.10, cameraYaw=40, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.75], physicsClientId=self.id)

        # Disable collisions between the person's hips/legs and the mattress/frame
        for bed_part in self.bed_parts:
            for i in list(range(28, 42)) + [0, 1, 2, 3]:
                p.setCollisionFilterPair(self.human, bed_part, i, -1, 0, physicsClientId=self.id)

        # Continually resample random human pose until no contact with robot or environment is occurring
        if self.vr or self.replay:
            if self.new:
                joints_positions = [(7, np.deg2rad(20)), (8, np.deg2rad(-20)), (10, np.deg2rad(-45)), (20, np.deg2rad(-45)), (28, np.deg2rad(-60)), (35, np.deg2rad(-60))]
            else:
                joint_angles = [0.39717707, 0.27890519, -0.00883447, -0.67345593, -0.00568484, 0.05987911, 0.00957937]
                joints_positions = [(i, joint_angles[i]) for i in range(7)]
                joints_positions += [(13, np.deg2rad(-30)), (28, np.deg2rad(-60)), (35, np.deg2rad(-60))]

            self.human_controllable_joint_indices = [0, 1, 2, 25, 26, 27] + list(range(7, 14)) + list(range(17, 24))
            self.world_creation.setup_human_joints(self.human, joints_positions, self.human_controllable_joint_indices, use_static_joints=True, human_reactive_force=None)
            human_joint_states = p.getJointStates(self.human, jointIndices=list(range(4, 14)), physicsClientId=self.id)
            self.target_human_joint_positions = np.array([x[0] for x in human_joint_states])

            events = p.getVREvents(deviceTypeFilter=p.VR_DEVICE_HMD+p.VR_DEVICE_CONTROLLER, physicsClientId=self.id)
            if len(events)==3:
                self.arm_sim(events[1], "right")
                self.arm_sim(events[1], "left")

            left_arm_joint_states = p.getJointStates(self.human, jointIndices=list(range(17, 24)), physicsClientId=self.id)
            left_arm_joint_positions = np.array([x[0] for x in left_arm_joint_states])
            for i in range(7):
                p.resetJointState(self.left_arm, jointIndex=i, targetValue=left_arm_joint_positions[i], targetVelocity=0, physicsClientId=self.id)
            
            right_arm_joint_states = p.getJointStates(self.human, jointIndices=list(range(7, 14)), physicsClientId=self.id)
            right_arm_joint_positions = np.array([x[0] for x in right_arm_joint_states])
            for i in range(7):
                p.resetJointState(self.right_arm, jointIndex=i, targetValue=right_arm_joint_positions[i], targetVelocity=0, physicsClientId=self.id)
        else:
            if self.new:
                self.human_controllable_joint_indices = list(range(4, 14))
                human_positioned = False
                right_arm_links = [9, 11, 13]
                left_arm_links = [19, 21, 23]
                r_arm_dists = []
                right_arm_shoulder_links = [6, 9, 11, 13]
                left_arm_shoulder_links = [16, 19, 21, 23]
                r_arm_bed_dists = []
                l_arm_bed_dists = []

                while not human_positioned or (np.min(r_arm_dists + r_arm_bed_dists + l_arm_bed_dists + [1]) < 0.01):
                    human_positioned = True
                    joints_positions = [(7, np.deg2rad(20)), (8, np.deg2rad(-20)), (10, np.deg2rad(-45)), (20, np.deg2rad(-45)), (28, np.deg2rad(-60)), (35, np.deg2rad(-60))]
                    joints_positions += [(i, 0) for i in list(range(7, 14)) + list(range(17, 24))]
                    joints_positions += [(i, self.np_random.uniform(np.deg2rad(-10), np.deg2rad(10))) for i in [0, 1, 2]]
                    self.world_creation.setup_human_joints(self.human, joints_positions, self.human_controllable_joint_indices if (self.human_control or self.world_creation.human_impairment == 'tremor') else [], use_static_joints=True, human_reactive_force=None, non_static_joints=(list(range(4, 24)) if self.human_control else []))
                    joints_positions = [(i, self.np_random.uniform(np.deg2rad(-10), np.deg2rad(10))) for i in list(range(7, 14))]
                    self.world_creation.setup_human_joints(self.human, joints_positions, self.human_controllable_joint_indices if (self.human_control or self.world_creation.human_impairment == 'tremor') else [], use_static_joints=False, human_reactive_force=None, add_joint_positions=True)
                    r_arm_dists = [c[8] for link in right_arm_links for c in p.getClosestPoints(bodyA=self.human, bodyB=self.human, linkIndexA=link, distance=0.05, physicsClientId=self.id) if c[4] not in (right_arm_links + [3, 6])]
                    r_arm_bed_dists = [c[8] for link in right_arm_shoulder_links for bed_part in self.bed_parts for c in p.getClosestPoints(bodyA=self.human, bodyB=bed_part, linkIndexA=link, distance=0.05, physicsClientId=self.id)]
                    l_arm_bed_dists = [c[8] for link in left_arm_shoulder_links for bed_part in self.bed_parts for c in p.getClosestPoints(bodyA=self.human, bodyB=bed_part, linkIndexA=link, distance=0.05, physicsClientId=self.id)]
                human_joint_states = p.getJointStates(self.human, jointIndices=self.human_controllable_joint_indices, physicsClientId=self.id)
                self.target_human_joint_positions = np.array([x[0] for x in human_joint_states])
            else:
                for bed_parts in self.bed_parts:
                    p.changeDynamics(bed_parts, -1, lateralFriction=5, spinningFriction=5, rollingFriction=5, physicsClientId=self.id)
                joints_positions = [(7, np.deg2rad(50)), (8, np.deg2rad(-50)), (17, np.deg2rad(-30)), (28, np.deg2rad(-60)), (35, np.deg2rad(-60))]
                self.world_creation.setup_human_joints(self.human, joints_positions, [], use_static_joints=True, human_reactive_force=None, non_static_joints=(list(range(4, 14))))
     
                p.setGravity(0, 0, -1, physicsClientId=self.id)
                # Let the arm settle on the bed
                for _ in range(100):
                    p.stepSimulation(physicsClientId=self.id)

                self.human_controllable_joint_indices = list(range(4, 14)) if self.human_control else []
                self.world_creation.setup_human_joints(self.human, [], self.human_controllable_joint_indices, use_static_joints=True, human_reactive_force=None, human_reactive_gain=0.01)
                self.target_human_joint_positions = []

                if self.human_control:
                    human_joint_states = p.getJointStates(self.human, jointIndices=self.human_controllable_joint_indices, physicsClientId=self.id)
                    self.target_human_joint_positions = np.array([x[0] for x in human_joint_states])

                p.changeDynamics(self.human, -1, mass=0, physicsClientId=self.id)
                p.resetBaseVelocity(self.human, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0], physicsClientId=self.id)

        self.human_lower_limits = self.human_lower_limits[self.human_controllable_joint_indices]
        self.human_upper_limits = self.human_upper_limits[self.human_controllable_joint_indices]

        shoulder_pos, shoulder_orient = p.getLinkState(self.human, 9, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        elbow_pos, elbow_orient = p.getLinkState(self.human, 11, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        wrist_pos, wrist_orient = p.getLinkState(self.human, 13, computeForwardKinematics=True, physicsClientId=self.id)[:2]

        if self.vr or self.replay:
            human_joint_indices = list(range(4, 14))
        else:
            human_joint_indices = self.human_controllable_joint_indices
        
        target_pos = np.array([-0.5, -0.1, 1])
        if self.robot_type == 'pr2':
            target_orient = np.array(p.getQuaternionFromEuler(np.array([0, 0, 0]), physicsClientId=self.id))
            _, _, self.target_robot_joint_positions = self.position_robot_toc(self.robot, 76, [(target_pos, target_orient)], [(shoulder_pos, None), (elbow_pos, None), (wrist_pos, None)], self.robot_left_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=range(29, 29+7), pos_offset=np.array([0, 0, 0]), max_ik_iterations=200, step_sim=True, check_env_collisions=False, human_joint_indices=human_joint_indices, human_joint_positions=self.target_human_joint_positions)
            self.target_robot_joint_positions = self.target_robot_joint_positions[0]
            self.world_creation.set_gripper_open_position(self.robot, position=0.2, left=True, set_instantly=True)
            self.tool = self.world_creation.init_tool(self.robot, mesh_scale=[1]*3, pos_offset=[0, 0, 0], orient_offset=p.getQuaternionFromEuler([0, 0, 0], physicsClientId=self.id), maximal=False)
        elif self.robot_type == 'jaco':
            target_orient = p.getQuaternionFromEuler(np.array([0, np.pi/2.0, 0]), physicsClientId=self.id)
            base_position, _, self.target_robot_joint_positions = self.position_robot_toc(self.robot, 8, [(target_pos, target_orient)], [(shoulder_pos, None), (elbow_pos, None), (wrist_pos, None)], self.robot_left_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=[0, 1, 2, 3, 4, 5, 6], pos_offset=np.array([0.1, 0.55, 0.6]), max_ik_iterations=200, step_sim=True, random_position=0.1, check_env_collisions=False, human_joint_indices=human_joint_indices, human_joint_positions=self.target_human_joint_positions)
            self.target_robot_joint_positions = self.target_robot_joint_positions[0]
            self.world_creation.set_gripper_open_position(self.robot, position=1.1, left=True, set_instantly=True)
            self.tool = self.world_creation.init_tool(self.robot, mesh_scale=[1]*3, pos_offset=[-0.01, 0, 0.03], orient_offset=p.getQuaternionFromEuler([0, -np.pi/2.0, 0], physicsClientId=self.id), maximal=False)
            # Load a nightstand in the environment for the jaco arm
            self.nightstand_scale = 0.275
            visual_filename = os.path.join(self.world_creation.directory, 'nightstand', 'nightstand.obj')
            collision_filename = os.path.join(self.world_creation.directory, 'nightstand', 'nightstand.obj')
            nightstand_visual = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=visual_filename, meshScale=[self.nightstand_scale]*3, rgbaColor=[0.5, 0.5, 0.5, 1.0], physicsClientId=self.id)
            nightstand_collision = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=collision_filename, meshScale=[self.nightstand_scale]*3, physicsClientId=self.id)
            nightstand_pos = np.array([-0.85, 0.12, 0]) + base_position
            nightstand_orient = p.getQuaternionFromEuler(np.array([np.pi/2.0, 0, 0]), physicsClientId=self.id)
            self.nightstand = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=nightstand_collision, baseVisualShapeIndex=nightstand_visual, basePosition=nightstand_pos, baseOrientation=nightstand_orient, baseInertialFramePosition=[0, 0, 0], useMaximalCoordinates=False, physicsClientId=self.id)

        self.generate_targets()
        p.setPhysicsEngineParameter(numSubSteps=0, numSolverIterations=50, physicsClientId=self.id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        p.setGravity(0, 0, 0, body=self.robot, physicsClientId=self.id)
        p.setGravity(0, 0, 0, body=self.human, physicsClientId=self.id)
        p.setGravity(0, 0, 0, body=self.tool, physicsClientId=self.id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        if self.replay:
            print(os.path.join(self.replay_dir, 'frame_0.bullet'))
            p.restoreState(fileName=os.path.join(self.replay_dir, 'frame_0.bullet'))

        obs = self._get_obs([0], [0, 0])
        if self.vr and self.participant >= 0:
            p.saveBullet(self.save_filename % 0)
            with open(os.path.join(self.directory, 'setup.pkl'), 'wb') as f:
                pickle.dump([self.robot_type, self.gender, self.hipbone_to_mouth_height], f)

        return obs

    def generate_targets(self):
        self.target_indices_to_ignore = []
        if self.gender == 'male':
            hmhs = self.hipbone_to_mouth_height / 0.6
            self.upperarm, self.upperarm_length, self.upperarm_radius = 9, 0.279, 0.043
            self.forearm, self.forearm_length, self.forearm_radius = 11, 0.257, 0.033
        else:
            hmhs = self.hipbone_to_mouth_height / 0.54
            self.upperarm, self.upperarm_length, self.upperarm_radius = 9, 0.264, 0.0355
            self.forearm, self.forearm_length, self.forearm_radius = 11, 0.234, 0.027
        self.targets_pos_on_upperarm = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.upperarm_length]), radius=self.upperarm_radius, distance_between_points=0.03, position_scale=hmhs)
        self.targets_pos_on_forearm = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.forearm_length]), radius=self.forearm_radius, distance_between_points=0.03, position_scale=hmhs)
        sphere_collision = -1
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 1, 1], physicsClientId=self.id)
        self.targets_upperarm = []
        self.targets_forearm = []
        for _ in range(len(self.targets_pos_on_upperarm)):
            self.targets_upperarm.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=[0, 0, 0], useMaximalCoordinates=False, physicsClientId=self.id))
        for _ in range(len(self.targets_pos_on_forearm)):
            self.targets_forearm.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=[0, 0, 0], useMaximalCoordinates=False, physicsClientId=self.id))
        self.total_target_count = len(self.targets_upperarm) + len(self.targets_forearm)
        self.update_targets()

    def update_targets(self):
        upperarm_pos, upperarm_orient = p.getLinkState(self.human, self.upperarm, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        self.targets_pos_upperarm_world = []
        for target_pos_on_arm, target in zip(self.targets_pos_on_upperarm, self.targets_upperarm):
            target_pos = np.array(p.multiplyTransforms(upperarm_pos, upperarm_orient, target_pos_on_arm, [0, 0, 0, 1], physicsClientId=self.id)[0])
            self.targets_pos_upperarm_world.append(target_pos)
            p.resetBasePositionAndOrientation(target, target_pos, [0, 0, 0, 1], physicsClientId=self.id)
        forearm_pos, forearm_orient = p.getLinkState(self.human, self.forearm, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        self.targets_pos_forearm_world = []
        for target_pos_on_arm, target in zip(self.targets_pos_on_forearm, self.targets_forearm):
            target_pos = np.array(p.multiplyTransforms(forearm_pos, forearm_orient, target_pos_on_arm, [0, 0, 0, 1], physicsClientId=self.id)[0])
            self.targets_pos_forearm_world.append(target_pos)
            p.resetBasePositionAndOrientation(target, target_pos, [0, 0, 0, 1], physicsClientId=self.id)