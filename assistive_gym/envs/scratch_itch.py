import os, sys, pickle, time, glob
from gym import spaces
import numpy as np
import pybullet as p
from datetime import datetime

from .env import AssistiveEnv

class ScratchItchEnv(AssistiveEnv):
    def __init__(self, robot_type='pr2', human_control=False, vr=False, new=False):
        self.participant = -1
        self.gender = 'male'
        self.hipbone_to_mouth_height = 0.6
        self.policy_name = ''
        self.replay = False
        self.replay_dir = None 
        self.human_gains, self.waist_gains, self.human_forces, self.waist_forces = 0.09, 0.09, 1.0, 4.0
        super(ScratchItchEnv, self).__init__(robot_type=robot_type, task='scratch_itch', human_control=human_control, vr=vr, new=new, frame_skip=5, time_step=0.02, action_robot_len=7, action_human_len=(10 if human_control else 0), obs_robot_len=30, obs_human_len=(34 if human_control else 0))

    def setup(self, gender, participant, policy_name, hipbone_to_mouth_height):
        self.gender = gender
        self.participant = participant
        self.policy_name = policy_name
        if hipbone_to_mouth_height is None:
            self.hipbone_to_mouth_height = self.calc_hipbone_to_mouth_height()
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

        total_force_on_human, tool_force, tool_force_at_target, target_contact_pos = self.get_total_force()
        end_effector_velocity = np.linalg.norm(p.getLinkState(self.tool, 1, computeForwardKinematics=True, computeLinkVelocity=True, physicsClientId=self.id)[6])
        if target_contact_pos is not None:
            target_contact_pos = np.array(target_contact_pos)
        obs = self._get_obs([tool_force], [total_force_on_human, tool_force_at_target])

        # Get human preferences
        preferences_score = self.human_preferences(end_effector_velocity=end_effector_velocity, total_force_on_human=total_force_on_human, tool_force_at_target=tool_force_at_target)

        tool_pos = np.array(p.getLinkState(self.tool, 1, computeForwardKinematics=True, physicsClientId=self.id)[0])
        reward_distance = -np.linalg.norm(self.target_pos - tool_pos)  # Penalize distances away from target
        reward_action = -np.sum(np.square(action))  # Penalize actions
        reward_force_scratch = 0.0  # Reward force near the target
        if target_contact_pos is not None and np.linalg.norm(target_contact_pos - self.prev_target_contact_pos) > 0.01 and tool_force_at_target < 10:
            # Encourage the robot to move around near the target to simulate scratching
            reward_force_scratch = tool_force_at_target
            self.prev_target_contact_pos = target_contact_pos
            self.task_success += 1

        reward = self.config('distance_weight')*reward_distance + self.config('action_weight')*reward_action + self.config('tool_force_weight')*tool_force_at_target + self.config('scratch_reward_weight')*reward_force_scratch + preferences_score

        if self.gui and tool_force_at_target > 0 and not self.vr:
            print('Task success:', self.task_success, 'Tool force at target:', tool_force_at_target, reward_force_scratch)

        info = {'total_force_on_human': total_force_on_human, 'task_success': int(self.task_success >= self.config('task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = False

        if self.replay and self.iteration == 200:
            done = True
        return obs, reward, done, info

    def get_total_force(self):
        total_force_on_human = 0
        tool_force = 0
        tool_force_at_target = 0
        target_contact_pos = None
        for c in p.getContactPoints(bodyA=self.tool, physicsClientId=self.id):
            tool_force += c[9]
        for c in p.getContactPoints(bodyA=self.tool, bodyB=self.human, physicsClientId=self.id):
            total_force_on_human += c[9]
            linkA = c[3]
            contact_position = c[6]
            if linkA in [0, 1]:
                # Enforce that contact is close to the target location
                if np.linalg.norm(contact_position - self.target_pos) < 0.025:
                    tool_force_at_target += c[9]
                    target_contact_pos = contact_position
        for c in p.getContactPoints(bodyA=self.robot, bodyB=self.human, physicsClientId=self.id):
            total_force_on_human += c[9]
        return total_force_on_human, tool_force, tool_force_at_target, target_contact_pos

    def _get_obs(self, forces, forces_human):
        torso_pos = np.array(p.getLinkState(self.robot, 15 if self.robot_type == 'pr2' else 0, computeForwardKinematics=True, physicsClientId=self.id)[0])
        state = p.getLinkState(self.tool, 1, computeForwardKinematics=True, physicsClientId=self.id)
        tool_pos = np.array(state[0])
        tool_orient = np.array(state[1])
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

        robot_obs = np.concatenate([tool_pos-torso_pos, tool_orient, tool_pos-self.target_pos, self.target_pos-torso_pos, robot_joint_positions, shoulder_pos-torso_pos, elbow_pos-torso_pos, wrist_pos-torso_pos, forces]).ravel()
        if self.human_control:
            human_obs = np.concatenate([tool_pos-human_pos, tool_orient, tool_pos-self.target_pos, self.target_pos-human_pos, human_joint_positions, shoulder_pos-human_pos, elbow_pos-human_pos, wrist_pos-human_pos, forces_human]).ravel()
        else:
            human_obs = []

        return np.concatenate([robot_obs, human_obs]).ravel()

    def reset(self):
        p.setRealTimeSimulation(enableRealTimeSimulation=0, physicsClientId=self.id)

        if self.vr and self.participant >= 0:
            self.directory = os.path.join(os.getcwd(), 'participant_%d' % self.participant, 'scratch_itch_vr_data_' + self.robot_type + '_' + self.policy_name + ('_participant_%d' % self.participant) + datetime.now().strftime('_%Y-%m-%d_%H-%M-%S'))
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
        self.prev_target_contact_pos = np.zeros(3)
        if self.vr or self.replay:
            if self.vr:
                seat_to_hmd_height = self.hipbone_to_mouth_height + (0.068 + 0.1335*self.config('radius_scale', 'human_female') if self.gender == 'male' else 0.058 + 0.127*self.config('radius_scale', 'human_female'))
                p.setOriginCameraPositionAndOrientation(deviceTypeFilter=p.VR_DEVICE_HMD, pos_offset=[0, -0.06, -(0.47 + seat_to_hmd_height)], orn_offset=[0, 0, 0, 1])
            self.human, self.left_arm, self.right_arm, self.wheelchair, self.robot, self.robot_lower_limits, self.robot_upper_limits, self.human_lower_limits, self.human_upper_limits, self.robot_right_arm_joint_indices, self.robot_left_arm_joint_indices, self.gender = self.world_creation.create_new_world(furniture_type='wheelchair', static_human_base=True, human_impairment='random', print_joints=False, gender=self.gender, hipbone_to_mouth_height=self.hipbone_to_mouth_height)
        else:
            if self.participant < 0:
                self.gender = self.np_random.choice(['male', 'female'])
            if self.new:
                self.hipbone_to_mouth_height = self.np_random.uniform(0.6 - 0.1, 0.6 + 0.1) if self.gender == 'male' else self.np_random.uniform(0.54 - 0.1, 0.54 + 0.1)
                self.human, self.left_arm, self.right_arm, self.wheelchair, self.robot, self.robot_lower_limits, self.robot_upper_limits, self.human_lower_limits, self.human_upper_limits, self.robot_right_arm_joint_indices, self.robot_left_arm_joint_indices, self.gender = self.world_creation.create_new_world(furniture_type='wheelchair', static_human_base=True, human_impairment='none', print_joints=False, gender=self.gender, hipbone_to_mouth_height=self.hipbone_to_mouth_height)
            else:
                self.hipbone_to_mouth_height = 0.6 if self.gender == 'male' else 0.54            
                self.human, self.left_arm, self.right_arm, self.wheelchair, self.robot, self.robot_lower_limits, self.robot_upper_limits, self.human_lower_limits, self.human_upper_limits, self.robot_right_arm_joint_indices, self.robot_left_arm_joint_indices, self.gender = self.world_creation.create_new_world(furniture_type='wheelchair', static_human_base=True, human_impairment='random', print_joints=False, gender=self.gender, hipbone_to_mouth_height=self.hipbone_to_mouth_height)

        self.robot_lower_limits = self.robot_lower_limits[self.robot_left_arm_joint_indices]
        self.robot_upper_limits = self.robot_upper_limits[self.robot_left_arm_joint_indices]
        self.reset_robot_joints()
        if self.robot_type == 'jaco':
            p.resetBasePositionAndOrientation(self.robot, [-0.35, -0.3, 0.36], [0.0, 0.0, -0.7071067811865475, 0.7071067811865476], physicsClientId=self.id)
            if self.new:
                target_pos = np.array([-0.5, 0, 0.8]) + self.np_random.uniform(-0.05, 0.05, size=3)
                target_orient = p.getQuaternionFromEuler(np.array([0, np.pi/2.0, 0]), physicsClientId=self.id)
                _, self.target_robot_joint_positions = self.util.ik_random_restarts(self.robot, 8, target_pos, target_orient, self.world_creation, self.robot_left_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=[0, 1, 2, 3, 4, 5, 6], max_iterations=1000, max_ik_random_restarts=40, random_restart_threshold=0.03, step_sim=True)
                self.world_creation.set_gripper_open_position(self.robot, position=1, left=True, set_instantly=True)
                self.tool = self.world_creation.init_tool(self.robot, mesh_scale=[0.001]*3, pos_offset=[0, 0, 0.02], orient_offset=p.getQuaternionFromEuler([0, -np.pi/2.0, 0], physicsClientId=self.id), maximal=False)

        if self.vr or self.replay:
            joints_positions = [(7, np.deg2rad(30)), (10, np.deg2rad(-90)), (20, np.deg2rad(-90)), (28, np.deg2rad(-90)), (31, np.deg2rad(80)), (35, np.deg2rad(-90)), (38, np.deg2rad(80))]
            self.human_controllable_joint_indices = [0, 1, 2, 25, 26, 27] + list(range(7, 14)) + list(range(17, 24))
            self.world_creation.setup_human_joints(self.human, joints_positions, self.human_controllable_joint_indices, use_static_joints=True, human_reactive_force=None)
            p.resetBasePositionAndOrientation(self.human, [0, 0.03, 0.89-0.23725 if self.gender == 'male' else 0.86-0.225], [0, 0, 0, 1], physicsClientId=self.id)
            human_joint_states = p.getJointStates(self.human, jointIndices=list(range(4, 14)), physicsClientId=self.id)

            left_shoulder_pos, left_shoulder_orient = p.getLinkState(self.human, 16, computeForwardKinematics=True, physicsClientId=self.id)[:2]
            p.resetBasePositionAndOrientation(self.left_arm, left_shoulder_pos, left_shoulder_orient, physicsClientId=self.id)
            left_arm_joint_states = p.getJointStates(self.human, jointIndices=list(range(17, 24)), physicsClientId=self.id)
            left_arm_joint_positions = np.array([x[0] for x in left_arm_joint_states])
            for i in range(7):
                p.resetJointState(self.left_arm, jointIndex=i, targetValue=left_arm_joint_positions[i], targetVelocity=0, physicsClientId=self.id)

            right_shoulder_pos, right_shoulder_orient = p.getLinkState(self.human, 6, computeForwardKinematics=True, physicsClientId=self.id)[:2]
            p.resetBasePositionAndOrientation(self.right_arm, right_shoulder_pos, right_shoulder_orient, physicsClientId=self.id)
            right_arm_joint_states = p.getJointStates(self.human, jointIndices=list(range(7, 14)), physicsClientId=self.id)
            right_arm_joint_positions = np.array([x[0] for x in right_arm_joint_states])
            for i in range(7):
                p.resetJointState(self.right_arm, jointIndex=i, targetValue=right_arm_joint_positions[i], targetVelocity=0, physicsClientId=self.id)
        else:
            self.human_controllable_joint_indices = list(range(4, 14))
            if self.new:
                p.resetBasePositionAndOrientation(self.human, [0, 0.03, 0.89-0.23725 if self.gender == 'male' else 0.86-0.225], [0, 0, 0, 1], physicsClientId=self.id)
                human_positioned = False 
                right_arm_links = [9, 11, 13]
                left_arm_links = [19, 21, 23]
                r_arm_dists = []
                l_arm_dists = []
                right_arm_shoulder_links = [6, 9, 11, 13]
                left_arm_shoulder_links = [16, 19, 21, 23]
                human_robot_dists = []
                r_arm_wheelchair_dists = []
                l_arm_wheelchair_dists = []

                while not human_positioned or (np.min(r_arm_dists + l_arm_dists + r_arm_wheelchair_dists + l_arm_wheelchair_dists + human_robot_dists + [1]) < 0.01):
                    human_positioned = True
                    joints_positions = [(7, np.deg2rad(30)), (10, np.deg2rad(-90)), (20, np.deg2rad(-90)), (28, np.deg2rad(-90)), (31, np.deg2rad(80)), (35, np.deg2rad(-90)), (38, np.deg2rad(80))]
                    joints_positions += [(i, 0) for i in list(range(7, 14)) + list(range(17, 24))]
                    joints_positions += [(i, self.np_random.uniform(np.deg2rad(-10), np.deg2rad(10))) for i in [0, 1, 2]]

                    self.world_creation.setup_human_joints(self.human, joints_positions, self.human_controllable_joint_indices, use_static_joints=True, human_reactive_force=None)

                    joints_positions = [(i, self.np_random.uniform(np.deg2rad(-10), np.deg2rad(10))) for i in list(range(7, 14))]
                    self.world_creation.setup_human_joints(self.human, joints_positions, self.human_controllable_joint_indices, use_static_joints=False, human_reactive_force=None, add_joint_positions=True)

                    r_arm_dists = [c[8] for link in right_arm_links for c in p.getClosestPoints(bodyA=self.human, bodyB=self.human, linkIndexA=link, distance=0.05, physicsClientId=self.id) if c[4] not in (right_arm_links + [3, 6])]
                    l_arm_dists = []
                    human_robot_dists = [c[8] for c in p.getClosestPoints(bodyA=self.human, bodyB=self.robot, distance=0.05, physicsClientId=self.id)]
                    r_arm_wheelchair_dists = [c[8] for link in right_arm_shoulder_links for c in p.getClosestPoints(bodyA=self.human, bodyB=self.wheelchair, linkIndexA=link, distance=0.05, physicsClientId=self.id)]
                    l_arm_wheelchair_dists = [c[8] for link in left_arm_shoulder_links for c in p.getClosestPoints(bodyA=self.human, bodyB=self.wheelchair, linkIndexA=link, distance=0.05, physicsClientId=self.id)]

                self.world_creation.setup_human_joints(self.human, [], self.human_controllable_joint_indices, use_static_joints=False, human_reactive_force=1, human_reactive_gain=0.01, add_joint_positions=True)
            else:
                joints_positions = [(7, np.deg2rad(30)), (10, np.deg2rad(-90)), (20, np.deg2rad(-90)), (28, np.deg2rad(-90)), (31, np.deg2rad(80)), (35, np.deg2rad(-90)), (38, np.deg2rad(80))]
                self.world_creation.setup_human_joints(self.human, joints_positions, self.human_controllable_joint_indices, use_static_joints=True, human_reactive_force=None if self.human_control else 1, human_reactive_gain=0.01)
                p.resetBasePositionAndOrientation(self.human, [0, 0.03, 0.89-0.23725 if self.gender == 'male' else 0.86-0.225], [0, 0, 0, 1], physicsClientId=self.id)
            human_joint_states = p.getJointStates(self.human, jointIndices=self.human_controllable_joint_indices, physicsClientId=self.id)

        self.target_human_joint_positions = np.array([x[0] for x in human_joint_states])
        self.human_lower_limits = self.human_lower_limits[self.human_controllable_joint_indices]
        self.human_upper_limits = self.human_upper_limits[self.human_controllable_joint_indices]

        shoulder_pos, shoulder_orient = p.getLinkState(self.human, 9, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        elbow_pos, elbow_orient = p.getLinkState(self.human, 11, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        wrist_pos, wrist_orient = p.getLinkState(self.human, 13, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        human_joint_indices = list(range(4, 14))
        if self.robot_type == 'pr2':
            target_pos = np.array([-0.55, 0, 0.8]) + self.np_random.uniform(-0.05, 0.05, size=3)
            target_orient = np.array(p.getQuaternionFromEuler(np.array([0, 0, 0]), physicsClientId=self.id))
            _, _, self.target_robot_joint_positions = self.position_robot_toc(self.robot, 76, [(target_pos, target_orient)], [(shoulder_pos, None), (elbow_pos, None), (wrist_pos, None)], self.robot_left_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=range(29, 29+7), pos_offset=np.array([0.1, 0, 0]), max_ik_iterations=200, step_sim=True, check_env_collisions=False, human_joint_indices=human_joint_indices, human_joint_positions=self.target_human_joint_positions)
            self.target_robot_joint_positions = self.target_robot_joint_positions[0]
            self.world_creation.set_gripper_open_position(self.robot, position=0.25, left=True, set_instantly=True)
            self.tool = self.world_creation.init_tool(self.robot, mesh_scale=[0.001]*3, pos_offset=[0, 0, 0], orient_offset=p.getQuaternionFromEuler([0, 0, 0], physicsClientId=self.id), maximal=False)
        elif self.robot_type == 'jaco' and not self.new:
            target_pos = np.array([-0.5, 0, 0.8]) + self.np_random.uniform(-0.05, 0.05, size=3)
            target_orient = p.getQuaternionFromEuler(np.array([0, np.pi/2.0, 0]), physicsClientId=self.id)
            _, self.target_robot_joint_positions = self.util.ik_random_restarts(self.robot, 8, target_pos, target_orient, self.world_creation, self.robot_left_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=[0, 1, 2, 3, 4, 5, 6], max_iterations=1000, max_ik_random_restarts=40, random_restart_threshold=0.03, step_sim=True)
            self.world_creation.set_gripper_open_position(self.robot, position=1, left=True, set_instantly=True)
            self.tool = self.world_creation.init_tool(self.robot, mesh_scale=[0.001]*3, pos_offset=[0, 0, 0.02], orient_offset=p.getQuaternionFromEuler([0, -np.pi/2.0, 0], physicsClientId=self.id), maximal=False)

        self.generate_target()
        p.setPhysicsEngineParameter(numSubSteps=0, numSolverIterations=50, physicsClientId=self.id)
        p.setGravity(0, 0, 0, physicsClientId=self.id)
        p.setGravity(0, 0, -1, body=self.human, physicsClientId=self.id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        if self.replay:
            p.restoreState(fileName=os.path.join(self.replay_dir, 'frame_0.bullet'))

        obs = self._get_obs([0], [0, 0])
        if self.vr and self.participant >= 0:
            p.saveBullet(self.save_filename % 0)
            with open(os.path.join(self.directory, 'setup.pkl'), 'wb') as f:
                pickle.dump([self.robot_type, self.gender, self.hipbone_to_mouth_height], f)
        return obs

    def generate_target(self):
        # Randomly select either upper arm or forearm for the target limb to scratch
        if self.gender == 'male':
            self.limb, length, radius = [[9, 0.279, 0.043], [11, 0.257, 0.033]][self.np_random.randint(2)]
        else:
            self.limb, length, radius = [[9, 0.264, 0.0355], [11, 0.234, 0.027]][self.np_random.randint(2)]
        self.target_on_arm = self.util.point_on_capsule(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -length]), radius=radius, theta_range=(0, np.pi*2))
        arm_pos, arm_orient = p.getLinkState(self.human, self.limb, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        target_pos, target_orient = p.multiplyTransforms(arm_pos, arm_orient, self.target_on_arm, [0, 0, 0, 1], physicsClientId=self.id)
        sphere_collision = -1
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 1, 1], physicsClientId=self.id)
        self.target = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=target_pos, useMaximalCoordinates=False, physicsClientId=self.id)
        self.update_targets()

    def update_targets(self):
        arm_pos, arm_orient = p.getLinkState(self.human, self.limb, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        target_pos, target_orient = p.multiplyTransforms(arm_pos, arm_orient, self.target_on_arm, [0, 0, 0, 1], physicsClientId=self.id)
        self.target_pos = np.array(target_pos)
        p.resetBasePositionAndOrientation(self.target, self.target_pos, [0, 0, 0, 1], physicsClientId=self.id)