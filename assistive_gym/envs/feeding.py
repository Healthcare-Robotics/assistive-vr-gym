import os, sys, pickle, time, glob
from gym import spaces
import numpy as np
import pybullet as p
from datetime import datetime

from .env import AssistiveEnv

class FeedingEnv(AssistiveEnv):
    def __init__(self, robot_type='pr2', human_control=False, vr=False, new=False):
        self.participant = -1
        self.gender = 'male'
        self.hipbone_to_mouth_height = 0.6
        self.policy_name = ''
        self.replay = False
        self.replay_dir = None
        self.human_gains, self.waist_gains, self.human_forces, self.waist_forces = 0.05, 0.05, 1.0, 4.0
        super(FeedingEnv, self).__init__(robot_type=robot_type, task='feeding', human_control=human_control, vr=vr, new=new, frame_skip=5, time_step=0.02, action_robot_len=7, action_human_len=(4 if human_control else 0), obs_robot_len=25, obs_human_len=(23 if human_control else 0))

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
                if len(action) == 3:
                    self.update_objects(fix=False)
                else:
                    self.free_move(robot_arm='right', gains=self.config('robot_gains'), forces=self.config('robot_forces'))
                obs = self._get_obs([0], [0, 0])
                return obs, 0, False, dict()
            self.take_step(action, robot_arm='right', gains=self.config('robot_gains'), forces=self.config('robot_forces'), human_gains=0.005)
            self.action_list.append(action)
            if self.vr and self.participant >= 0:
                if self.iteration == 200:
                    # End of simulation, save action_list
                    with open(os.path.join(self.directory, 'actions.pkl'), 'wb') as f:
                        pickle.dump(self.action_list, f)

        robot_force_on_human, spoon_force_on_human = self.get_total_force()
        total_force_on_human = robot_force_on_human + spoon_force_on_human
        reward_food, food_mouth_velocities, food_hit_human_reward = self.get_food_rewards()
        end_effector_velocity = np.linalg.norm(p.getBaseVelocity(self.spoon, physicsClientId=self.id)[0])
        obs = self._get_obs([spoon_force_on_human], [robot_force_on_human, spoon_force_on_human])

        # Get human preferences
        preferences_score = self.human_preferences(end_effector_velocity=end_effector_velocity, total_force_on_human=robot_force_on_human, tool_force_at_target=spoon_force_on_human, food_hit_human_reward=food_hit_human_reward, food_mouth_velocities=food_mouth_velocities)

        spoon_pos, spoon_orient = p.getBasePositionAndOrientation(self.spoon, physicsClientId=self.id)
        spoon_pos = np.array(spoon_pos)

        reward_distance_mouth_target = -np.linalg.norm(self.target_pos - spoon_pos) # Penalize robot for distance between the spoon and human mouth.
        reward_action = -np.sum(np.square(action))  # Penalize actions

        reward = self.config('distance_weight')*reward_distance_mouth_target + self.config('action_weight')*reward_action + self.config('food_reward_weight')*reward_food + preferences_score

        if self.gui and reward_food != 0 and not self.vr:
            print('Task success:', self.task_success, 'Food reward:', reward_food)

        info = {'total_force_on_human': total_force_on_human, 'task_success': int(self.task_success >= self.total_food_count*self.config('task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = False

        if self.replay and self.iteration == 200:
            done = True
        return obs, reward, done, info

    def get_total_force(self):
        robot_force_on_human = 0
        spoon_force_on_human = 0
        for c in p.getContactPoints(bodyA=self.robot, bodyB=self.human, physicsClientId=self.id):
            robot_force_on_human += c[9]
        for c in p.getContactPoints(bodyA=self.spoon, bodyB=self.human, physicsClientId=self.id):
            spoon_force_on_human += c[9]
        return robot_force_on_human, spoon_force_on_human

    def get_food_rewards(self):
        # Check all food particles to see if they have left the spoon or entered the person's mouth
        # Give the robot a reward or penalty depending on food particle status
        food_reward = 0
        food_hit_human_reward = 0
        food_mouth_velocities = []
        foods_to_remove = []
        for f in self.foods:
            food_pos, food_orient = p.getBasePositionAndOrientation(f, physicsClientId=self.id)
            distance_to_mouth = np.linalg.norm(self.target_pos - food_pos)
            if distance_to_mouth < 0.02:
                # Delete particle and give robot a reward
                food_reward += 20
                self.task_success += 1
                food_velocity = np.linalg.norm(p.getBaseVelocity(f, physicsClientId=self.id)[0])
                food_mouth_velocities.append(food_velocity)
                foods_to_remove.append(f)
                p.resetBasePositionAndOrientation(f, self.np_random.uniform(1000, 2000, size=3), [0, 0, 0, 1], physicsClientId=self.id)
                continue
            elif food_pos[-1] < 0.5 or len(p.getContactPoints(bodyA=f, bodyB=self.table, physicsClientId=self.id)) > 0 or len(p.getContactPoints(bodyA=f, bodyB=self.bowl, physicsClientId=self.id)) > 0:
                # Delete particle and give robot a penalty for spilling food
                food_reward -= 5
                foods_to_remove.append(f)
                continue
            if len(p.getContactPoints(bodyA=f, bodyB=self.human, physicsClientId=self.id)) > 0 and f not in self.foods_hit_person:
                # Record that this food particle just hit the person, so that we can penalize the robot
                self.foods_hit_person.append(f)
                food_hit_human_reward -= 1
        self.foods = [f for f in self.foods if f not in foods_to_remove]
        return food_reward, food_mouth_velocities, food_hit_human_reward

    def _get_obs(self, forces, forces_human):
        torso_pos = np.array(p.getLinkState(self.robot, 15 if self.robot_type == 'pr2' else 0, computeForwardKinematics=True, physicsClientId=self.id)[0])
        spoon_pos, spoon_orient = p.getBasePositionAndOrientation(self.spoon, physicsClientId=self.id)
        robot_right_joint_states = p.getJointStates(self.robot, jointIndices=self.robot_right_arm_joint_indices, physicsClientId=self.id)
        robot_right_joint_positions = np.array([x[0] for x in robot_right_joint_states])
        robot_pos, robot_orient = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.id)
        if self.human_control:
            human_pos = np.array(p.getLinkState(self.human, 3, computeForwardKinematics=True, physicsClientId=self.id)[:2][0])
            human_joint_states = p.getJointStates(self.human, jointIndices=[24, 25, 26, 27], physicsClientId=self.id)
            human_joint_positions = np.array([x[0] for x in human_joint_states])

        head_pos, head_orient = p.getLinkState(self.human, 27, computeForwardKinematics=True, physicsClientId=self.id)[:2]

        robot_obs = np.concatenate([spoon_pos-torso_pos, spoon_orient, spoon_pos-self.target_pos, robot_right_joint_positions, head_pos-torso_pos, head_orient, forces]).ravel()
        if self.human_control:
            human_obs = np.concatenate([spoon_pos-human_pos, spoon_orient, spoon_pos-self.target_pos, human_joint_positions, head_pos-human_pos, head_orient, forces_human]).ravel()
        else:
            human_obs = []

        return np.concatenate([robot_obs, human_obs]).ravel()

    def reset(self):
        p.setRealTimeSimulation(enableRealTimeSimulation=0, physicsClientId=self.id)
        if self.vr and self.participant >= 0:
            self.directory = os.path.join(os.getcwd(), 'participant_%d' % self.participant, 'feeding_vr_data_' + self.robot_type + '_' + self.policy_name + ('_participant_%d' % self.participant) + datetime.now().strftime('_%Y-%m-%d_%H-%M-%S'))
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

        if self.vr or self.replay:
            if self.vr:
                seat_to_hmd_height = self.hipbone_to_mouth_height + (0.068 + 0.1335*self.config('radius_scale', 'human_female') if self.gender == 'male' else 0.058 + 0.127*self.config('radius_scale', 'human_female'))
                p.setOriginCameraPositionAndOrientation(deviceTypeFilter=p.VR_DEVICE_HMD, pos_offset=[0, -0.06, -(0.47+seat_to_hmd_height)], orn_offset=[0, 0, 0, 1])
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

        self.robot_lower_limits = self.robot_lower_limits[self.robot_right_arm_joint_indices]
        self.robot_upper_limits = self.robot_upper_limits[self.robot_right_arm_joint_indices]
        self.reset_robot_joints()

        # Place a bowl of food on a table
        self.table = p.loadURDF(os.path.join(self.world_creation.directory, 'table', 'table_tall.urdf'), basePosition=[0.35, -0.9, 0], baseOrientation=p.getQuaternionFromEuler([0, 0, 0], physicsClientId=self.id), physicsClientId=self.id)
        self.bowl_scale = 0.75
        bowl_pos = np.array([-0.15, -0.55, 0.75]) + np.array([self.np_random.uniform(-0.05, 0.05), self.np_random.uniform(-0.05, 0.05), 0])
        self.bowl = p.loadURDF(os.path.join(self.world_creation.directory, 'dinnerware', 'bowl.urdf'), basePosition=bowl_pos, baseOrientation=p.getQuaternionFromEuler([np.pi/2.0, 0, 0], physicsClientId=self.id), useMaximalCoordinates=False, physicsClientId=self.id)

        if self.robot_type == 'jaco':
            p.resetBasePositionAndOrientation(self.robot, [-0.35, -0.3, 0.36], [0.0, 0.0, -0.7071067811865475, 0.7071067811865476], physicsClientId=self.id)
            if self.new:
                target_pos = np.array(bowl_pos) + np.array([0, -0.1, 0.4]) + self.np_random.uniform(-0.05, 0.05, size=3)
                target_orient = p.getQuaternionFromEuler(np.array([np.pi/2.0, 0, np.pi/2.0]), physicsClientId=self.id)
                # , check_env_collisions=True!!!!!!!!!!!
                _, self.target_robot_joint_positions = self.util.ik_random_restarts(self.robot, 8, target_pos, target_orient, self.world_creation, self.robot_right_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=[0, 1, 2, 3, 4, 5, 6], max_iterations=1000, max_ik_random_restarts=40, random_restart_threshold=0.01, step_sim=True, check_env_collisions=True if self.vr or self.replay else False or self.replay)
                self.world_creation.set_gripper_open_position(self.robot, position=1.33, left=False, set_instantly=True)
                self.spoon = self.world_creation.init_tool(self.robot, mesh_scale=[0.08]*3, pos_offset=[0.1, -0.0225, 0.03], orient_offset=p.getQuaternionFromEuler([-0.1, -np.pi/2.0, 0], physicsClientId=self.id), left=False, maximal=False)

        if self.vr or self.replay:
            self.human_controllable_joint_indices = [0, 1, 2, 25, 26, 27] + list(range(7, 14)) + list(range(17, 24))
            joints_positions = [(10, np.deg2rad(-90)), (20, np.deg2rad(-90)), (28, np.deg2rad(-90)), (31, np.deg2rad(80)), (35, np.deg2rad(-90)), (38, np.deg2rad(80))]
            joints_positions += [(25, self.np_random.uniform(np.deg2rad(-30), np.deg2rad(30))), (26, self.np_random.uniform(np.deg2rad(-30), np.deg2rad(30))), (27, self.np_random.uniform(np.deg2rad(-30), np.deg2rad(30)))]
            self.world_creation.setup_human_joints(self.human, joints_positions, self.human_controllable_joint_indices, use_static_joints=True, human_reactive_force=None)
            p.resetBasePositionAndOrientation(self.human, [0, 0.03, 0.89-0.23725 if self.gender == 'male' else 0.86-0.225], [0, 0, 0, 1], physicsClientId=self.id)
            human_joint_states = p.getJointStates(self.human, jointIndices=[24, 25, 26, 27], physicsClientId=self.id)

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
            self.human_controllable_joint_indices = [24, 25, 26, 27]
            if self.new:
                p.resetBasePositionAndOrientation(self.human, [0, 0.03, 0.89-0.23725 if self.gender == 'male' else 0.86-0.225], [0, 0, 0, 1], physicsClientId=self.id)
                human_positioned = False
                right_arm_shoulder_links = [6, 9, 11, 13]
                left_arm_shoulder_links = [16, 19, 21, 23]
                r_arm_wheelchair_dists = []
                l_arm_wheelchair_dists = []
                while not human_positioned or (np.min(human_robot_dists + [1]) < 0.01):
                    human_positioned = True
                    joints_positions = [(10, np.deg2rad(-90)), (20, np.deg2rad(-90)), (28, np.deg2rad(-90)), (31, np.deg2rad(80)), (35, np.deg2rad(-90)), (38, np.deg2rad(80))]
                    
                    joints_positions += [(i, 0) for i in list(range(7, 14)) + list(range(17, 24))]
                    joints_positions += [(i, self.np_random.uniform(np.deg2rad(-30), np.deg2rad(30))) for i in [25, 26, 27]]
                    joints_positions += [(i, self.np_random.uniform(np.deg2rad(-10), np.deg2rad(10))) for i in [0, 1, 2]]

                    self.world_creation.setup_human_joints(self.human, joints_positions, [], use_static_joints=True, human_reactive_force=None)

                    if self.robot_type == 'jaco':
                        human_robot_dists = [c[8] for c in p.getClosestPoints(bodyA=self.human, bodyB=self.robot, distance=0.05, physicsClientId=self.id)]
                    else:
                        human_robot_dists = []
            else:
                joints_positions = [(10, np.deg2rad(-90)), (20, np.deg2rad(-90)), (28, np.deg2rad(-90)), (31, np.deg2rad(80)), (35, np.deg2rad(-90)), (38, np.deg2rad(80))]
                joints_positions += [(i, self.np_random.uniform(np.deg2rad(-30), np.deg2rad(30))) for i in [25, 26, 27]]
                self.world_creation.setup_human_joints(self.human, joints_positions, self.human_controllable_joint_indices if (self.human_control or self.world_creation.human_impairment == 'tremor') else [], use_static_joints=True, human_reactive_force=None)
                p.resetBasePositionAndOrientation(self.human, [0, 0.03, 0.89-0.23725 if self.gender == 'male' else 0.86-0.225], [0, 0, 0, 1], physicsClientId=self.id)
            human_joint_states = p.getJointStates(self.human, jointIndices=self.human_controllable_joint_indices, physicsClientId=self.id)
        
        self.target_human_joint_positions = np.array([x[0] for x in human_joint_states])
        self.human_lower_limits = self.human_lower_limits[self.human_controllable_joint_indices]
        self.human_upper_limits = self.human_upper_limits[self.human_controllable_joint_indices]

        # Set target on mouth
        self.mouth_pos = [0, -0.11, 0.03] if self.gender == 'male' else [0, -0.1, 0.03]
        head_pos, head_orient = p.getLinkState(self.human, 27, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        target_pos, target_orient = p.multiplyTransforms(head_pos, head_orient, self.mouth_pos, [0, 0, 0, 1], physicsClientId=self.id)
        self.target_pos = np.array(target_pos)
        sphere_collision = -1
        if self.vr:
            sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 0, 0], physicsClientId=self.id)
        else:
            sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 0, 1], physicsClientId=self.id)
        self.target = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=self.target_pos, useMaximalCoordinates=False, physicsClientId=self.id)

        p.resetDebugVisualizerCamera(cameraDistance=1.10, cameraYaw=40, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.75], physicsClientId=self.id)

        human_joint_indices = [24, 25, 26, 27]
        if self.robot_type == 'pr2':
            target_pos = np.array(bowl_pos) + np.array([0, -0.1, 0.4]) + self.np_random.uniform(-0.05, 0.05, size=3)
            target_orient = p.getQuaternionFromEuler([np.pi/2.0, 0, 0], physicsClientId=self.id)
            _, _, self.target_robot_joint_positions = self.position_robot_toc(self.robot, 54, [(target_pos, target_orient), (self.target_pos, None)], [(self.target_pos, target_orient)], self.robot_right_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=range(15, 15+7), pos_offset=np.array([0.1, 0.2, 0]), max_ik_iterations=200, step_sim=True, check_env_collisions=False, human_joint_indices=human_joint_indices, human_joint_positions=self.target_human_joint_positions)
            self.target_robot_joint_positions = self.target_robot_joint_positions[0]
            self.world_creation.set_gripper_open_position(self.robot, position=0.03, left=False, set_instantly=True)
            self.spoon = self.world_creation.init_tool(self.robot, mesh_scale=[0.08]*3, pos_offset=[0, -0.03, -0.11], orient_offset=p.getQuaternionFromEuler([-0.2, 0, 0], physicsClientId=self.id), left=False, maximal=False)
        elif self.robot_type == 'jaco' and not self.new:
            # CHECK ENV_COLLISION
            target_pos = np.array(bowl_pos) + np.array([0, -0.1, 0.4]) + self.np_random.uniform(-0.05, 0.05, size=3)
            target_orient = p.getQuaternionFromEuler(np.array([np.pi/2.0, 0, np.pi/2.0]), physicsClientId=self.id)
            _, self.target_robot_joint_positions = self.util.ik_random_restarts(self.robot, 8, target_pos, target_orient, self.world_creation, self.robot_right_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=[0, 1, 2, 3, 4, 5, 6], max_iterations=1000, max_ik_random_restarts=40, random_restart_threshold=0.01, step_sim=True, check_env_collisions=True if self.vr or self.replay else False)
            self.world_creation.set_gripper_open_position(self.robot, position=1.33, left=False, set_instantly=True)
            self.spoon = self.world_creation.init_tool(self.robot, mesh_scale=[0.08]*3, pos_offset=[0.1, -0.0225, 0.03], orient_offset=p.getQuaternionFromEuler([-0.1, -np.pi/2.0, 0], physicsClientId=self.id), left=False, maximal=False)

        p.resetBasePositionAndOrientation(self.bowl, bowl_pos, p.getQuaternionFromEuler([np.pi/2.0, 0, 0], physicsClientId=self.id), physicsClientId=self.id)

        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        p.setGravity(0, 0, 0, body=self.robot, physicsClientId=self.id)
        p.setGravity(0, 0, 0, body=self.human, physicsClientId=self.id)
        p.setGravity(0, 0, 0, body=self.spoon, physicsClientId=self.id)

        p.setPhysicsEngineParameter(numSubSteps=2, numSolverIterations=10, physicsClientId=self.id)

        # Generate food
        spoon_pos, spoon_orient = p.getBasePositionAndOrientation(self.spoon, physicsClientId=self.id)
        spoon_pos = np.array(spoon_pos)
        food_radius = 0.005
        food_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=food_radius, physicsClientId=self.id)
        food_visual = -1
        if self.vr:
            food_mass = 0.0
        else:
            food_mass = 0.001
        food_count = 2*2*2
        batch_positions = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    batch_positions.append(np.array([i*2*food_radius-0.005, j*2*food_radius, k*2*food_radius+0.02]) + spoon_pos)
        last_food_id = p.createMultiBody(baseMass=food_mass, baseCollisionShapeIndex=food_collision, baseVisualShapeIndex=food_visual, basePosition=[0, 0, 0], useMaximalCoordinates=False, batchPositions=batch_positions, physicsClientId=self.id)
        self.foods = list(range(last_food_id-food_count+1, last_food_id+1))
        self.foods_hit_person = []
        self.total_food_count = len(self.foods)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        if self.vr:
            self.update_objects()
        else:
            # Drop food in the spoon
            for _ in range(100):
                p.stepSimulation(physicsClientId=self.id)

        if self.replay:
            p.restoreState(fileName=os.path.join(self.replay_dir, 'frame_0.bullet'))

        obs = self._get_obs([0], [0, 0])
        if self.vr and self.participant >= 0:
            p.saveBullet(self.save_filename % 0)
            with open(os.path.join(self.directory, 'setup.pkl'), 'wb') as f:
                pickle.dump([self.robot_type, self.gender, self.hipbone_to_mouth_height], f)

        return obs

    def update_objects(self, fix=True):
        if fix:
            p.changeDynamics(self.spoon, -1, mass=0, physicsClientId=self.id)
        else:
            for f in self.foods:
                p.changeDynamics(f, -1, mass=0.001, physicsClientId=self.id)
            p.changeDynamics(self.spoon, -1, mass=1, physicsClientId=self.id)

            # Drop food in the spoon
            for _ in range(100):
                p.stepSimulation(physicsClientId=self.id)

    def update_targets(self):
        head_pos, head_orient = p.getLinkState(self.human, 27, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        target_pos, target_orient = p.multiplyTransforms(head_pos, head_orient, self.mouth_pos, [0, 0, 0, 1], physicsClientId=self.id)
        self.target_pos = np.array(target_pos)
        p.resetBasePositionAndOrientation(self.target, self.target_pos, [0, 0, 0, 1], physicsClientId=self.id)
