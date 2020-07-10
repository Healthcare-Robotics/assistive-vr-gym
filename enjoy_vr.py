import argparse
import os, sys, select, argparse
import pybullet as p
import time
from gym.utils import seeding
import copy
import numpy as np
import torch

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--participant', default=0, help='Participant ID', type=int)
parser.add_argument('--gender', default='male', help='male / female')
parser.add_argument('--task', default=0, help='Pick one task to resume', type=int)
parser.add_argument('--trial', default=0, help='Pick one trial to resume', type=int)
parser.add_argument('--add-timestep', action='store_true', default=False,
                    help='add timestep to observations')
parser.add_argument('--non-det', action='store_true', default=False,
                    help='whether to use a non-deterministic policy')
args = parser.parse_args()
args.det = not args.non_det

task_names = ['Feeding', 'Drinking', 'ScratchItch', 'BedBathing']
trial_names = [('Jaco', '%s%sNew-v0.pt'), ('Jaco', '%s%s-v0.pt'), ('PR2', '%s%sNew-v0.pt'), ('PR2', '%s%s-v0.pt')]
np.random.seed(args.participant)
participant_code = int(np.random.randint(1000, 10000))
np_random, _ = seeding.np_random(participant_code)

while True:
    np_random.shuffle(task_names)
    if task_names[0] != "BedBathing":
        break

sequence = []

new_trials = [('Jaco', '%s%sNew-v0.pt'), ('PR2', '%s%sNew-v0.pt')]
for task in task_names:
    np_random.shuffle(trial_names)
    pract_trial = new_trials[np_random.choice(2)]
    new_trial_names = copy.deepcopy(trial_names)
    new_trial_names.insert(0, pract_trial)
    sequence.append((task, new_trial_names))

obs_robot_len = 25
action_human_len = 6
close_function = None
# Run once and record the height manually.
hipbone_to_mouth_height = 0.54

print("next trial: ", sequence[args.task][0], sequence[args.task][1][args.trial])
for i, (task, trials) in enumerate(sequence):
    if i < args.task:   
        pass
    else:
        for j, (robot, policy_filename) in enumerate(trials):
            if i == args.task and j < args.trial:
                continue

            def setup(env):
                global obs_robot_len, close_function, action_human_len, hipbone_to_mouth_height
                env.setup(args.gender, args.participant, 'Human' if 'Human' in policy_filename else 'NewStatic' if 'New' in policy_filename else 'Static', hipbone_to_mouth_height)
                obs_robot_len = env.obs_robot_len
                close_function = env.close
                action_human_len = env.action_human_len

            if 'Human' in policy_filename:
                env = make_vec_envs('%sVR%sHuman-v0' % (task, robot), np_random.randint(1000, 10000), 1, None, None, args.add_timestep, device='cpu', allow_early_resets=False, setup_function=setup)
            elif 'New' in policy_filename:
                env = make_vec_envs('%sVR%sNew-v0' % (task, robot), np_random.randint(1000, 10000), 1, None, None, args.add_timestep, device='cpu', allow_early_resets=False, setup_function=setup)
            else:
                env = make_vec_envs('%sVR%s-v0' % (task, robot), np_random.randint(1000, 10000), 1, None, None, args.add_timestep, device='cpu', allow_early_resets=False, setup_function=setup)

            render_func = get_render_func(env)
            if 'Human' in policy_filename:
                actor_critic, _, ob_rms = torch.load(os.path.join('trained_models', 'ppo', policy_filename % (task, robot)))
            else:    
                actor_critic, ob_rms = torch.load(os.path.join('trained_models', 'ppo', policy_filename % (task, robot)))

            vec_norm = get_vec_normalize(env)
            if vec_norm is not None:
                vec_norm.eval()
                vec_norm.ob_rms = ob_rms

            recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
            masks = torch.zeros(1, 1)

            if render_func is not None:
                render_func('human')

            obs = env.reset()
            obs = obs[:, :obs_robot_len]

            # Allow the person to freely move around while the robot is static
            while True:
                keys = p.getKeyboardEvents()
                if p.B3G_RETURN in keys and keys[p.B3G_RETURN] & p.KEY_IS_DOWN:
                    break
                # If enter is pressed, then end is set to True and we exit free movement
                env.step(torch.zeros(1, 2))

            env.step(torch.zeros(1, 3))
            print(task, robot, policy_filename)
            t = time.time()
            for _ in range(200):
                with torch.no_grad():
                    value, action, _, recurrent_hidden_states = actor_critic.act(
                        obs, recurrent_hidden_states, masks, deterministic=args.det)

                # Obser reward and next obs
                if 'Human' in policy_filename:
                    action = torch.cat((action, torch.zeros(1, action_human_len)), dim=-1)
                obs, reward, done, info = env.step(action)
                
                obs = obs[:, :obs_robot_len]
                masks.fill_(0.0 if done else 1.0)

                if render_func is not None:
                    render_func('human')

            print('-'*15, 'Simulation ended', '-'*15)

            if j == 4:
                if i == 3:
                    pass
                else:
                    print("next trial: ", sequence[i+1][0], sequence[i+1][1][0])
            else:
                print("next trial: ", task, trials[j+1])
            # Allow the person to freely move around while the robot is static
            while True:
                keys = p.getKeyboardEvents()
                if p.B3G_RETURN in keys and keys[p.B3G_RETURN] & p.KEY_IS_DOWN:
                    break
                # If enter is pressed, then end is set to True and we exit free movement
                env.step(torch.zeros(1, 2))

            close_function()
# Final reset to ensure the last data file is saved
# This is critical, otherwise the last trial won't be saved.
obs = env.reset()