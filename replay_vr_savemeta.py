import os, glob, pickle, gym, assistive_gym, argparse
import numpy as np

# python3 replay_vr_savemeta.py --replay-dir ..\..\participant_data

parser = argparse.ArgumentParser(description='VR replay')
parser.add_argument('--replay-dir', default='', help='Replay directory', required=True)
args = parser.parse_args()

env_names = []
observations_vr = []
# positions_vr = []
rewards_vr = []
actions_vr = []
forces_vr = []
task_success_vr = []
for directory in glob.glob(os.path.join(args.replay_dir, 'participant_*', '*')):
    # if 'drinking' not in directory:# or 'pr2' not in directory:
    #     continue
    env_name = '%s%s-v0' % ('ScratchItch' if 'scratch_itch' in directory else 'Feeding' if 'feeding' in directory else 'Drinking' if 'drinking' in directory else 'BedBathing' if 'bed_bathing' in directory else 'skip', 'Jaco' if 'jaco' in directory else 'PR2')
    if 'skip' in env_name:
        continue
    print(env_name)
    print(directory)
    env = gym.make(env_name)
    env.replay_setup(directory)

    # env.render()
    observation = env.reset()

    observations = []
    # positions = []
    rewards = []
    forces = []
    task_success = 0.0
    done = False
    while not done:
        # env.render()
        observation, reward, done, info = env.step(env.action_space.sample())
        observations.append(observation)
        # positions_vr.append(observation[-10:-1])
        # TODO: THIS IS WRONG FOR NON ITCH SCRATCHING ENVS!
        # positions.append(observation[-10:-1])
        rewards.append(reward)
        forces.append(info['total_force_on_human'])
        task_success = info['task_success']
        # print(reward, info)

    env_names.append(directory)
    observations_vr.append(observations)
    rewards_vr.append(rewards)
    actions_vr.append(env.action_list)
    forces_vr.append(forces)
    task_success_vr.append(task_success)
    print(np.sum(rewards), np.mean(forces), task_success)
    env.close()

with open('observations_vr.pkl', 'wb') as f:
    pickle.dump([env_names, observations_vr, rewards_vr, actions_vr, forces_vr, task_success_vr], f, pickle.HIGHEST_PROTOCOL)

