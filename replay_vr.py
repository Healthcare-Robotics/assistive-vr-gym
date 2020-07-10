import gym, assistive_gym, argparse

parser = argparse.ArgumentParser(description='VR replay')
parser.add_argument('--env', default='FeedingVRJaco-v0', help='env', required=True)
parser.add_argument('--replay-dir', default='', help='Replay directory', required=True)
args = parser.parse_args()

env = gym.make(args.env)
env.replay_setup(args.replay_dir)

env.render()
observation = env.reset()

while True:
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample())