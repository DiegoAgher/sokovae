import gym
import gym_sokoban
import time
import cv2
from PIL import Image
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Run environment with random selected actions.')
parser.add_argument('--rounds', '-r', metavar='rounds', type=int,
                    help='number of rounds to play (default: 1)', default=1)
parser.add_argument('--steps', '-s', metavar='steps', type=int,
                    help='maximum number of steps to be played each round (default: 300)', default=300)
parser.add_argument('--env', '-e', metavar='env',
                    help='Environment to load (default: Sokoban-v0)', default='Sokoban-v0')
parser.add_argument('--save', action='store_true',
                    help='Save images of single steps')
parser.add_argument('--gifs', action='store_true',
                    help='Generate Gif files from images')
parser.add_argument('--render_mode', '-m', metavar='render_mode',
                    help='Render Mode (default: human)', default='human')

args = parser.parse_args()
env_name = args.env
n_rounds = args.rounds
n_steps = args.steps
save_images = args.save or args.gifs
generate_gifs = args.gifs
render_mode = args.render_mode
observation_mode = 'tiny_rgb_array' if 'tiny' in render_mode else 'rgb_array'
scale_image = 16
resize_ = 40

# Creating target directory if images are to be stored
if save_images and not os.path.exists('images'):
    try:
        os.makedirs('images')
    except OSError:
        print('Error: Creating images target directory. ')

ts = time.time()
env = gym.make(env_name)
ACTION_LOOKUP = env.unwrapped.get_action_lookup()
print("Created environment: {}".format(env_name))


def print_available_actions():
    """
    Prints all available actions nicely formatted..
    :return:
    """
    available_actions_list = []
    for i in range(len(ACTION_LOOKUP)):
        available_actions_list.append(
            'Key: {} - Action: {}'.format(i, ACTION_LOOKUP[i])
        )
    display_actions = '\n'.join(available_actions_list)
    print()
    print('Action out of Range!')
    print('Available Actions:\n{}'.format(display_actions))
    print()

states = []
scaled = []
actions = []
rewards = []
next_states = []
next_states_resized= []
dones = []

for i_episode in range(n_rounds):
    print('Starting new game!')
    observation = env.reset()

    curr_resized = cv2.resize(observation, dsize=(resize_, resize_))
    for t in range(n_steps):
        env.render(render_mode, scale=scale_image)

        action = input('Select action: ')
        try:
            action = int(action)

            if not action in range(len(ACTION_LOOKUP)):
                raise ValueError

        except ValueError:
            print_available_actions()
            continue
 
        scaled.append(curr_resized)
        actions.append(action)

        observation, reward, done, info = env.step(action, observation_mode=observation_mode)
        rewards.append(reward)
        next_state_scaled = cv2.resize(observation, dsize=(resize_,resize_))
        curr_resized = next_state_scaled
        next_states.append(observation)
        next_states_resized.append(next_state_scaled)
        dones.append(done)
        print(ACTION_LOOKUP[action], reward, done, info)
        print(len(observation), len(observation[0]), len(observation[0][0]))
        if save_images:
            img = Image.fromarray(np.array(env.render(render_mode, scale=scale_image)), 'RGB')
            img.save(os.path.join('images', 'observation_{}_{}.png'.format(i_episode, t)))

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            env.render(render_mode, scale=scale_image)
            break

    with open('states_gameplay_{}.npy'.format(i_episode), 'wb') as f:
        np.save(f, states)

    with open('states_scaled_gameplay_{}.npy'.format(i_episode), 'wb') as f:
        np.save(f, scaled)


    with open('next_states_scaled_gameplay_{}.npy'.format(i_episode), 'wb') as f:
        np.save(f, next_states_resized)

    with open('actions_gameplay_{}.npy'.format(i_episode), 'wb') as f:
        np.save(f, actions)

    with open('rewards_gameplay_{}.npy'.format(i_episode), 'wb') as f:
        np.save(f, rewards)

    with open('next_states_gameplay_{}.npy'.format(i_episode), 'wb') as f:
        np.save(f, next_states)

    with open('done_gameplay_{}.npy'.format(i_episode), 'wb') as f:
        np.save(f, dones)

    if False:
        print('')
        import imageio

        with imageio.get_writer(os.path.join('images', 'round_{}.gif'.format(i_episode)), mode='I', fps=1) as writer:

                for t in range(n_steps):
                    try:

                        filename = os.path.join('images', 'observation_{}_{}.png'.format(i_episode, t))
                        image = imageio.imread(filename)
                        writer.append_data(image)

                    except:
                        pass

env.close()

