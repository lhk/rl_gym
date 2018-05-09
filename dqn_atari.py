# coding: utf-8

import random
random.seed(0)

import gym
import keras
import keras.backend as K
import lycon
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Flatten, Input, Multiply, Lambda
from keras.models import Model
from keras.optimizers import RMSprop
from pylab import subplot, plot, title


# a queue for past observations
from collections import deque

# force tensorflow to run on cpu
# do this if you want to evaluate trained networks, without interrupting
# an ongoing training process
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

# use this to influence the tensorflow behaviour
config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.log_device_placement=True

sess = tf.Session(config=config)
K.set_session(sess)

from loss_functions import huber_loss

# only a subset of matplotlib backends supports forwarding over X11
# this Qt5Agg is compatible with remote debugging
# you can ignore the setting
import matplotlib
matplotlib.use('Qt5Agg')

# this is all that's needed to set up the openai gym
env = gym.make('Breakout-v4')
env.reset()

# parameters for the training setup
# the parameters exposed here are taken from the deepmind paper
# but their values are changed
# do not assume that this is an optimal setup

# parameters for the structure of the neural network
num_actions = env.action_space.n
frame_size = (84, 84)
input_shape = (*frame_size, 4)
batch_size = 32

# parameters for the reinforcement process
gamma = 0.99 # discount factor for future updates

# parameters for the optimizer
learning_rate = 0.00025
rho = 0.95
epsilon = 0.01

# parameters for the training
total_interactions = int(3e6) # after this many interactions, the training stops
train_skips = 2    # interact with the environment X times, update the network once

target_network_update_freq = 1e4 # update the target network every X training steps

# parameters for interacting with the environment
initial_exploration = 1.0  # initial chance of sampling a random action
final_exploration = 0.1  # final chance
final_exploration_frame = int(total_interactions // 2) # frame at which final value is reached
repeat_action_max = 30 # maximum number of repeated actions before sampling random action

# parameters for the memory
replay_memory_size = int(3e5)
replay_start_size = int(5e4)

# variables for the network
exploration = initial_exploration
exploration_step = (initial_exploration - final_exploration) / final_exploration_frame

network_updates_counter = 0
last_action = None
repeat_action_counter = 0

replay_memory = deque(maxlen=replay_memory_size)



retrain = True



def preprocess_frame(frame):
    downsampled = lycon.resize(frame, width=frame_size[0], height=frame_size[1],
                               interpolation=lycon.Interpolation.NEAREST)
    grayscale = downsampled.mean(axis=-1).astype(np.uint8)
    return grayscale


def create_model():
    input_layer = Input(input_shape)

    rescaled = Lambda(lambda x: x / 255.)(input_layer)
    conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(rescaled)
    conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
    # conv = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv)

    conv_flattened = Flatten()(conv)

    hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
    output_layer = keras.layers.Dense(num_actions)(hidden)

    mask_layer = Input((num_actions,))

    output_masked = Multiply()([output_layer, mask_layer])
    return Model(inputs=(input_layer, mask_layer), outputs=output_masked)


q_approximator = create_model()
q_approximator_fixed = create_model()

q_approximator.compile(RMSprop(learning_rate, rho=rho, epsilon=epsilon), loss=huber_loss)



from tqdm import tqdm


def interact(state, action):
    # record environments reaction for the chosen action
    observation, reward, done, _ = env.step(action)

    new_frame = preprocess_frame(observation)

    new_state = np.empty_like(state)
    new_state[:, :, :-1] = state[:, :, 1:]
    new_state[:, :, -1] = new_frame

    return new_state, reward, done


def get_starting_state():
    state = np.zeros(input_shape, dtype=np.uint8)
    frame = env.reset()
    state[:, :, -1] = preprocess_frame(frame)

    action = 0

    # we repeat the action 4 times, since our initial state needs 4 stacked frames
    times = 4
    for i in range(times):
        state, _, _ = interact(state, action)

    return state


state = get_starting_state()

plot_skips = 10

total_rewards = []
total_durations = []

total_reward = 0
total_duration = 0

highest_q_values = []
highest_q_value = -np.inf


# figure()


def draw_fig():
    subplot(2, 1, 1)
    title("rewards")
    plot(total_rewards[-50::2])

    subplot(2, 1, 2)
    title("durations")
    plot(total_durations[-50::2])


# drawnow(draw_fig)

if retrain:

    np.random.seed(0)
    state = get_starting_state()

    for interaction in tqdm(range(total_interactions), smoothing=1):

        # interact with the environment
        # take random or best action
        action = None
        if random.random() < exploration:
            action = env.action_space.sample()
        else:
            q_values = q_approximator_fixed.predict([state.reshape(1, *input_shape),
                                                     np.ones((1, num_actions))])
            action = q_values.argmax()
            if q_values.max() > highest_q_value:
                highest_q_value = q_values.max()

        # anneal the epsilon
        if exploration > final_exploration:
            exploration -= exploration_step

        # 0 is noop action,
        # we allow only a limited amount of noop actions
        # TODO: this is hardcoding for the breakout setup, remove action=1, sample at random
        if action == 0:
            noop_counter += 1

            if noop_counter > noop_max:
                action = 1
                noop_counter = 0

        # record environments reaction for the chosen action
        new_state, reward, done = interact_multiple(state, action, repeat_action)

        # done means the environment had to restart, this is bad
        # please note: the restart reward is chosen as -1
        # the rewards are clipped to [-1, 1] according to the paper
        # if that would not be done, we would have to scale this reward
        # to align to the other replays given in the game
        if done:
            reward = - 1

        # this is given in the paper, they use only the sign
        reward = np.sign(reward)

        if len(replay_memory) == replay_memory_size:
            replay_memory.pop()

        replay_memory.append((state, action, reward, new_state, done))

        total_reward += reward
        total_duration += 1

        if not done:
            state = new_state
        else:
            state = get_starting_state()

            print("an episode has finished")
            print("total reward: ", total_reward)
            print("total steps: ", total_duration)
            print("highest q-value: ", highest_q_value)
            total_rewards.append(total_reward)
            total_durations.append(total_duration)
            highest_q_values.append(highest_q_value)
            total_reward = 0
            total_duration = 0
            highest_q_value = -np.inf

            # if len(total_durations) % plot_skips == 0:
            #    drawnow(draw_fig)

        # first fill the replay queue, then start training
        if interaction < replay_start_size:
            continue

        # don't train the network at every step
        if interaction % train_skips != 0:
            continue

        # train the q function approximator
        batch = random.sample(replay_memory, batch_size)

        current_states = [replay[0] for replay in batch]
        current_states = np.array(current_states)

        # the target is
        # r + gamma * max Q(s_next)
        #
        # we need to get the predictions for the next state
        next_states = [replay[3] for replay in batch]
        next_states = np.array(next_states)

        q_predictions = q_approximator_fixed.predict(
            [next_states, np.ones((batch_size, num_actions))])
        q_max = q_predictions.max(axis=1, keepdims=True)

        rewards = [replay[2] for replay in batch]
        rewards = np.array(rewards)
        rewards = rewards.reshape((batch_size, 1))

        dones = [replay[4] for replay in batch]
        dones = np.array(dones, dtype=np.bool)
        dones = dones.reshape((batch_size, 1))

        # the value is immediate reward and discounted expected future reward
        # by definition, in a terminal state, the future reward is 0
        immediate_rewards = rewards
        future_rewards = gamma * q_max * (1 - dones)

        targets = immediate_rewards + future_rewards

        actions = [replay[1] for replay in batch]
        actions = np.array(actions)
        mask = np.zeros((batch_size, num_actions))
        mask[np.arange(batch_size), actions] = 1

        targets = targets * mask

        network_updates_counter += 1
        res = q_approximator.train_on_batch([current_states, mask], targets)

        if network_updates_counter % target_network_update_freq == 0:
            q_approximator_fixed.set_weights(q_approximator.get_weights())
            network_updates_counter = 0

    q_approximator.save_weights("q_approx_new.hdf5")
else:
    q_approximator.load_weights("q_approx_new.hdf5")

env.reset()

state = get_starting_state()

episodes = 0
max_episodes = 50
total_reward = 0
while True:
    env.render()
    q_values = q_approximator.predict([state.reshape((1, *input_shape)), np.ones((1, num_actions))])
    action = q_values.argmax()
    # action = env.action_space.sample()
    # 0 is noop action,
    # we allow only a limited amount of noop actions
    if action == 0:
        noop_counter += 1

        if noop_counter > noop_max:
            while action == 0:
                action = env.action_space.sample()
            noop_counter = 0

    state, reward, done = interact_multiple(state, action, repeat_action)
    total_reward += reward

    if done:
        state = get_starting_state()
        print("resetting", episodes)
        episodes += 1
        if episodes > max_episodes:
            break

print(total_reward / episodes)
