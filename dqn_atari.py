# coding: utf-8

import random

random.seed(0)

import gym
import keras
import keras.backend as K
import lycon
import numpy as np

np.random.seed(0)

import tensorflow as tf
from keras.layers import Conv2D, Flatten, Input, Multiply, Lambda
from keras.models import Model
from keras.optimizers import RMSprop
from pylab import subplot, plot, title

from tqdm import tqdm

# directory management:
# delete all previous memory maps
# and create dirs for checkpoints (if not present)
import os
import shutil
if os.path.exists(os.getcwd()+"/memory_maps/"):
    shutil.rmtree(os.getcwd()+"/memory_maps/")
os.mkdir(os.getcwd()+"/memory_maps/")

if not os.path.exists(os.getcwd()+"/checkpoints/"):
    os.mkdir(os.getcwd()+"/checkpoints/")

# a queue for past observations
from collections import deque

# force tensorflow to run on cpu
# do this if you want to evaluate trained networks, without interrupting
# an ongoing training process
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

# use this to influence the tensorflow behaviour
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
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

RETRAIN = True

# parameters for the structure of the neural network
NUM_ACTIONS = env.action_space.n
FRAME_SIZE = (84, 84)
INPUT_SHAPE = (*FRAME_SIZE, 4)
BATCH_SIZE = 32

# parameters for the reinforcement process
GAMMA = 0.99  # discount factor for future updates

# parameters for the optimizer
LEARNING_RATE = 0.00025
RHO = 0.95
EPSILON = 0.01

# parameters for the training
TOTAL_INTERACTIONS = int(3e6)  # after this many interactions, the training stops
TRAIN_SKIPS = 2  # interact with the environment X times, update the network once

TARGET_NETWORK_UPDATE_FREQ = 1e4  # update the target network every X training steps
SAVE_NETWORK_FREQ = 5  # save every Xth version of the target network

# parameters for interacting with the environment
INITIAL_EXPLORATION = 1.0  # initial chance of sampling a random action
FINAL_EXPLORATION = 0.1  # final chance
FINAL_EXPLORATION_FRAME = int(TOTAL_INTERACTIONS//2)  # frame at which final value is reached
EXPLORATION_STEP = (INITIAL_EXPLORATION - FINAL_EXPLORATION) / FINAL_EXPLORATION_FRAME

REPEAT_ACTION_MAX = 30  # maximum number of repeated actions before sampling random action

# parameters for the memory
REPLAY_MEMORY_SIZE = int(3e5)
REPLAY_START_SIZE = int(1e3)

# variables, these are not meant to be edited by the user
# they are used to keep track of various properties of the training setup
exploration = INITIAL_EXPLORATION  # chance of sampling a random action

number_recorded_replays = 0
replay_index = 0 # index in the replay memory arrays

network_updates_counter = 0  # number of times the network has been updated
target_network_updates_counter = 0  # number of times the target has been updated
last_action = None  # action chosen at the last step
repeat_action_counter = 0  # number of times this action has been repeated


# replay memory as numpy arrays
# this makes it possible to store the states on disk as memory mapped arrays
from tempfile import mkstemp

#from_state_memory = np.memmap(mkstemp(dir="memory_maps")[0], dtype=np.uint8, mode="w+", shape=(REPLAY_MEMORY_SIZE, *INPUT_SHAPE))
#to_state_memory = np.memmap(mkstemp(dir="memory_maps")[0], dtype=np.uint8, mode="w+", shape=(REPLAY_MEMORY_SIZE, *INPUT_SHAPE))

from_state_memory = np.empty(shape=(REPLAY_MEMORY_SIZE, *INPUT_SHAPE), dtype=np.uint8)
to_state_memory = np.empty(shape=(REPLAY_MEMORY_SIZE, *INPUT_SHAPE), dtype=np.uint8)

# these other parts of the memory consume only very little memory and can be kept in ram
action_memory = np.empty(shape=(REPLAY_MEMORY_SIZE), dtype=np.uint8)
reward_memory = np.empty(shape=(REPLAY_MEMORY_SIZE, 1), dtype=np.int16)
terminal_memory = np.empty(shape=(REPLAY_MEMORY_SIZE, 1), dtype=np.bool)
#action_memory = np.memmap(mkstemp(dir="memory_maps")[0], dtype=np.uint8, mode="w+", shape=(REPLAY_MEMORY_SIZE, 1))
#reward_memory = np.memmap(mkstemp(dir="memory_maps")[0], dtype=np.float32, mode="w+", shape=(REPLAY_MEMORY_SIZE, 1))
#terminal_memory = np.memmap(mkstemp(dir="memory_maps")[0], dtype=np.bool, mode="w+", shape=(REPLAY_MEMORY_SIZE, 1))


# helper methods

def preprocess_frame(frame):
    downsampled = lycon.resize(frame, width=FRAME_SIZE[0], height=FRAME_SIZE[1],
                               interpolation=lycon.Interpolation.NEAREST)
    grayscale = downsampled.mean(axis=-1).astype(np.uint8)
    return grayscale


def interact(state, action):
    observation, reward, done, _ = env.step(action)

    new_frame = preprocess_frame(observation)

    new_state = np.empty_like(state)
    new_state[:, :, :-1] = state[:, :, 1:]
    new_state[:, :, -1] = new_frame

    return new_state, reward, done


def get_starting_state():
    state = np.zeros(INPUT_SHAPE, dtype=np.uint8)
    frame = env.reset()
    state[:, :, -1] = preprocess_frame(frame)

    action = 0

    # we repeat the action 4 times, since our initial state needs 4 stacked frames
    times = 4
    for i in range(times):
        state, _, _ = interact(state, action)

    return state


def create_model():
    input_layer = Input(INPUT_SHAPE)

    rescaled = Lambda(lambda x: x / 255.)(input_layer)
    conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(rescaled)
    conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
    # conv = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv)

    conv_flattened = Flatten()(conv)

    hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
    output_layer = keras.layers.Dense(NUM_ACTIONS)(hidden)

    mask_layer = Input((NUM_ACTIONS,))

    output_masked = Multiply()([output_layer, mask_layer])
    return Model(inputs=(input_layer, mask_layer), outputs=output_masked)


q_approximator = create_model()
q_approximator_fixed = create_model()

q_approximator.compile(RMSprop(LEARNING_RATE, rho=RHO, epsilon=EPSILON), loss=huber_loss)

state = get_starting_state()

total_rewards = []
total_durations = []

total_reward = 0
total_duration = 0

highest_q_values = []
highest_q_value = -np.inf


def draw_fig():
    subplot(2, 1, 1)
    title("rewards")
    plot(total_rewards[-50::2])

    subplot(2, 1, 2)
    title("durations")
    plot(total_durations[-50::2])


from drawnow import drawnow, figure

PLOT_SKIPS = 10
figure()
drawnow(draw_fig)

if RETRAIN:

    np.random.seed(0)
    env.seed(0)

    state = get_starting_state()

    for interaction in tqdm(range(TOTAL_INTERACTIONS), smoothing=1):

        # take random or best action
        action = None
        if random.random() < exploration:
            action = env.action_space.sample()
        else:
            q_values = q_approximator_fixed.predict([state.reshape(1, *INPUT_SHAPE),
                                                     np.ones((1, NUM_ACTIONS))])
            action = q_values.argmax()
            if q_values.max() > highest_q_value:
                highest_q_value = q_values.max()

        # anneal the epsilon
        if exploration > FINAL_EXPLORATION:
            exploration -= EXPLORATION_STEP

        # we only allow a limited amount of repeated actions
        if action == last_action:
            repeat_action_counter += 1

            if repeat_action_counter > REPEAT_ACTION_MAX:
                action = env.action_space.sample()
        else:
            last_action = action
            repeat_action_counter = 0

        new_state, reward, done = interact(state, action)

        # done means the environment had to restart, this is bad
        # please note: the restart reward is chosen as -1
        # the rewards are clipped to [-1, 1] according to the paper
        # if that would not be done, we would have to scale this reward
        # to align to the other replays given in the game
        if done:
            reward = - 1

        # this is given in the paper, they use only the sign
        #reward = np.sign(reward)

        from_state_memory[replay_index] = state
        to_state_memory[replay_index] = new_state
        action_memory[replay_index] = action
        reward_memory[replay_index] = reward
        terminal_memory[replay_index] = done

        replay_index += 1
        replay_index %= REPLAY_MEMORY_SIZE

        number_recorded_replays += 1

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
        if interaction < REPLAY_START_SIZE:
            continue

        # don't train the network at every step
        if interaction % TRAIN_SKIPS != 0:
            continue


        # train the q function approximator
        training_indices = np.random.choice(min(number_recorded_replays, REPLAY_MEMORY_SIZE),
                                            size=BATCH_SIZE, replace=False)

        current_states = from_state_memory[training_indices]
        next_states = to_state_memory[training_indices]
        rewards = reward_memory[training_indices]
        actions = action_memory[training_indices]
        terminal = terminal_memory[training_indices]

        q_predictions = q_approximator_fixed.predict(
            [next_states, np.ones((BATCH_SIZE, NUM_ACTIONS))])
        q_max = q_predictions.max(axis=1, keepdims=True)

        # the value is immediate reward and discounted expected future reward
        # by definition, in a terminal state, the future reward is 0
        immediate_rewards = rewards
        future_rewards = GAMMA * q_max * (1 - terminal)

        targets = immediate_rewards + future_rewards

        mask = np.zeros((BATCH_SIZE, NUM_ACTIONS))
        mask[np.arange(BATCH_SIZE), actions] = 1

        targets = targets * mask

        network_updates_counter += 1
        res = q_approximator.train_on_batch([current_states, mask], targets)

        if network_updates_counter % TARGET_NETWORK_UPDATE_FREQ == 0:
            q_approximator_fixed.set_weights(q_approximator.get_weights())
            network_updates_counter = 0

            target_network_updates_counter += 1
            if target_network_updates_counter % SAVE_NETWORK_FREQ == 0:
                q_approximator.save_weights("checkpoints/Breakout" + str(target_network_updates_counter) + ".hdf5")
else:
    q_approximator.load_weights("checkpoints/Breakout140.hdf5")

env.reset()

state = get_starting_state()

episodes = 0
max_episodes = 50
total_reward = 0
while True:
    env.render()
    q_values = q_approximator.predict([state.reshape((1, *INPUT_SHAPE)), np.ones((1, NUM_ACTIONS))])
    action = q_values.argmax()

    # we allow only a limited amount of repeated actions
    if action == last_action:
        repeat_action_counter += 1

        if repeat_action_counter > REPEAT_ACTION_MAX:
            action = env.action_space.sample()
    else:
        last_action = action
        repeat_action_counter = 0

    state, reward, done = interact(state, action)
    total_reward += reward

    if done:
        state = get_starting_state()
        print("resetting", episodes)
        episodes += 1
        if episodes > max_episodes:
            break

print(total_reward / episodes)
env.close()
