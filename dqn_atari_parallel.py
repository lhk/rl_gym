# coding: utf-8

import random

random.seed(0)
# only a subset of matplotlib backends supports forwarding over X11
# this Qt5Agg is compatible with remote debugging
# you can ignore the setting
import matplotlib

matplotlib.use('Qt5Agg')

import gym
import keras
import keras.backend as K
import lycon
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Flatten, Input, Multiply, Lambda
from keras.models import Model
from keras.optimizers import RMSprop

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
config.log_device_placement = True

sess = tf.Session(config=config)
K.set_session(sess)

from loss_functions import huber_loss

# parameters for the training setup
# the parameters exposed here are taken from the deepmind paper
# but their values are changed
# do not assume that this is an optimal setup

RETRAIN = True

# parameters for the structure of the neural network
NUM_ACTIONS = 4  # for breakout
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
TOTAL_INTERACTIONS = int(1e7)  # after this many interactions, the training stops
TRAIN_SKIPS = 4  # interact with the environment X times, update the network once

TARGET_NETWORK_UPDATE_FREQ = 1e4  # update the target network every X training steps
SAVE_NETWORK_FREQ = 5  # save every Xth version of the target network

# parameters for interacting with the environment
INITIAL_EXPLORATION = 1.0  # initial chance of sampling a random action
FINAL_EXPLORATION = 0.1  # final chance
FINAL_EXPLORATION_FRAME = int(1e7)  # frame at which final value is reached
EXPLORATION_STEP = (INITIAL_EXPLORATION - FINAL_EXPLORATION) / FINAL_EXPLORATION_FRAME

REPEAT_ACTION_MAX = 30  # maximum number of repeated actions before sampling random action

# parameters for the memory
REPLAY_MEMORY_SIZE = int(3.5e5)
REPLAY_START_SIZE = int(5e2)

# variables, these are not meant to be edited by the user
# they are used to keep track of various properties of the training setup
exploration = INITIAL_EXPLORATION  # chance of sampling a random action

network_updates_counter = 0  # number of times the network has been updated
target_network_updates_counter = 0  # number of times the target has been updated

replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)  # a buffer for past observations


# helper methods

def preprocess_frame(frame):
    downsampled = lycon.resize(frame, width=FRAME_SIZE[0], height=FRAME_SIZE[1],
                               interpolation=lycon.Interpolation.NEAREST)
    grayscale = downsampled.mean(axis=-1).astype(np.uint8)
    return grayscale


def interact(state, action, env):
    observation, reward, done, _ = env.step(action)

    new_frame = preprocess_frame(observation)

    new_state = np.empty_like(state)
    new_state[:, :, :-1] = state[:, :, 1:]
    new_state[:, :, -1] = new_frame

    return new_state, reward, done


def get_starting_state(env):
    state = np.zeros(INPUT_SHAPE, dtype=np.uint8)
    frame = env.reset()
    state[:, :, -1] = preprocess_frame(frame)

    action = 0

    # we repeat the action 4 times, since our initial state needs 4 stacked frames
    times = 4
    for i in range(times):
        state, _, _ = interact(state, action, env)

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


from multiprocessing import Value, Lock


def interaction_generator(q_approximator_fixed, replay_memory, exploration,
                          interaction_counter, interaction_lock):
    import keras.backend as K
    import tensorflow as tf

    # initialize state of generator
    env = gym.make('SpaceInvaders-v4')
    env.reset()

    state = get_starting_state(env)

    last_action = None  # action chosen at the last step
    repeat_action_counter = 0  # number of times this action has been repeated

    # use this to influence the tensorflow behaviour
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True

    sess = tf.Session(config=config, graph=graph)
    K.set_session(sess)

    # the generator will never stop interacting with the environment
    while True:

        # take random or best action
        if random.random() < exploration:
            action = env.action_space.sample()
        else:
            q_values = q_approximator_fixed.predict([state.reshape(1, *INPUT_SHAPE),
                                                     np.ones((1, NUM_ACTIONS))])
            action = q_values.argmax()
            if q_values.max() > highest_q_value:
                highest_q_value = q_values.max()

        # we only allow a limited amount of repeated actions
        if action == last_action:
            repeat_action_counter += 1

            if repeat_action_counter > REPEAT_ACTION_MAX:
                action = env.action_space.sample()
        else:
            last_action = action
            repeat_action_counter = 0

        new_state, reward, done = interact(state, action, env)

        # done means the environment had to restart, this is bad
        # please note: the restart reward is chosen as -1
        # the rewards are clipped to [-1, 1] according to the paper
        # if that would not be done, we would have to scale this reward
        # to align to the other replays given in the game
        if done:
            reward = - 1

        # this is given in the paper, they use only the sign
        reward = np.sign(reward)

        with interaction_lock:
            if len(replay_memory) == REPLAY_MEMORY_SIZE:
                replay_memory.pop()

            replay_memory.append((state, action, reward, new_state, done))

        if not done:
            state = new_state
        else:
            state = get_starting_state(env)

        # every generator shares this counter
        # only after we have filled the replay memory with enough new information
        # and only every Nth step, the network can be trained
        with interaction_lock:
            interaction_counter.value += 1

            if interaction_counter.value % 100 == 0:
                print(interaction_counter.value)

            if interaction_counter.value < REPLAY_START_SIZE:
                continue

            if interaction_counter.value % TRAIN_SKIPS != 0:
                continue

        batch = random.sample(replay_memory, BATCH_SIZE)

        current_states = [replay[0] for replay in batch]
        current_states = np.array(current_states)

        # the target is
        # r + gamma * max Q(s_next)
        #
        # we need to get the predictions for the next state
        next_states = [replay[3] for replay in batch]
        next_states = np.array(next_states)

        q_predictions = q_approximator_fixed.predict(
            [next_states, np.ones((BATCH_SIZE, NUM_ACTIONS))])
        q_max = q_predictions.max(axis=1, keepdims=True)

        rewards = [replay[2] for replay in batch]
        rewards = np.array(rewards)
        rewards = rewards.reshape((BATCH_SIZE, 1))

        dones = [replay[4] for replay in batch]
        dones = np.array(dones, dtype=np.bool)
        dones = dones.reshape((BATCH_SIZE, 1))

        # the value is immediate reward and discounted expected future reward
        # by definition, in a terminal state, the future reward is 0
        immediate_rewards = rewards
        future_rewards = GAMMA * q_max * (1 - dones)

        targets = immediate_rewards + future_rewards

        actions = [replay[1] for replay in batch]
        actions = np.array(actions)
        mask = np.zeros((BATCH_SIZE, NUM_ACTIONS))
        mask[np.arange(BATCH_SIZE), actions] = 1

        targets = targets * mask

        yield ([current_states, mask], targets)


q_approximator = create_model()
q_approximator_fixed = create_model()

# necessary for thread safe parallel prediction
q_approximator._make_predict_function()

# only this one will be trained
q_approximator.compile(RMSprop(LEARNING_RATE, rho=RHO, epsilon=EPSILON), loss=huber_loss)

graph = tf.get_default_graph()
# graph = K.get_session().graph

if RETRAIN:

    for i in range(100):
        interaction_counter = Value("i", 0)
        interaction_lock = Lock()

        q_approximator.fit_generator(interaction_generator(q_approximator_fixed,
                                                           replay_memory,
                                                           exploration,
                                                           interaction_counter,
                                                           interaction_lock),
                                     epochs=10, steps_per_epoch=BATCH_SIZE * 1000,
                                     use_multiprocessing=True,
                                     workers=1)

        # q_approximator_fixed.set_weights(q_approximator.get_weights())
