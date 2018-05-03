# coding: utf-8

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

import random

from visualization_helpers import *

import keras
import keras.backend as K
from keras.layers import Conv2D, Dense, Activation, Flatten, Input, Multiply
from keras.optimizers import Adam
from keras.models import Model

from keras.regularizers import l2
from keras.layers import TimeDistributed, BatchNormalization, MaxPool2D

import gym

env = gym.make('Breakout-v4')
env.reset()

# a network to predict q values for every action
num_actions = env.action_space.n
batch_size = 20
input_shape = (105, 80, 4)

def preprocess(frame):
    downsampled = frame[::2, ::2]
    grayscale = downsampled.mean(axis=2)/255
    return grayscale


def create_model():
    input_layer = Input(input_shape)

    conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(input_layer)
    conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
    conv_flattened = Flatten()(conv)
    hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
    output_layer = keras.layers.Dense(num_actions)(hidden)

    mask_layer = Input((num_actions,))

    output_masked = Multiply()([output_layer, mask_layer])
    return Model(inputs=(input_layer, mask_layer), outputs=output_masked)


q_approximator = create_model()
q_approximator_fixed = create_model()

q_approximator.compile(Adam(1e-3), loss="mse")

# a queue for past observations
replay_memory = []

from tqdm import tqdm

res_values = []
state = env.reset()

# parameters
gamma = 0.98  # for discounting future rewards
eps = 0.1  # for eps-greedy policy


retrain = False


def get_starting_state():
    state = np.zeros(input_shape, dtype=np.float32)
    frame = env.reset()
    state[:, :, 0] = preprocess(frame)

    for i in range(1, 4):
        action = env.action_space.sample()
        observation = env.step(action)
        state[:, :, i] = preprocess(observation[0])

    return state


state = get_starting_state()

#import time
#debug = True


if retrain:
    for i in tqdm(range(10000)):

        # interact with the environment
        # take random or best action
        action = None
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            q_values = q_approximator_fixed.predict([state.reshape(1, *input_shape), np.ones((1, num_actions))])
            action = q_values.argmax()

        # record environments reaction for the chosen action
        observation, reward, done, _ = env.step(action)

        new_frame = preprocess(observation)

        new_state = np.empty_like(state)
        new_state[:, :, :-1] = state[:, :, 1:]
        new_state[:, :, -1] = new_frame

        # done means the environment had to restart, this is bad
        if done:
            reward = - 100


        # this is given in the paper, they use only the sign
        reward = np.sign(reward)

        replay_memory.append((state, action, reward, new_state))

        if len(replay_memory) > 100000:
            replay_memory.pop(0)

        if not done:
            state = new_state
        else:
            state = get_starting_state()

        # train the q function approximator
        if len(replay_memory) > batch_size:
            batch = random.sample(replay_memory, batch_size)

            current_states = [replay[0] for replay in batch]
            current_states = np.array(current_states)

            # the target is
            # r + gamma * max Q(s_next)
            #
            # we need to get the predictions for the next state
            next_states = [replay[3] for replay in batch]
            next_states = np.array(next_states)

            q_predictions = q_approximator_fixed.predict([next_states, np.ones((batch_size, num_actions))])
            q_max = q_predictions.max(axis=1, keepdims=True)

            rewards = [replay[2] for replay in batch]
            rewards = np.array(rewards)
            rewards = rewards.reshape((batch_size, 1))

            targets = rewards + gamma * q_max

            actions = [replay[1] for replay in batch]
            actions = np.array(actions)
            mask = np.zeros((batch_size, num_actions))
            mask[np.arange(batch_size), actions] = 1

            targets = targets * mask

            res = q_approximator.train_on_batch([current_states, mask], targets)
            res_values.append(res)

        if i%500 == 0:
            q_approximator_fixed.set_weights(q_approximator.get_weights())

    q_approximator.save_weights("q_approx.hdf5")
else:
    q_approximator.load_weights("q_approx.hdf5")

import matplotlib.pyplot as plt

plt.plot(res_values)
plt.show()

env.reset()

env.render()

state = get_starting_state()
done = False
while True:
    env.render()
    q_values = q_approximator.predict([state.reshape((1,*input_shape)), np.ones((1, num_actions))])
    action = q_values.argmax()

    observation, reward, done, _ = env.step(action)
    new_frame = preprocess(observation)

    new_state = np.empty_like(state)
    new_state[:, :, :-1] = state[:, :, 1:]
    new_state[:, :, -1] = new_frame

    state = new_state

    if done:
        state = get_starting_state()
        print("resetting")

