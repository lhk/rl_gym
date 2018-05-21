# coding: utf-8

import random

import gym
import numpy as np
from keras.layers import Dense, Input, Multiply
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

env = gym.make('CartPole-v0')
env.reset()

# a network to predict q values for every action
num_actions = env.action_space.n
batch_size = 12
input_shape = (4,)


def create_model():
    input_layer = Input(input_shape)
    dense1 = Dense(20,
                   activation="tanh",
                   kernel_regularizer=l2(0.01),
                   bias_regularizer=l2(0.01))(input_layer)
    dense2 = Dense(20,
                   activation="tanh",
                   kernel_regularizer=l2(0.01),
                   bias_regularizer=l2(0.01))(dense1)
    output_layer = Dense(num_actions)(dense2)

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
gamma = 0.99  # for discounting future rewards
eps = 0.1  # for eps-greedy policy

retrain = False

if retrain:
    for i in tqdm(range(20000)):

        # interact with the environment
        # take random or best action
        action = None
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            q_values = q_approximator_fixed.predict([state.reshape((1, 4)), np.ones((1, 2))])
            action = q_values.argmax()

        # record environments reaction for the chosen action
        new_state, reward, done, _ = env.step(action)

        # done means the environment had to restart, this is bad
        if done:
            reward = - 1

        replay_memory.append((state, action, reward, new_state))

        if len(replay_memory) > 10000:
            replay_memory.pop(0)

        if not done:
            state = new_state
        else:
            state = env.reset()

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

        if i % 300 == 0:
            q_approximator_fixed.set_weights(q_approximator.get_weights())

    q_approximator.save_weights("q_approx.hdf5")
else:
    q_approximator.load_weights("q_approx.hdf5")

import matplotlib.pyplot as plt

plt.plot(res_values)
plt.show()

env.reset()

env.render()

state = env.reset()
done = False
while True:
    env.render()
    q_values = q_approximator.predict([state.reshape((1, 4)), np.ones((1, 2))])
    action = q_values.argmax()
    state, _, done, _ = env.step(action)

    if done:
        state = env.reset()
        print("resetting")
