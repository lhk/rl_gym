# coding: utf-8

import random

import gym
import keras
import matplotlib
import numpy as np
from keras.layers import Conv2D, Flatten, Input, Multiply, Dense
from keras.models import Model
from keras.optimizers import RMSprop
from keras.regularizers import l2
from loss_functions import huber_loss

matplotlib.use('Qt5Agg')

env = gym.make('CartPole-v0')
env.reset()

# a network to predict q values for every action
num_actions = env.action_space.n
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

# parameters, taken from the paper

batch_size = 32
learning_rate = 0.00025
network_updates = 0
target_network_update_freq = 5e2

noop_max = 20
noop_counter = 0

replay_memory_size = int(1e6)
replay_start_size = int(1e5)

total_interactions = int(1e5)

initial_exploration = 1.0
final_exploration = 0.1
final_exploration_frame = total_interactions

# multiplying exploration by this factor brings it down to final_exploration
# after final_exploration_frame frames
exploration_factor = (final_exploration / initial_exploration) ** (1 / final_exploration_frame)
exploration = initial_exploration

gamma = 0.99

retrain = True

q_approximator = create_model()
q_approximator_fixed = create_model()

q_approximator.compile(RMSprop(learning_rate, rho=0.95, epsilon=0.01), loss=huber_loss)

# a queue for past observations
replay_memory = []

from tqdm import tqdm

res_values = []

state = env.reset()

if retrain:

    # sample random behaviour to fill the replay queue
    # please note: according to the paper, the annealing of epsilon seems to start here already
    # but that seems detrimental, we are not yet training the network
    for interaction in tqdm(range(replay_start_size), smoothing=0.9):
        action = env.action_space.sample()

        # record environments reaction for the chosen action
        new_state, reward, done, _ = env.step(action)

        # done means the environment had to restart, this is bad
        # please note: the restart reward is chosen as -1
        # the rewards are clipped to [-1, 1] according to the paper
        # if that would not be done, we would have to scale this reward
        # to align to the other replays given in the game
        if done:
            reward = - 1

        # this is given in the paper, they use only the sign
        reward = np.sign(reward)

        replay_memory.append((state, action, reward, new_state, done))

        if not done:
            state = new_state
        else:
            state = env.reset()

    # now train the network
    for interaction in tqdm(range(total_interactions), smoothing=0.9):

        # anneal an the epsilon
        exploration *= exploration_factor
        if exploration < final_exploration:
            exploration_factor = 1

        # interact with the environment
        # take random or best action
        action = None
        if random.random() < exploration:
            action = env.action_space.sample()
        else:
            q_values = q_approximator_fixed.predict([state.reshape(1, *input_shape), np.ones((1, num_actions))])
            action = q_values.argmax()

        # 0 is noop action,
        # we allow only a limited amount of noop actions
        if action == 0:
            noop_counter += 1

            if noop_counter > noop_max:
                while action == 0:
                    action = env.action_space.sample()

                noop_counter = 0

        # record environments reaction for the chosen action
        new_state, reward, done, _ = env.step(action)

        # done means the environment had to restart, this is bad
        # please note: the restart reward is chosen as -1
        # the rewards are clipped to [-1, 1] according to the paper
        # if that would not be done, we would have to scale this reward
        # to align to the other replays given in the game
        if done:
            reward = - 1

        # this is given in the paper, they use only the sign
        reward = np.sign(reward)

        replay_memory.append((state, action, reward, new_state, done))

        if len(replay_memory) > replay_memory_size:
            replay_memory.pop(np.random.randint(len(replay_memory)))

        if not done:
            state = new_state
        else:
            state = env.reset()

        # train the q function approximator
        batch = random.sample(replay_memory, batch_size)

        current_states = [replay[0] for replay in batch]
        current_states = np.array(current_states)
        current_states_float = current_states / 255.

        # the target is
        # r + gamma * max Q(s_next)
        #
        # we need to get the predictions for the next state
        next_states = [replay[3] for replay in batch]
        next_states = np.array(next_states)
        next_states_float = next_states / 255.

        q_predictions = q_approximator_fixed.predict([next_states_float, np.ones((batch_size, num_actions))])
        q_max = q_predictions.max(axis=1, keepdims=True)

        rewards = [replay[2] for replay in batch]
        rewards = np.array(rewards)
        rewards = rewards.reshape((batch_size, 1))

        dones = [replay[4] for replay in batch]
        dones = np.array(dones, dtype=np.bool)
        not_dones = np.logical_not(dones)
        not_dones = not_dones.reshape((batch_size, 1))

        # the value is immediate reward and discounted expected future reward
        # by definition, in a terminal state, the future reward is 0
        immediate_rewards = rewards
        future_rewards = gamma * q_max * (not_dones)

        targets = immediate_rewards + future_rewards

        actions = [replay[1] for replay in batch]
        actions = np.array(actions)
        mask = np.zeros((batch_size, num_actions))
        mask[np.arange(batch_size), actions] = 1

        targets = targets * mask

        network_updates += 1
        res = q_approximator.train_on_batch([current_states_float, mask], targets)
        res_values.append(res)

        if network_updates % target_network_update_freq == 0:
            q_approximator_fixed.set_weights(q_approximator.get_weights())
            network_updates = 0

    q_approximator.save_weights("q_approx.hdf5")
else:
    q_approximator.load_weights("q_approx.hdf5")

env.reset()

state = env.reset()
while True:
    env.render()
    q_values = q_approximator.predict([state.reshape((1, *input_shape)) / 255., np.ones((1, num_actions))])
    action = q_values.argmax()

    # 0 is noop action,
    # we allow only a limited amount of noop actions
    if action == 0:
        noop_counter += 1

        if noop_counter > noop_max:
            while action == 0:
                action = env.action_space.sample()

            noop_counter = 0

    new_state, reward, done, _ = env.step(action)

    state = new_state

    if done:
        state = env.reset()
        print("resetting")
