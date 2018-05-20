# coding: utf-8

import random

import gym
import keras
import keras.backend as K
import numpy as np

np.random.seed(0)

import skimage.color
import skimage.transform
import tensorflow as tf
from drawnow import drawnow, figure
from keras.layers import Conv2D, Flatten, Input, Multiply
from keras.models import Model
from keras.optimizers import RMSprop
from pylab import subplot, plot, title

# check wether tensorflow really runs on gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.log_device_placement=True

sess = tf.Session(config=config)
K.set_session(sess)

from util.loss_functions import huber_loss

# matplotlib.use('Qt5Agg')
env = gym.make('Breakout-v4')
env.seed(0)

# a network to predict q values for every action
num_actions = env.action_space.n
frame_size = (84, 84)
input_shape = (*frame_size, 4)

random.seed(0)

import lycon

def preprocess_frame(frame):
    downsampled = lycon.resize(frame, width=frame_size[0], height=frame_size[1],
                               interpolation=lycon.Interpolation.NEAREST)
    grayscale = downsampled.mean(axis=-1).astype(np.uint8)
    return grayscale


def preprocess_state(state):
    return state / 255.


def create_model():
    input_layer = Input(input_shape)

    conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(input_layer)
    conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
    #conv = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv)

    conv_flattened = Flatten()(conv)

    hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
    output_layer = keras.layers.Dense(num_actions)(hidden)

    mask_layer = Input((num_actions,))

    output_masked = Multiply()([output_layer, mask_layer])
    return Model(inputs=(input_layer, mask_layer), outputs=output_masked)


# parameters, taken from the paper

batch_size = 32

learning_rate = 0.00025
rho = 0.95
epsilon = 0.01

train_skips = 2
network_updates = 0
target_network_update_freq = 1e4

noop_max = 30
noop_counter = 0

replay_memory_size = int(3e5)
replay_start_size = int(5e4)

total_interactions = int(3e6)

initial_exploration = 1.0
final_exploration = 0.1
final_exploration_frame = int(total_interactions//2)

repeat_action = 1

# multiplying exploration by this factor brings it down to final_exploration
# after final_exploration_frame frames
exploration_step = (initial_exploration - final_exploration) / final_exploration_frame
exploration = initial_exploration

gamma = 0.99

retrain = True

q_approximator = create_model()
q_approximator_fixed = create_model()

q_approximator.compile(RMSprop(learning_rate, rho=rho, epsilon=epsilon), loss=huber_loss)

# a queue for past observations
from collections import deque

replay_memory = deque(maxlen=replay_memory_size)

from tqdm import tqdm


def interact(state, action):
    # record environments reaction for the chosen action
    observation, reward, done, _ = env.step(action)

    new_frame = preprocess_frame(observation)

    new_state = np.empty_like(state)
    new_state[:, :, :-1] = state[:, :, 1:]
    new_state[:, :, -1] = new_frame

    return new_state, reward, done


def interact_multiple(state, action, times):
    total_reward = 0

    for i in range(times):
        state, reward, done = interact(state, action)
        total_reward += reward

        if (done):
            break

    return state, total_reward, done


def get_starting_state():
    state = np.zeros(input_shape, dtype=np.uint8)
    frame = env.reset()
    state[:, :, -1] = preprocess_frame(frame)

    action = 0

    # we repeat the action 4 times, since our initial state needs 4 stacked frames
    times = 4
    state, _, _ = interact_multiple(state, action, times)

    return state


state = get_starting_state()

plot_skips = 10

total_rewards = []
total_durations = []

total_reward = 0
total_duration = 0

highest_q_values = []
highest_q_value = -np.inf
#figure()


def draw_fig():
    subplot(2, 1, 1)
    title("rewards")
    plot(total_rewards[-50::2])

    subplot(2, 1, 2)
    title("durations")
    plot(total_durations[-50::2])


#drawnow(draw_fig)

if retrain:

    np.random.seed(0)
    env.seed(0)
    state = get_starting_state()

    for interaction in tqdm(range(total_interactions), smoothing=0.95):

        # interact with the environment
        # take random or best action
        action = None
        if random.random() < exploration:
            action = env.action_space.sample()
        else:
            q_values = q_approximator_fixed.predict([preprocess_state(state.reshape(1, *input_shape)),
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
        else :
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

            #if len(total_durations) % plot_skips == 0:
            #    drawnow(draw_fig)

        # first fill the replay queue, then start training
        if interaction < replay_start_size:
            continue

        # don't train the network at every step
        if interaction % train_skips != 0:
            continue

        # train the q function approximator
        training_indices = np.random.choice(len(replay_memory), batch_size, replace=False)
        #batch = replay_memory[training_indices]
        batch = [replay_memory[idx] for idx in training_indices]

        current_states = [replay[0] for replay in batch]
        current_states = np.array(current_states)

        # the target is
        # r + gamma * max Q(s_next)
        #
        # we need to get the predictions for the next state
        next_states = [replay[3] for replay in batch]
        next_states = np.array(next_states)

        q_predictions = q_approximator_fixed.predict(
            [preprocess_state(next_states), np.ones((batch_size, num_actions))])
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

        network_updates += 1
        res = q_approximator.train_on_batch([preprocess_state(current_states), mask], targets)

        if network_updates % target_network_update_freq == 0:
            q_approximator_fixed.set_weights(q_approximator.get_weights())
            network_updates = 0

    q_approximator.save_weights("q_approx.hdf5")
else:
    q_approximator.load_weights("q_approx.hdf5")

env.reset()

state = get_starting_state()

episodes = 0
max_episodes = 50
total_reward = 0
while True:
    env.render()
    q_values = q_approximator.predict([preprocess_state(state.reshape((1, *input_shape))), np.ones((1, num_actions))])
    action = q_values.argmax()
    # action = env.action_space.sample()
    # 0 is noop action,
    # we allow only a limited amount of noop actions
    if action !=1 :
       noop_counter += 1

       if noop_counter > noop_max:
           action=1
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
