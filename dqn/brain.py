import numpy as np

np.random.seed(0)

import tensorflow as tf
import keras
from keras.layers import Conv2D, Flatten, Input, Multiply, Lambda, Subtract, Add
from keras.models import Model
from keras.optimizers import RMSprop
import keras.backend as K

import dqn.params as params
from dqn.memory import Memory
import os
import shutil


class Brain:
    def __init__(self, memory: Memory, loss="mse"):

        self.memory = memory
        # use this to influence the tensorflow behaviour
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = params.TF_ALLOW_GROWTH
        config.log_device_placement = params.TF_LOG_DEVICE_PLACEMENT

        sess = tf.Session(config=config)
        K.set_session(sess)

        # dqn works on two models
        self.model = self.create_model()
        self.target_model = self.create_model()

        # only one of them needs to be compiled for training
        self.model.compile(RMSprop(params.LEARNING_RATE, rho=params.RHO, epsilon=params.EPSILON), loss=loss)

        self.target_updates = 0

        # cleaning a directory for checkpoints
        if os.path.exists(os.getcwd() + "/checkpoints/"):
            shutil.rmtree(os.getcwd() + "/checkpoints/")
        os.mkdir(os.getcwd() + "/checkpoints/")

        self.model.load_weights(os.getcwd() + "/checkpoints_bkup2/dqn_model370.hd5")
        self.target_model.load_weights(os.getcwd() + "/checkpoints_bkup2/dqn_model370.hd5")

    def create_model(self):
        assert False, "use one of the subclasses instead"

    def predict_q(self, state):

        # keras only works if there is a batch dimension
        if state.shape == params.INPUT_SHAPE:
            state = state.reshape((-1, *params.INPUT_SHAPE))
        return self.model.predict([state, np.ones((state.shape[0], params.NUM_ACTIONS))])

    def predict_q_target(self, state):

        # keras only works if there is a batch dimension
        if state.shape == params.INPUT_SHAPE:
            state = state.reshape((-1, *params.INPUT_SHAPE))
        return self.target_model.predict([state, np.ones((state.shape[0], params.NUM_ACTIONS))])

    def get_targets(self, to_states, rewards, done):
        next_q_target = self.predict_q_target(to_states)
        next_q = self.predict_q(to_states)
        chosen_q = next_q_target[np.arange(next_q.shape[0]), next_q.argmax(axis=-1)]

        # this is the value that should be predicted by the network
        q_targets = rewards + params.GAMMA * chosen_q * (1 - done)

        return q_targets

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

        self.target_updates += 1

        # save the target network every N steps
        if self.target_updates % params.SAVE_NETWORK_FREQ == 0:
            self.target_model.save("checkpoints/dqn_model{}.hd5".format(self.target_updates + 120))

    def train_once(self):

        # sample batch with priority as weight, train on it
        training_indices = self.memory.sample_indices()
        batch = self.memory[training_indices]

        from_states, to_states, actions, q_targets, terminals = batch

        assert from_states.shape[0] == params.BATCH_SIZE, "batchsize must be as defined in dqn.params.BATCH_SIZE"
        assert from_states.dtype == np.uint8, "we work on uint8. are you mixing different types of preprocessing ?"

        # create a one-hot mask for the actions
        action_mask = np.zeros((actions.shape[0], params.NUM_ACTIONS))
        action_mask[np.arange(actions.shape[0]), actions] = 1

        q_targets = q_targets.reshape((-1, 1)) * action_mask

        self.model.train_on_batch([from_states, action_mask], q_targets)

        if (self.memory.priority_based_sampling):
            q_predicted = self.predict_q(from_states)
            q_predicted *= action_mask

            errors = np.abs(q_targets - q_predicted)
            errors = errors.sum(axis=-1)
            priorities = np.power(errors + params.ERROR_BIAS, params.ERROR_POW)

            self.memory.update_priority(training_indices, priorities)


class DQN_Brain(Brain):
    def __init__(self, memory: Memory, loss="mse"):
        Brain.__init__(self, memory, loss)

    def create_model(self):
        input_layer = Input(params.INPUT_SHAPE)

        rescaled = Lambda(lambda x: x / 255.)(input_layer)
        conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(rescaled)
        conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
        conv = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv)

        conv_flattened = Flatten()(conv)

        hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
        output_layer = keras.layers.Dense(params.NUM_ACTIONS)(hidden)

        mask_layer = Input((params.NUM_ACTIONS,))

        output_masked = Multiply()([output_layer, mask_layer])
        return Model(inputs=(input_layer, mask_layer), outputs=output_masked)


class Dueling_Brain(Brain):
    def __init__(self, memory: Memory, loss="mse"):
        Brain.__init__(self, memory, loss)

    def create_model(self):
        input_layer = Input(params.INPUT_SHAPE)

        rescaled = Lambda(lambda x: x / 255.)(input_layer)
        conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(rescaled)
        conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
        conv = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv)

        conv_flattened = Flatten()(conv)

        hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
        advantage = keras.layers.Dense(params.NUM_ACTIONS)(hidden)
        advantage_mean = keras.layers.Lambda(lambda x: K.mean(x, axis=-1))(advantage)

        hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
        value = keras.layers.Dense(params.NUM_ACTIONS)(hidden)

        advantage_white = Subtract()([advantage, advantage_mean])
        q_values = Add()([advantage_white, value])

        mask_layer = Input((params.NUM_ACTIONS,))

        q_values_masked = Multiply()([q_values, mask_layer])
        return Model(inputs=(input_layer, mask_layer), outputs=q_values_masked)
