import numpy as np

np.random.seed(0)

import keras
from keras.layers import *
from keras.models import Model
from keras.regularizers import *
import lycon

import algorithms.dqn.params as params


class DQN_Model():
    OBSERVATION_SHAPE = (84, 84, params.FRAME_STACK)
    STATEFUL = False

    def __init__(self):
        input_observation = Input(self.OBSERVATION_SHAPE)

        rescaled = Lambda(lambda x: x / 255.)(input_observation)
        conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(rescaled)
        conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
        # conv = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv)

        conv_flattened = Flatten()(conv)

        hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
        q_values = keras.layers.Dense(params.NUM_ACTIONS)(hidden)

        mask_layer = Input((params.NUM_ACTIONS,))

        q_values_masked = Multiply()([q_values, mask_layer])
        self.model = Model(inputs=(input_observation, mask_layer), outputs=q_values_masked)

        self.input_observation = input_observation
        self.mask_layer = mask_layer

        self.loss_regularization = sum(self.model.losses)
        self.trainable_weights = self.model.trainable_weights
        self.q_values_masked = q_values_masked

    def preprocess(self, observation):
        downsampled = lycon.resize(observation, width=self.OBSERVATION_SHAPE[0], height=self.OBSERVATION_SHAPE[1],
                                   interpolation=lycon.Interpolation.NEAREST)
        grayscale = downsampled.mean(axis=-1).astype(np.uint8)
        return grayscale

    def predict(self, observation, state=None):
        if not state is None:
            raise AssertionError("this model is not stateful")

        # keras only works if there is a batch dimension
        if observation.shape == params.INPUT_SHAPE:
            observation = observation.reshape((-1, *params.INPUT_SHAPE))
        return self.model.predict([observation, np.ones((observation.shape[0], params.NUM_ACTIONS))]), None

    def get_initial_state(self):
        raise AssertionError("this model is not stateful")

    def create_feed_dict(self, observation, state, mask):
        if not state is None:
            raise AssertionError("this model is not stateful")
        return {self.input_observation: observation, self.mask_layer: mask}

class FullyConnectedModel():
    OBSERVATION_SHAPE = (7,)
    STATEFUL = False
    def __init__(self):
        # some parameters now belong to the model
        self.FC_SIZE = 64

        self.input_observation = Input(shape=(*self.OBSERVATION_SHAPE,))

        # predicting q values
        hidden = Dense(self.FC_SIZE, activation="tanh", kernel_regularizer=l2(params.L2_REG_FULLY))(
            self.input_observation)
        hidden = Dense(self.FC_SIZE, activation="tanh", kernel_regularizer=l2(params.L2_REG_FULLY))(hidden)
        q_values = Dense(params.NUM_ACTIONS, kernel_regularizer=l2(params.L2_REG_FULLY))(
            hidden)

        mask_layer = Input((params.NUM_ACTIONS,))

        q_values_masked = Multiply()([q_values, mask_layer])
        self.model = Model(inputs=(self.input_observation, mask_layer), outputs=q_values_masked)

        self.input_observation = self.input_observation
        self.mask_layer = mask_layer

        self.loss_regularization = sum(self.model.losses)
        self.trainable_weights = self.model.trainable_weights
        self.q_values_masked = q_values_masked

    def preprocess(self, observation):
        return observation

    def get_initial_state(self):
        raise AssertionError("this model is not stateful")

    def predict(self, observation, state=None):

        if not state is None:
            raise AssertionError("this model is not stateful")

        # keras always needs a batch dimension
        if observation.shape == self.OBSERVATION_SHAPE:
            observation = observation.reshape((-1, *self.OBSERVATION_SHAPE))

        return self.model.predict([observation, np.ones((observation.shape[0], params.NUM_ACTIONS))]), None

    def create_feed_dict(self, observation, state, mask):
        if not state is None:
            raise AssertionError("this model is not stateful")
        return {self.input_observation: observation, self.mask_layer : mask}
