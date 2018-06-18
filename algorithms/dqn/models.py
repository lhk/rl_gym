import numpy as np

np.random.seed(0)

import tensorflow as tf
import keras
from keras.layers import Conv2D, Flatten, Input, Multiply, Lambda, Subtract, Add
from keras.models import Model
from keras.optimizers import RMSprop
import keras.backend as K

import algorithms.dqn.params as params
from algorithms.dqn.memory import Memory
import os
import shutil

class DQN_Model():
    def __init__(self):
        input_layer = Input(params.INPUT_SHAPE)

        rescaled = Lambda(lambda x: x / 255.)(input_layer)
        conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(rescaled)
        conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
        # conv = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv)

        conv_flattened = Flatten()(conv)

        hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
        q_values = keras.layers.Dense(params.NUM_ACTIONS)(hidden)

        mask_layer = Input((params.NUM_ACTIONS,))

        q_values_masked = Multiply()([q_values, mask_layer])
        self.model = Model(inputs=(input_layer, mask_layer), outputs=q_values_masked)

        self.loss_regularization = sum(self.model.losses)
        self.trainable_weights = self.model.trainable_weights
        self.q_values_masked = q_values_masked