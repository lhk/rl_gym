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
import os

class Brain():
    def __init__(self, NUM_ACTIONS, LEARNING_RATE, RHO, EPSILON):

        self.num_actions = NUM_ACTIONS

        model = self.__create_model()
        target_model = self.__create_model()

        model.compile(RMSprop(LEARNING_RATE, rho=RHO, epsilon=EPSILON), loss=huber_loss)

    def __create_model(self):
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