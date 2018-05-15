import numpy as np
np.random.seed(0)

import tensorflow as tf
import keras
from keras.layers import Conv2D, Flatten, Input, Multiply, Lambda
from keras.models import Model
from keras.optimizers import RMSprop
import keras.backend as K

import dqn.params as params

class Brain():
    def __init__(self, loss="mse"):

        self.model = self.__create_model()
        self.target_model = self.__create_model()

        self.model.compile(RMSprop(params.LEARNING_RATE, rho=params.RHO, epsilon=params.EPSILON), loss=loss)

    def __create_model(self):
        input_layer = Input(params.INPUT_SHAPE)

        rescaled = Lambda(lambda x: x / 255.)(input_layer)
        conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(rescaled)
        conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
        # conv = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv)

        conv_flattened = Flatten()(conv)

        hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
        output_layer = keras.layers.Dense(params.NUM_ACTIONS)(hidden)

        mask_layer = Input((params.NUM_ACTIONS,))

        output_masked = Multiply()([output_layer, mask_layer])
        return Model(inputs=(input_layer, mask_layer), outputs=output_masked)

    def predict_q(self, state):

        # keras only works if there is a batch dimension
        if state.shape == params.INPUT_SHAPE:
            state = state.reshape((-1, *params.INPUT_SHAPE))
        return self.model.predict(state)

    def predict_q_target(self, state):

        # keras only works if there is a batch dimension
        if state.shape == params.INPUT_SHAPE:
            state = state.reshape((-1, *params.INPUT_SHAPE))
        return self.model.predict(state)