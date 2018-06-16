import time

import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras.regularizers import l2
import ppo.params as params
from ppo.memory import Memory


class ConvLSTMModel():
    def __init__(self):
        # build a model to predict action probabilities and values
        input_state = Input(shape=(*params.INPUT_SHAPE,))

        rescaled = Lambda(lambda x: x / 255.)(input_state)
        conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu', kernel_regularizer=l2(params.L2_REG_CONV))(
            rescaled)
        conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', kernel_regularizer=l2(params.L2_REG_CONV))(conv)

        conv_flattened = Flatten()(conv)
        dense = Dense(256, activation="relu", kernel_regularizer=l2(params.L2_REG_FULLY))(conv_flattened)

        # shape = [batch_size, time_steps, input_dim]
        dense = Reshape((1, 256))(dense)

        # apply an rnn
        # expose the state of the cell, so that we can recreate the setup
        # of the cell during training
        gru_cell = GRU(params.RNN_SIZE, return_state=True, kernel_regularizer=l2(params.L2_REG_FULLY))
        input_memory = Input(shape=(params.RNN_SIZE,))
        gru_tensor, output_memory = gru_cell(dense, initial_state=input_memory)

        pred_policy = Dense(params.NUM_ACTIONS, activation='softmax', kernel_regularizer=l2(params.L2_REG_FULLY))(
            gru_tensor)
        pred_value = Dense(1, activation='linear', kernel_regularizer=l2(params.L2_REG_FULLY))(gru_tensor)

        model = Model(inputs=[input_state, input_memory], outputs=[pred_policy, pred_value, output_memory])

        # the model is not compiled with any loss function
        # but the regularizers are still exposed as losses
        loss_regularization = sum(model.losses)

        self.model = model
        self.input_state = input_state
        self.input_memory = input_memory
        self.trainable_weights = model.trainable_weights
        self.pred_policy = pred_policy
        self.pred_value = pred_value
        self.loss_regularization = loss_regularization
