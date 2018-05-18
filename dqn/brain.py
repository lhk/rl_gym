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

        # use this to influence the tensorflow behaviour
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = params.TF_ALLOW_GROWTH
        config.log_device_placement= params.TF_LOG_DEVICE_PLACEMENT

        sess = tf.Session(config=config)
        K.set_session(sess)

        # set up two models
        self.model = self.__create_model()
        self.target_model = self.__create_model()

        # only one of them needs to be compiled for training
        self.model.compile(RMSprop(params.LEARNING_RATE, rho=params.RHO, epsilon=params.EPSILON), loss=loss)

        self.train_steps = 0

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
        return self.model.predict([state, np.ones((state.shape[0], params.NUM_ACTIONS))])

    def predict_q_target(self, state):

        # keras only works if there is a batch dimension
        if state.shape == params.INPUT_SHAPE:
            state = state.reshape((-1, *params.INPUT_SHAPE))
        return self.target_model.predict([state, np.ones((state.shape[0], params.NUM_ACTIONS))])

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def train_on_batch(self, batch):
        from_states, to_states, actions, rewards, terminals = batch

        assert from_states.shape[0] == params.BATCH_SIZE, "batchsize must be as defined in dqn.params.BATCH_SIZE"
        assert from_states.dtype == np.uint8, "we work on uint8. are you mixing different types of preprocessing ?"

        # the target is
        # r + gamma * max Q(s_next)
        #
        # we need to get the predictions for the next state
        next_q_target = self.predict_q_target(to_states)
        next_q = self.predict_q(to_states)
        #q_max = next_q_target[np.arange(next_q.shape[0]), next_q.argmax(axis=1)]
        #q_max = q_max.reshape((-1, 1))
        q_max = next_q_target.max(axis=1, keepdims=True)

        immediate_rewards = rewards
        future_rewards = params.GAMMA * q_max * (1 - terminals)
        targets = immediate_rewards + future_rewards

        # create a one-hot mask for the actions
        action_mask = np.zeros((params.BATCH_SIZE, params.NUM_ACTIONS))
        action_mask[np.arange(params.BATCH_SIZE), actions] = 1

        targets = targets * action_mask

        self.model.train_on_batch([from_states, action_mask], targets)

        # update the target network every N steps
        self.train_steps += 1
        if self.train_steps % params.TARGET_NETWORK_UPDATE_FREQ == 0:
            self.update_target()
            self.train_steps = 0
