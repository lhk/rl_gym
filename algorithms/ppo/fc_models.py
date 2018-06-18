from keras.layers import *
from keras.models import *
from keras.regularizers import l2

import algorithms.ppo.params as params


class FullyConnectedModel():
    def __init__(self):
        # some parameters now belong to the model
        self.INPUT_SHAPE = (7,)
        self.FC_SIZE = 32

        # build a model to predict action probabilities and values
        self.input_observation = Input(shape=(*self.INPUT_SHAPE,))
        # bnorm = BatchNormalization()(self.input_observation)

        hidden = Dense(self.FC_SIZE, activation='relu', kernel_regularizer=l2(params.L2_REG_FULLY))(
            self.input_observation)
        bnorm = BatchNormalization()(hidden)

        hidden = Dense(self.FC_SIZE, activation='relu', kernel_regularizer=l2(params.L2_REG_FULLY))(bnorm)
        bnorm = BatchNormalization()(hidden)

        pred_policy = Dense(params.NUM_ACTIONS, activation='softmax', kernel_regularizer=l2(params.L2_REG_FULLY))(bnorm)
        pred_value = Dense(1, activation='linear', kernel_regularizer=l2(params.L2_REG_FULLY))(hidden)

        model = Model(inputs=[self.input_observation], outputs=[pred_policy, pred_value])

        # the model is not compiled with any loss function
        # but the regularizers are still exposed as losses
        loss_regularization = sum(model.losses)

        # the model and its inputs
        self.model = model

        # the weights that can be updated
        self.trainable_weights = model.trainable_weights

        # tensors, these will be used for loss formulations
        self.pred_policy = pred_policy
        self.pred_value = pred_value
        self.loss_regularization = loss_regularization

    def preprocess(self, observation):
        return observation

    def get_initial_state(self):
        return []

    def predict(self, observation, state):
        # keras always needs a batch dimension
        if observation.shape == self.INPUT_SHAPE:
            observation = observation.reshape((-1, *self.INPUT_SHAPE))

        return [*self.model.predict(observation), []]

    def create_feed_dict(self, observation, state):
        return {self.input_observation: observation}
