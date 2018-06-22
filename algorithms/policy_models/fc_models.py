from keras.layers import *
from keras.models import *
from keras.regularizers import l2

import algorithms.ppo_sequential.params as params


class FCModel():
    INPUT_SHAPE = (7,)
    FC_SIZE = 64
    NUM_HIDDEN_LAYERS = 2
    def __init__(self):
        # some parameters now belong to the model

        # build a model to predict action probabilities and values
        self.input_observation = Input(shape=(*self.INPUT_SHAPE,))

        # predicting policy
        layer = self.input_observation
        for _ in range(self.NUM_HIDDEN_LAYERS):
            layer = Dense(self.FC_SIZE, activation="tanh", kernel_regularizer=l2(params.L2_REG_FULLY))(layer)

        pred_policy = Dense(params.NUM_ACTIONS, activation='softmax', kernel_regularizer=l2(params.L2_REG_FULLY))(layer)

        # predicting value
        layer = self.input_observation
        for _ in range(self.NUM_HIDDEN_LAYERS):
            layer = Dense(self.FC_SIZE, activation="tanh", kernel_regularizer=l2(params.L2_REG_FULLY))(layer)
        pred_value = Dense(1, kernel_regularizer=l2(params.L2_REG_FULLY))(layer)

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

class FCCartPole(FCModel):
    INPUT_SHAPE = (4,)
    FC_SIZE = 16
    NUM_HIDDEN_LAYERS = 1

    def __init__(self):
        FCModel.__init__(self)

class FCRadialCar(FCModel):
    INPUT_SHAPE = (7,)
    FC_SIZE = 32
    NUM_HIDDEN_LAYERS = 2

    def __init__(self):
        FCModel.__init__(self)