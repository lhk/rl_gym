import lycon
import ppo.params as params
from keras.layers import *
from keras.models import *
from keras.regularizers import l2


class ConvLSTMModel():
    def __init__(self):
        # some parameters now belong to the model
        self.FRAME_SIZE = (84, 84)
        self.INPUT_SHAPE = (*self.FRAME_SIZE, 3)
        self.RNN_SIZE = 64

        # build a model to predict action probabilities and values
        self.input_observation = Input(shape=(*self.INPUT_SHAPE,))

        rescaled = Lambda(lambda x: x / 255.)(self.input_observation)
        conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu', kernel_regularizer=l2(params.L2_REG_CONV))(
            rescaled)
        conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', kernel_regularizer=l2(params.L2_REG_CONV))(conv)

        conv_flattened = Flatten()(conv)
        dense = Dense(64, activation="relu", kernel_regularizer=l2(params.L2_REG_FULLY))(conv_flattened)

        # shape = [batch_size, time_steps, input_dim]
        dense = Reshape((1, 64))(dense)

        # apply an rnn
        # expose the state of the cell, so that we can recreate the setup
        # of the cell during training
        gru_cell = GRU(self.RNN_SIZE, return_state=True, kernel_regularizer=l2(params.L2_REG_FULLY))
        self.input_state = Input(shape=(self.RNN_SIZE,))
        gru_tensor, output_memory = gru_cell(dense, initial_state=self.input_state)

        pred_policy = Dense(params.NUM_ACTIONS, activation='softmax', kernel_regularizer=l2(params.L2_REG_FULLY))(
            gru_tensor)
        pred_value = Dense(1, activation='linear', kernel_regularizer=l2(params.L2_REG_FULLY))(gru_tensor)

        model = Model(inputs=[self.input_observation, self.input_state],
                      outputs=[pred_policy, pred_value, output_memory])

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
        downsampled = lycon.resize(observation, width=self.FRAME_SIZE[0], height=self.FRAME_SIZE[1],
                                   interpolation=lycon.Interpolation.NEAREST)

        # grayscale = downsampled.mean(axis=-1)
        return downsampled.reshape((self.INPUT_SHAPE))

    def get_initial_state(self):
        # this model has a state: the rnn cell
        # the initial state of this rnn cell is given by the following code
        # read from the keras source for initializers
        return np.random.rand(self.RNN_SIZE) * 0.1 - 0.05

    def predict(self, observation, state):

        # keras always needs a batch dimension
        if observation.shape == self.INPUT_SHAPE:
            observation = observation.reshape((-1, *self.INPUT_SHAPE))

        if state.shape == (self.RNN_SIZE,):
            state = state.reshape((-1, self.RNN_SIZE))

        return self.model.predict([observation, state])

    def create_feed_dict(self, observation, state):
        return {self.input_observation: observation,
                self.input_state: state}
