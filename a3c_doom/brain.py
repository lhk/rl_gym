import time

import tensorflow as tf
from keras.layers import *
from keras.models import *

import a3c_doom.params as params
from a3c_doom.memory import Memory


class Brain:

    def __init__(self, memory: Memory):
        # use this to influence the tensorflow behaviour
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = params.TF_ALLOW_GROWTH
        config.log_device_placement = params.TF_LOG_DEVICE_PLACEMENT

        self.session = tf.Session(config=config)
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        # set up a model for policy and values
        # set up placeholders for the inputs during training
        model, input_state, input_memory, action_mask, t_step_reward, advantage, minimize_step = self.__setup_model()

        self.model = model
        self.input_state = input_state
        self.input_memory = input_memory
        self.action_mask = action_mask
        self.t_step_reward = t_step_reward
        self.advantage = advantage
        self.minimize_step = minimize_step

        # running tensorflow in a multithreaded environment requires additional setup work
        # and freezing the resulting graph
        self.model._make_predict_function()
        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()
        self.default_graph.finalize()

        # a globally shared memory, this will be filled by the asynchronous agents
        self.memory = memory

    def __setup_model(self):

        # build a model to predict action probabilities and values
        input_state = Input(shape=(*params.INPUT_SHAPE,))

        rescaled = Lambda(lambda x: x / 255.)(input_state)
        conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(rescaled)
        conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)

        conv_flattened = Flatten()(conv)
        dense = Dense(256, activation="relu")(conv_flattened)

        # shape = [batch_size, time_steps, input_dim]
        dense = Reshape((1, 256))(dense)

        # apply an rnn
        # expose the state of the cell, so that we can recreate the setup
        # of the cell during training
        gru_cell = CuDNNGRU(256, return_state=True)
        input_memory = Input(shape=(256,))
        gru_tensor, output_memory = gru_cell(dense, initial_state=input_memory)

        pred_actions = Dense(params.NUM_ACTIONS, activation='softmax')(gru_tensor)
        pred_values = Dense(1, activation='linear')(gru_tensor)

        model = Model(inputs=[input_state, input_memory], outputs=[pred_actions, pred_values, output_memory])

        # due to keras' restrictions on loss functions,
        # the above can't be trained with a simple keras loss function
        # we use tensorflow to create a minimization step for the custom loss

        # placeholders
        action_mask = Input(shape=(params.NUM_ACTIONS,))

        # TODO: rename n_step_reward, this also includes the target value at the end
        n_step_reward = Input(shape=(1,))
        advantage = Input(shape=(1,))

        # loss formulation of a3c_doom
        chosen_action = pred_actions * action_mask
        chosen_action = K.sum(chosen_action, axis=-1, keepdims=True)
        log_prob = K.log(chosen_action)

        loss_policy = -log_prob * advantage
        loss_value = params.LOSS_VALUE * (n_step_reward - pred_values) ** 2

        eps = 1e-10
        entropy = params.LOSS_ENTROPY * K.sum(pred_actions * K.log(pred_actions + eps), axis=-1, keepdims=True)

        loss = loss_policy + loss_value + entropy

        # we have to use tensorflow, this is not possible withing a custom keras loss function
        optimizer = tf.train.AdamOptimizer(learning_rate=params.LEARNING_RATE)
        gradients_variables = optimizer.compute_gradients(loss)
        gradients, variables = zip(*gradients_variables)
        gradients, gradient_norms = tf.clip_by_global_norm(gradients, params.GRADIENT_NORM_CLIP)
        gradients_variables = zip(gradients, variables)
        minimize_step = optimizer.apply_gradients(gradients_variables)

        return model, input_state, input_memory, action_mask, n_step_reward, advantage, minimize_step

    def optimize(self):

        # yield control if there is not enough training data in the memory
        if len(self.memory) < params.MIN_BATCH:
            time.sleep(0)
            return

        # get up to MAX_BATCH items from the training queue
        from_states, from_memories, to_states, to_memories, actions, rewards, advantages, terminal, length = self.memory.pop(
            params.MAX_BATCH)
        from_states = np.array(from_states)
        from_memories = np.array(from_memories)
        to_states = np.array(to_states)
        to_memories = np.array(to_memories)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)
        terminal = np.vstack(terminal)
        advantages = np.vstack(advantages)
        length = np.vstack(length)

        # predict the final value
        # TODO: this is incorrect, if the local memory of the states unrols after an episode end. it might not be N steps into the future
        _, end_values, _ = self.predict(to_states, to_memories)
        n_step_reward = rewards + params.GAMMA ** length * end_values * (1 - terminal)

        self.session.run(self.minimize_step, feed_dict={
            self.input_state: from_states,
            self.input_memory: from_memories,
            self.action_mask: actions,
            self.advantage : advantages,
            self.t_step_reward: n_step_reward})

        print("step")

    def predict(self, state, memory):
        # keras always needs a batch dimension
        if state.shape == params.INPUT_SHAPE:
            state = state.reshape((-1, *params.INPUT_SHAPE))

        # the memory shape is given by the number of cells in the rnn layer
        # I don't want to move that to a parameter, so right now, it is a "magic number"
        # TODO: maybe a parameter after all ?
        if memory.shape == (256,):
            memory = memory.reshape((-1, 256))

        with self.default_graph.as_default():
            return self.model.predict([state, memory])
