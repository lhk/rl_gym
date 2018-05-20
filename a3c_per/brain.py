import time

import tensorflow as tf
from keras.layers import *
from keras.models import *

import a3c_per.params as params
from a3c_per.memory import Memory


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
        model, input_state, action_mask, t_step_reward, minimize_step = self.__setup_model()
        self.model = model
        self.input_state = input_state
        self.action_mask = action_mask
        self.t_step_reward = t_step_reward
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
        input_state = Input(shape=(params.NUM_STATE,))

        dense = Dense(16, activation='relu')(input_state)

        pred_actions = Dense(params.NUM_ACTIONS, activation='softmax')(dense)
        pred_values = Dense(1, activation='linear')(dense)

        model = Model(inputs=[input_state], outputs=[pred_actions, pred_values])

        # due to keras' restrictions on loss functions,
        # the above can't be trained with a simple keras loss function
        # we use tensorflow to create a minimization step for the custom loss

        # placeholders
        action_mask = Input(shape=(params.NUM_ACTIONS,))
        n_step_reward = Input(shape=(1,))

        # loss formulation of A3C
        chosen_action = pred_actions * action_mask
        log_prob = K.log(K.sum(chosen_action, axis=-1, keepdims=True))

        advantage = n_step_reward - pred_values

        loss_policy = -log_prob * advantage
        loss_value = params.LOSS_VALUE * advantage ** 2

        eps = 1e-10
        entropy = params.LOSS_ENTROPY * K.sum(pred_actions * K.log(pred_actions + eps), axis=-1, keepdims=True)

        loss = loss_policy + loss_value + entropy

        # we have to use tensorflow, this is not possible withing a custom keras loss function
        rmsprop = tf.train.RMSPropOptimizer(learning_rate=params.LEARNING_RATE, decay=params.DECAY)
        minimize_step = rmsprop.minimize(loss)

        return model, input_state, action_mask, n_step_reward, minimize_step

    def __get_targets(self, batch):
        from_states, to_states, actions, rewards, terminal, length = batch
        from_states = np.vstack(from_states)
        to_states = np.vstack(to_states)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)
        terminal = np.vstack(terminal)
        length = np.vstack(length)

        # predict the final value
        _, end_values = self.predict(to_states)
        n_step_reward = rewards + params.GAMMA ** length * end_values * (1 - terminal)

        return n_step_reward

    def optimize(self):

        # yield control if there is not enough training data in the memory
        if len(self.memory) < params.MIN_BATCH:
            time.sleep(0)
            return

        # get up to MAX_BATCH items from the training queue
        sample_indices = self.memory.sample_indices(params.MAX_BATCH)
        batch = self.memory[sample_indices]

        # train on batch
        from_states, to_states, actions, _, _, _ = batch
        from_states = np.vstack(from_states)
        actions = np.vstack(actions)

        n_step_reward = self.__get_targets(batch)

        self.session.run(self.minimize_step, feed_dict={
            self.input_state: from_states,
            self.action_mask: actions,
            self.t_step_reward: n_step_reward})

        # update priorities
        errors = self.get_error(batch)
        priorities = (errors + params.ERROR_BIAS)**params.ERROR_POW
        self.memory.update_priority(sample_indices, priorities)


    def get_error(self, batch):
        observed_value = self.__get_targets(batch)
        from_states, _, _, _, _, _  = batch
        from_states = np.vstack(from_states)

        _, predicted_value = self.predict(from_states)

        error = np.abs(observed_value - predicted_value)
        return error

    def predict(self, state):
        # keras always needs a batch dimension
        if state.shape == params.INPUT_SHAPE:
            state = state.reshape((-1, *params.INPUT_SHAPE))

        with self.default_graph.as_default():
            return self.model.predict(state)
