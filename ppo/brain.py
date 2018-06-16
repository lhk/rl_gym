import time

import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras.regularizers import l2
import ppo.params as params
from ppo.memory import Memory
from ppo.model import ConvLSTMModel
from threading import Lock


class Brain:

    def __init__(self, memory: Memory, ModelClass: ConvLSTMModel):
        # use this to influence the tensorflow behaviour
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = params.TF_ALLOW_GROWTH
        config.log_device_placement = params.TF_LOG_DEVICE_PLACEMENT

        self.session = tf.Session(config=config)
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        # set up a model for policy and values
        # set up placeholders for the inputs during training
        self.old_model = ModelClass()
        self.new_model = ModelClass()

        # the model only contains the function approximator
        # the loss function for training is set up here
        self.__setup_losses()

        # running tensorflow in a multithreaded environment requires additional setup work
        # and freezing the resulting graph
        self.old_model.model._make_predict_function()
        self.new_model.model._make_predict_function()
        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()
        self.default_graph.finalize()

        # a globally shared memory, this will be filled by the asynchronous agents
        self.memory = memory

        # updating old to new policy needs to be synchronized
        self.lock = Lock()
        self.num_updates = 0

    def __setup_losses(self):

        # due to keras' restrictions on loss functions,
        # we use tensorflow to create a minimization step for the custom loss

        # placeholders
        self.action_mask = Input(shape=(params.NUM_ACTIONS,))

        self.target_value = Input(shape=(1,))
        self.advantage = Input(shape=(1,))

        # the policies as predicted by old and new network
        old_policy = self.old_model.pred_policy
        new_policy = self.new_model.pred_policy

        # masking them, only looking at the action that was actually taken
        old_action = old_policy * self.action_mask
        new_action = new_policy * self.action_mask

        old_action = K.sum(old_action, axis=-1, keepdims=True)
        new_action = K.sum(new_action, axis=-1, keepdims=True)

        # creating the ratio for ppo
        ratio = K.exp(K.log(new_action) - K.log(old_action))

        # ppo looks at two losses for the policy
        # their minimum is maximized
        loss1 = ratio * self.advantage
        loss2 = tf.clip_by_value(ratio, 1.0 - params.RATIO_CLIP_VALUE, 1.0 + params.RATIO_CLIP_VALUE)
        loss_policy = - tf.reduce_mean(tf.minimum(loss1, loss2))

        # the next component of the loss is the value function
        # this is the same as for A3C: an n_step TD-lambda
        # TODO: since we look many steps into the future, this will be high variance. Replace it with huber loss
        pred_value = self.new_model.pred_value
        loss_value = params.LOSS_VALUE * (self.target_value - pred_value) ** 2

        # the loss contains an entropy component which rewards exploration
        eps = 1e-10
        loss_entropy = - params.LOSS_ENTROPY * K.sum(new_policy * K.log(new_policy + eps), axis=-1, keepdims=True)

        # and we also add regularization
        loss_regularization = self.new_model.loss_regularization
        loss = tf.reduce_sum(loss_policy + loss_value + loss_regularization + loss_entropy)

        # we have to use tensorflow, this is not possible withing a custom keras loss function
        new_policy_variables = self.new_model.trainable_weights
        optimizer = tf.train.AdamOptimizer(learning_rate=params.LEARNING_RATE)
        gradients_variables = optimizer.compute_gradients(loss, new_policy_variables)
        gradients, variables = zip(*gradients_variables)
        gradients, gradient_norms = tf.clip_by_global_norm(gradients, params.GRADIENT_NORM_CLIP)
        gradients_variables = zip(gradients, variables)
        minimize_step = optimizer.apply_gradients(gradients_variables)

        self.minimize_step = minimize_step

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
        _, end_values, _ = self.predict(to_states, to_memories)
        n_step_reward = rewards + params.GAMMA ** length * end_values * (1 - terminal)

        self.session.run(self.minimize_step, feed_dict={
            self.old_model.input_state: from_states,
            self.new_model.input_state: from_states,
            self.old_model.input_memory: from_memories,
            self.new_model.input_memory: from_memories,
            self.action_mask: actions,
            self.advantage: advantages,
            self.target_value: n_step_reward})

        #print("step")

    def predict(self, state, memory):
        # keras always needs a batch dimension
        if state.shape == params.INPUT_SHAPE:
            state = state.reshape((-1, *params.INPUT_SHAPE))

        # the memory shape is given by the number of cells in the rnn layer
        # I don't want to move that to a parameter, so right now, it is a "magic number"
        # TODO: maybe a parameter after all ?
        if memory.shape == (params.RNN_SIZE,):
            memory = memory.reshape((-1, params.RNN_SIZE))

        with self.default_graph.as_default():
            return self.old_model.model.predict([state, memory])

    def update_model(self):
        self.old_model.model.set_weights(self.new_model.model.get_weights())
