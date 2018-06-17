import time

import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras.regularizers import l2
import ppo.params as params
from ppo.memory import Memory
from ppo.conv_models import ConvLSTMModel
from threading import Lock

from colorama import Fore, Style


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
        self.__setup_training()

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

    def __setup_training(self):
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
        loss2 = tf.clip_by_value(ratio, 1.0 - params.RATIO_CLIP_VALUE, 1.0 + params.RATIO_CLIP_VALUE) * self.advantage
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

        self.assignments = [tf.assign(to_var, from_var) for (to_var, from_var) in zip(self.old_model.trainable_weights, self.new_model.trainable_weights)]

    def optimize(self):
        # we train on blocks of this size
        num_samples = params.BATCH_SIZE * params.NUM_BATCHES

        # yield control if there is not enough training data in the memory
        if len(self.memory) < num_samples:
            time.sleep(0)
            return

        # get the training items from the training queue
        batch = self.memory.pop(num_samples)

        from_observations, from_states, to_observations, to_states, actions, rewards, advantages, terminals, length = batch
        from_observations = np.array(from_observations)
        from_states = np.array(from_states)
        to_observations = np.array(to_observations)
        to_states = np.array(to_states)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)
        terminals = np.vstack(terminals)
        advantages = np.vstack(advantages)
        length = np.vstack(length)

        # TODO: is this necessary
        # after reading the openAI baseline, I'm adding some small tweaks to my implementation
        # such as z-normalizing the advantages in a batch
        advantages = (advantages - advantages.mean())/(advantages.std() + 1e-8)

        # predict the final value
        _, end_values, _ = self.predict(to_observations, to_states)
        target_values = rewards + params.GAMMA ** length * end_values * (1 - terminals)

        # now we iterate through the training data
        # for each iteration, we slice a block of BATCH_SIZE out of it
        for idx in range(params.NUM_BATCHES):
            lower_idx = idx*params.BATCH_SIZE
            upper_idx = (idx+1)*params.BATCH_SIZE

            batch_observations = from_observations[lower_idx : upper_idx]
            batch_states = from_states[lower_idx : upper_idx]
            batch_action_mask = actions[lower_idx : upper_idx]
            batch_advantages = advantages[lower_idx : upper_idx]
            batch_target_values = target_values[lower_idx : upper_idx]

            # the model is responsible for plugging in observations and states as needed
            old_model_feed_dict = self.old_model.create_feed_dict(batch_observations, batch_states)
            new_model_feed_dict = self.new_model.create_feed_dict(batch_observations, batch_states)

            self.session.run(self.minimize_step, feed_dict={
                **old_model_feed_dict,
                **new_model_feed_dict,
                self.action_mask: batch_action_mask,
                self.advantage: batch_advantages,
                self.target_value: batch_target_values})

        # after some training runs, we update the model
        # the infrastructure allows more than one optimizer
        # tensorflow graphs are threadsafe, this num_updates is the only volatile data here
        with self.lock:
            self.num_updates+=1
            if self.num_updates > params.NUM_UPDATES:
                self.update_model()
                self.num_updates = 0
                print(Fore.RED + "update" + Style.RESET_ALL)

    # the following methods will simply be routed to the model
    # this routing is not really elegant but I didn't want to expose the model outside of the brain
    def predict(self, observation, state):
        with self.default_graph.as_default():
            return self.old_model.predict(observation, state)

    def preprocess(self, observation):
        return self.old_model.preprocess(observation)

    def get_initial_state(self):
        return self.old_model.get_initial_state()

    def update_model(self):
        with self.default_graph.as_default():
            self.session.run(self.assignments)
