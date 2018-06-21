import time
from threading import Lock

import tensorflow as tf
from colorama import Fore, Style
from keras.models import *

import algorithms.ppo_sequential.params as params
from algorithms.ppo_sequential.conv_models import ConvLSTMModel
from algorithms.ppo_sequential.memory import Memory


class Brain:

    def __init__(self, ModelClass: ConvLSTMModel):

        # use this to influence the tensorflow behaviour
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = params.TF_ALLOW_GROWTH
        config.log_device_placement = params.TF_LOG_DEVICE_PLACEMENT

        self.session = tf.Session(config=config)
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        # set up a model for policy and
        self.model = ModelClass()

        # the model only contains the function approximator
        # the loss function for training is set up here
        self.__setup_training()

        # running tensorflow in a multithreaded environment requires additional setup work
        # and freezing the resulting graph
        # this is the sequential version, but intuitively, freezing the graph sounds like a performance improvement
        # TODO: check if this is useful
        self.model.model._make_predict_function()
        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()
        self.default_graph.finalize()

    def __setup_training(self):
        # due to keras' restrictions on loss functions,
        # we use tensorflow to create a minimization step for the custom loss

        # placeholders
        self.action_mask = Input(shape=(params.NUM_ACTIONS,), name="action_mask")

        self.target_value = Input(shape=(1,), name="target_value")
        self.advantage = Input(shape=(1,), name="advantage")

        # the policies as predicted by old and new network
        # old policy should be cached in the memory, we can feed it here
        self.old_policy = Input(shape=(params.NUM_ACTIONS,), name="old_policy")
        new_policy = self.model.pred_policy

        # masking them, only looking at the action that was actually taken
        old_action = self.old_policy * self.action_mask
        new_action = new_policy * self.action_mask

        old_action = K.sum(old_action, axis=-1, keepdims=True)
        new_action = K.sum(new_action, axis=-1, keepdims=True)

        # set up the policy loss of ppo_sequential
        ratio = K.exp(K.log(new_action) - K.log(old_action))
        loss1 = ratio * self.advantage
        loss2 = tf.clip_by_value(ratio, 1.0 - params.RATIO_CLIP_VALUE, 1.0 + params.RATIO_CLIP_VALUE) * self.advantage
        loss_policy = - tf.reduce_mean(tf.minimum(loss1, loss2))

        # the values as predicted by old and new,
        # again we can feed the cached prediction
        self.old_value = Input(shape=(1,), name="old_value")
        new_value = self.model.pred_value
        new_value_clipped = self.old_value + tf.clip_by_value(new_value - self.old_value, -params.VALUE_CLIP_RANGE,
                                                              params.VALUE_CLIP_RANGE)
        value_loss_1 = (new_value - self.target_value) ** 2
        value_loss_2 = (new_value_clipped - self.target_value) ** 2
        loss_value = 0.5 * tf.reduce_mean(tf.maximum(value_loss_1, value_loss_2))
        loss_value = params.LOSS_VALUE * loss_value

        # the loss contains an entropy component which rewards exploration
        eps = 1e-10
        loss_entropy = - K.sum(new_policy * K.log(new_policy + eps), axis=-1, keepdims=True)
        loss_entropy = params.LOSS_ENTROPY * loss_entropy

        # and we also add regularization
        loss_regularization = self.model.loss_regularization

        # the sum of all losses is
        loss = tf.reduce_sum(loss_policy + loss_value + loss_regularization + loss_entropy)

        # set up a tensorflow minimizer
        new_policy_variables = self.model.trainable_weights
        optimizer = tf.train.AdamOptimizer(learning_rate=params.LEARNING_RATE)
        gradients_variables = optimizer.compute_gradients(loss, new_policy_variables)
        if params.GRADIENT_NORM_CLIP is not None:
            gradients, variables = zip(*gradients_variables)
            gradients, gradient_norms = tf.clip_by_global_norm(gradients, params.GRADIENT_NORM_CLIP)
            gradients_variables = zip(gradients, variables)
        minimize_step = optimizer.apply_gradients(gradients_variables)

        self.minimize_step = minimize_step

    def optimize(self, batch):

        (from_observations, from_states, to_observations, to_states, pred_policies, pred_values, actions, rewards,
         advantages,
         terminals, lengths) = batch

        from_observations = np.array(from_observations)
        from_states = np.array(from_states)
        to_observations = np.array(to_observations)
        to_states = np.array(to_states)
        pred_policies = np.array(pred_policies).reshape((-1, params.NUM_ACTIONS))
        pred_values = np.array(pred_values).reshape((-1, 1))
        actions = np.vstack(actions).reshape((-1, params.NUM_ACTIONS))
        rewards = np.vstack(rewards)
        terminals = np.vstack(terminals)
        advantages = np.vstack(advantages).reshape((-1, 1))
        lengths = np.vstack(lengths)

        num_samples = from_observations.shape[0]

        # predict the final value
        # _, end_values, _ = self.predict(to_observations, to_states)
        # target_values = rewards + params.GAMMA ** length * end_values * (1 - terminals)

        # TODO: again, this is the baseline version. find out why the z normalize the advantages
        target_values = advantages + pred_values
        #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indices = np.arange(num_samples)

        for epoch in range(params.NUM_EPOCHS):
            np.random.shuffle(indices)

            for idx in range(num_samples//params.BATCH_SIZE):
                lower_idx = idx * params.BATCH_SIZE
                upper_idx = (idx + 1) * params.BATCH_SIZE
                batch_indices = indices[lower_idx:upper_idx]

                batch_observations = from_observations[batch_indices]
                batch_states = from_states[batch_indices]
                batch_policies = pred_policies[batch_indices]
                batch_values = pred_values[batch_indices]
                batch_action_mask = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_target_values = target_values[batch_indices]

                # z-normalization
                batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

                # the model is responsible for plugging in observations and states as needed
                new_model_feed_dict = self.model.create_feed_dict(batch_observations, batch_states)

                self.session.run(self.minimize_step, feed_dict={
                    **new_model_feed_dict,
                    self.old_policy: batch_policies,
                    self.old_value: batch_values,
                    self.action_mask: batch_action_mask,
                    self.advantage: batch_advantages,
                    self.target_value: batch_target_values})

        print(Fore.RED+"policy updated"+Style.RESET_ALL)

    # the following methods will simply be routed to the model
    # this routing is not really elegant but I didn't want to expose the model outside of the brain
    def predict(self, observation, state):
        with self.default_graph.as_default():
            return self.model.predict(observation, state)

    def preprocess(self, observation):
        return self.model.preprocess(observation)

    def get_initial_state(self):
        return self.model.get_initial_state()
