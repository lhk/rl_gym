import time

import tensorflow as tf
from colorama import Fore, Style
from keras.models import *

import algorithms.a3c_threading.params as params
from algorithms.a3c_threading.memory import Memory
from util.loss_functions import huber_loss

class Brain:

    def __init__(self, ModelClass, memory: Memory):
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
        self.model.model._make_predict_function()
        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()
        self.default_graph.finalize()

        # a globally shared memory, this will be filled by the asynchronous agents
        self.memory = memory

    def __setup_training(self):
        # due to keras' restrictions on loss functions,
        # we use tensorflow to create a minimization step for the custom loss

        # placeholders
        self.action_mask = Input(shape=(params.NUM_ACTIONS,), name="action_mask")

        self.target_value = Input(shape=(1,), name="target_value")
        self.advantage = Input(shape=(1,), name="advantage")

        pred_values = self.model.pred_value
        policy = self.model.pred_policy

        # loss formulation of A3C
        chosen_action = policy * self.action_mask
        log_prob = K.log(K.sum(chosen_action, axis=-1, keepdims=True))

        # policy is the standard loss for a policy update with advantage function as weight
        loss_policy = -log_prob * self.advantage

        # value is trained on n_step TD-lambda value estimation
        # TODO: this is very high variance, maybe switch to Huber loss
        #loss_value = params.LOSS_VALUE * (self.target_value - pred_values) ** 2
        loss_value = params.LOSS_VALUE * huber_loss(self.target_value, pred_values)

        # entropy is maximized
        eps = 1e-8
        loss_entropy = - params.LOSS_ENTROPY * K.sum(policy * K.log(policy + eps), axis=-1, keepdims=True)

        loss_regularization = self.model.loss_regularization

        loss = tf.reduce_sum(loss_policy + loss_value + loss_regularization + loss_entropy)

        # we have to use tensorflow, this is not possible withing a custom keras loss function
        optimizer = tf.train.AdamOptimizer(learning_rate=params.LEARNING_RATE)
        gradients_variables = optimizer.compute_gradients(loss)
        if params.GRADIENT_NORM_CLIP is not None:
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

        batch = self.memory.pop()

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
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indices = np.arange(num_samples)

        for epoch in range(params.NUM_EPOCHS):
            np.random.shuffle(indices)

            for idx in range(num_samples // params.BATCH_SIZE):
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
                model_feed_dict = self.model.create_feed_dict(batch_observations, batch_states)

                self.session.run(self.minimize_step, feed_dict={
                    **model_feed_dict,
                    self.action_mask: batch_action_mask,
                    self.advantage: batch_advantages,
                    self.target_value: batch_target_values})

        print(Fore.RED)
        print("policy updated")
        print(Style.RESET_ALL)

    # the following methods will simply be routed to the model
    # this routing is not really elegant but I didn't want to expose the model outside of the brain
    def predict(self, observation, state):
        with self.default_graph.as_default():
            return self.model.predict(observation, state)

    def preprocess(self, observation):
        return self.model.preprocess(observation)

    def get_initial_state(self):
        return self.model.get_initial_state()
