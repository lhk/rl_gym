import numpy as np

np.random.seed(0)

import tensorflow as tf
from keras.layers import Input
import keras.backend as K

import algorithms.dqn.params as params
from algorithms.dqn.memory import Memory
import os
import shutil


class Brain:
    def __init__(self, Model, memory: Memory, loss_func="mse", load_path=None):

        self.memory = memory
        # use this to influence the tensorflow behaviour
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = params.TF_ALLOW_GROWTH
        config.log_device_placement = params.TF_LOG_DEVICE_PLACEMENT

        self.sess = tf.Session(config=config)
        K.set_session(self.sess)

        # dqn works on two models
        self.model = Model()
        self.target_model = Model()
        self.stateful = Model.STATEFUL

        self.loss_func = loss_func

        # set up ops for training
        self.__setup_training()

        self.target_updates = 0

        # cleaning a directory for checkpoints
        if os.path.exists(os.getcwd() + "/checkpoints/"):
            shutil.rmtree(os.getcwd() + "/checkpoints/")
        os.mkdir(os.getcwd() + "/checkpoints/")

    def __setup_training(self):

        self.q_target = Input(shape=(params.NUM_ACTIONS,))

        loss_q = self.loss_func(self.q_target, self.model.q_values_masked)

        loss = loss_q + self.model.loss_regularization

        model_variables = self.model.trainable_weights
        target_model_variables = self.target_model.trainable_weights
        optimizer = tf.train.RMSPropOptimizer(learning_rate=params.LEARNING_RATE, decay=params.RHO, epsilon=params.EPSILON)
        gradients_variables = optimizer.compute_gradients(loss, model_variables)
        if not params.GRADIENT_NORM_CLIP is None:
            gradients, variables = zip(*gradients_variables)
            gradients, gradient_norms = tf.clip_by_global_norm(gradients, params.GRADIENT_NORM_CLIP)
            gradients_variables = zip(gradients, variables)
        minimize_step = optimizer.apply_gradients(gradients_variables)

        self.minimize_step = minimize_step

        self.assignments = [tf.assign(to_var, from_var) for (to_var, from_var) in
                            zip(target_model_variables, model_variables)]

    # the following methods will simply be routed to the model
    # this routing is not really elegant but I didn't want to expose the model outside of the brain

    def preprocess(self, observation):
        return self.model.preprocess(observation)

    def get_initial_state(self):
        return self.model.get_initial_state()

    def predict_q(self, observation, state):
        return self.model.predict(observation, state)

    def predict_q_target(self, observation, state):
        return self.target_model.predict(observation, state)

    def get_targets(self, to_observations, to_states, rewards, done):
        next_q_target, _ = self.predict_q_target(to_observations, to_states)
        next_q, _ = self.predict_q(to_observations, to_states)
        chosen_q = next_q_target[np.arange(next_q.shape[0]), next_q.argmax(axis=-1)]

        # this is the value that should be predicted by the network
        q_targets = rewards + params.GAMMA * chosen_q * (1 - done)

        return q_targets

    def update_target_model(self):
        self.sess.run(self.assignments)

        self.target_updates += 1

        # save the target network every N steps
        if self.target_updates % params.SAVE_NETWORK_FREQ == 0:
            self.target_model.model.save("checkpoints/dqn_model{}.hd5".format(self.target_updates + 120))

    def train_once(self):

        # sample batch with priority as weight, train on it
        training_indices = self.memory.sample_indices()
        batch = self.memory[training_indices]

        from_observations, to_observations, from_states, to_states, actions, rewards, terminals = batch

        # create a one-hot mask for the actions
        action_mask = np.zeros((actions.shape[0], params.NUM_ACTIONS))
        action_mask[np.arange(actions.shape[0]), actions] = 1

        q_targets = self.get_targets(to_observations, to_states, rewards, terminals)
        q_targets = q_targets.reshape((-1, 1)) * action_mask

        model_feed_dict = self.model.create_feed_dict(from_observations, from_states, action_mask)
        self.sess.run(self.minimize_step, feed_dict={
            **model_feed_dict,
            self.q_target: q_targets
        })

        if (self.memory.priority_based_sampling):
            q_predicted, _ = self.predict_q(from_observations, from_states)
            q_predicted *= action_mask

            errors = np.abs(q_targets - q_predicted)
            errors = errors.sum(axis=-1)
            priorities = np.power(errors + params.ERROR_BIAS, params.ERROR_POW)

            self.memory.update_priority(training_indices, priorities)
