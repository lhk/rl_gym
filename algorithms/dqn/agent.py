import numpy as np

import algorithms.dqn.params as params


class Agent:
    def __init__(self, memory, brain, environment):
        """
        :param memory: act stores training samples in this memory
        :param brain: act uses this brain to determine actions
        :param environment: agent interacts with this environment
        """

        self.memory = memory
        self.brain = brain
        self.env = environment

        # the internal state of the agent:
        # current probability of random action
        self.exploration = params.INITIAL_EXPLORATION

        self.reset()

    def reset(self):
        self.observation = self.env.reset()

        # internally, the brain uses a model, and that can have a state, rnns for example
        if self.brain.stateful:
            self.state = self.brain.get_initial_state()
        else:
            self.state = None

        self.observation = self.brain.preprocess(self.observation)

        if params.FRAME_STACK:
            self.observation = np.stack([self.observation] * params.FRAME_STACK, axis=-1)

        self.total_reward = 0

        # don't repeat actions too often
        self.last_action = None
        self.repeat_action_counter = 0

    def act(self):
        """
        chooses an action and applies it to the environment.
        updates the memory to provide training data for the brain
        :return:
        """

        # if the brain is stateful, then the state can change due to this observation
        # we have to show the observation to the brain in any case, to get the new state
        # if the brain is not stateful, we have to get predictions for this observation only if we decide to exploit
        # if the brain is not stateful and we decide to explore, there is no need for predicting the q values
        # this is a potentially huge speedup
        if self.brain.stateful:
            # use the brain to determine the best action for this state
            current_q, to_state = self.brain.predict_q(self.observation, self.state)
            current_q = current_q[0]
            action = current_q.argmax()

            # maybe overwrite the action
            if np.random.rand() < self.exploration:
                action = np.random.choice(params.NUM_ACTIONS)
        else:
            # exploration vs exploitation
            if np.random.rand() < self.exploration:
                action = np.random.choice(params.NUM_ACTIONS)
            else:
                # use the brain to determine the best action for this state
                current_q, to_state = self.brain.predict_q(self.observation, self.state)
                current_q = current_q[0]
                action = current_q.argmax()

            to_state = None

        # anneal exploration
        if self.exploration > params.FINAL_EXPLORATION:
            self.exploration -= params.EXPLORATION_STEP

        # we only allow a limited amount of repeated actions
        if action == self.last_action:
            self.repeat_action_counter += 1

            if self.repeat_action_counter > params.REPEAT_ACTION_MAX:
                action = self.env.sample_action()
                self.last_action = action
                self.repeat_action_counter = 0
        else:
            self.last_action = action
            self.repeat_action_counter = 0

        # interact with the environment
        new_observation, reward, done, _ = self.env.step(action)
        reward *= params.REWARD_SCALE
        self.total_reward += reward

        # preprocess the new observation
        from_observation = self.observation
        from_state = self.state

        new_observation = self.brain.preprocess(new_observation)

        if params.FRAME_STACK:
            to_observation = np.zeros_like(self.observation)
            to_observation[:, :, :-1] = self.observation[:, :, 1:]
            to_observation[:, :, -1] = new_observation
        else:
            to_observation = new_observation

        if not done:
            self.observation = to_observation
            self.state = to_state
        else:
            print(self.total_reward)
            self.reset()

        # the value that should have been predicted
        # q_target = self.brain.get_targets(to_state, reward, done)

        # this is what we actually predicted
        # q_predicted = current_q[action]

        # error = q_target - q_predicted
        # error = np.abs(error)
        # priority = np.power(error + params.ERROR_BIAS, params.ERROR_POW)

        # new observations are pushed to the memory with a default priority
        # this means that for most interactions, we don't need to use the brain
        # if the agent is exploring, we don't have to calculate any q values
        self.memory.push(from_observation, to_observation, from_state, to_state, action, reward, done,
                         params.DEFAULT_PRIO)
