import numpy as np

import dqn.params as params
from dqn.preprocessing import preprocess_frame


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

        # current state
        # can be different from the env state
        # frame stacking, resizing, etc
        self.state = self.get_starting_state()
        self.total_reward = 0

        # don't repeat actions too often
        self.last_action = None
        self.repeat_action_counter = 0

    def interact(self, action):
        """
        applies the action to the environment.
        this takes care of preprocessing observations
        and updating the internal state of the agent
        :param action: action to apply
        :return: new_state, reward, done
        """
        observation, reward, done = self.env.step(action)

        new_frame = preprocess_frame(observation)

        new_state = np.empty_like(self.state)
        new_state[:, :, :-1] = self.state[:, :, 1:]
        new_state[:, :, -1] = new_frame

        return new_state, reward, done

    def act(self):
        """
        chooses an action and applies it to the environment.
        updates the memory to provide training data for the brain
        :return:
        """
        # use the brain to determine the best action for this state
        current_q = self.brain.predict_q(self.state)[0]
        action = current_q.argmax()

        # let agent choose to explore instead
        if np.random.rand() < self.exploration:
            action = np.random.choice(params.NUM_ACTIONS)

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
        new_state, reward, done = self.interact(action)
        reward *= params.REWARD_SCALE
        self.total_reward += reward

        from_state = self.state
        to_state = new_state

        if not done:
            self.state = new_state
        else:
            self.state = self.get_starting_state()
            to_state[:] = 0

            print(self.total_reward)
            self.total_reward = 0

        # the value that should have been predicted
        q_target = self.brain.get_targets(to_state, reward, done)

        # this is what we actually predicted
        q_predicted = current_q[action]

        error = q_target - q_predicted
        error = np.abs(error)
        priority = np.power(error + params.ERROR_BIAS, params.ERROR_POW)

        self.memory.push(from_state, to_state, action, q_target, done, priority)

    def render_frame(self):
        pass

    def render_episode(self):
        pass

    def get_starting_state(self):
        self.state = np.zeros(params.INPUT_SHAPE, dtype=np.uint8)
        self.env.reset()
        frame = self.env.render()
        self.state[:, :, -1] = preprocess_frame(frame)

        action = 0

        # we repeat the action 4 times, since our initial state needs 4 stacked frames
        for i in range(params.FRAME_STACK):
            state, _, _ = self.interact(action)

        return state
