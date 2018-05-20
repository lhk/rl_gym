import numpy as np
import gym
import dqn.params as params
import lycon
class Agent:

    def __init__(self, exploration = params.INITIAL_EXPLORATION):
        self.env = gym.make(params.ENV_NAME)
        self.env.seed(0)

        self.exploration = exploration

        self.last_action = None
        self.repeat_action_counter = 0

        self.state = self.get_starting_state()
        self.total_reward = 0

    def preprocess_frame(self, frame):
        downsampled = lycon.resize(frame, width=params.FRAME_SIZE[0], height=params.FRAME_SIZE[1],
                                   interpolation=lycon.Interpolation.NEAREST)
        grayscale = downsampled.mean(axis=-1).astype(np.uint8)
        return grayscale

    def interact(self, action):
        observation, reward, done, _ = self.env.step(action)

        new_frame = self.preprocess_frame(observation)

        new_state = np.empty_like(self.state)
        new_state[:, :, :-1] = self.state[:, :, 1:]
        new_state[:, :, -1] = new_frame

        return new_state, reward, done


    def get_starting_state(self):
        self.state = np.zeros(params.INPUT_SHAPE, dtype=np.uint8)
        frame = self.env.reset()
        self.state[:, :, -1] = self.preprocess_frame(frame)

        action = 0

        # we repeat the action 4 times, since our initial state needs 4 stacked frames
        for i in range(params.FRAME_STACK):
            state, _, _ = self.interact(action)

        return state

    def act(self, action):

        # exploit or explore
        if np.random.rand() < self.exploration:
            action = self.env.action_space.sample()

        # anneal exploration
        if self.exploration > params.FINAL_EXPLORATION:
            self.exploration-=params.EXPLORATION_STEP

        # we only allow a limited amount of repeated actions
        if action == self.last_action:
            self.repeat_action_counter += 1

            if self.repeat_action_counter > params.REPEAT_ACTION_MAX:
                action = 1 # self.env.action_space.sample()
                self.last_action = action
                self.repeat_action_counter = 0
        else:
            self.last_action = action
            self.repeat_action_counter = 0

        new_state, reward, done = self.interact(action)

        # done means the environment had to restart, this is bad
        # please note: the restart reward is chosen as -1
        # the rewards are clipped to [-1, 1] according to the paper
        # if that would not be done, we would have to scale this reward
        # to align to the other replays given in the game
        if done:
            reward = - 1

        self.total_reward += reward

        from_state = self.state
        to_state = new_state

        if not done:
            self.state = new_state
        else:
            self.state = self.get_starting_state()
            print(self.total_reward)
            self.total_reward=0

        return from_state, to_state, action, reward, done