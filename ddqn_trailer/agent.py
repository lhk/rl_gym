from environments.trailer_env.environment import Environment
import lycon
import numpy as np

import ddqn_trailer.params as params

import sys
import cv2
import pygame
from pygame.locals import *

import os


class Agent:

    def __init__(self, exploration=params.INITIAL_EXPLORATION, vis=False, store_vis = False):
        self.env = Environment()
        self.actions = [[1, 0], [0, 0], [0, -1], [0, 1]]
        # [[1, -1], [1, 0], [1, 1],
        # [0, -1], [0, 0], [0, 1]]
        # [-1, -1], [-1, 0], [-1, 1]]

        self.exploration = exploration

        self.last_action = None
        self.repeat_action_counter = 0

        if not vis:
            assert not store_vis, "can only store visualization if vis==True"

        self.vis = vis
        if self.vis:
            pygame.init()
            self.clock = pygame.time.Clock()
            self.window = pygame.display.set_mode(params.FRAME_SIZE)
            pygame.display.set_caption("Pygame cheat sheet")

        self.store_vis = store_vis
        self.rendered_images = []
        self.video_num = 0

        self.state = self.get_starting_state()
        self.total_reward = 0



    def preprocess_frame(self, frame):
        downsampled = lycon.resize(frame, width=params.FRAME_SIZE[0], height=params.FRAME_SIZE[1],
                                   interpolation=lycon.Interpolation.NEAREST)
        grayscale = downsampled.mean(axis=-1).astype(np.uint8)
        return grayscale

    def interact(self, action):
        action = self.actions[action]
        reward, done = self.env.make_action(action)
        observation = self.env.render()

        new_frame = self.preprocess_frame(observation)

        new_state = np.empty_like(self.state)
        new_state[:, :, :-1] = self.state[:, :, 1:]
        new_state[:, :, -1] = new_frame

        return new_state, reward, done

    def get_starting_state(self):
        self.state = np.zeros(params.INPUT_SHAPE, dtype=np.uint8)
        self.env.new_episode()
        frame = self.env.render()
        self.state[:, :, -1] = self.preprocess_frame(frame)

        action = 0

        # we repeat the action 4 times, since our initial state needs 4 stacked frames
        for i in range(params.FRAME_STACK):
            state, _, _ = self.interact(action)

        # this marks the start of a new episode
        # if we have set the flag to render to file
        # dump the recorded images to a file
        if self.store_vis:
            if self.rendered_images:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                video_writer = cv2.VideoWriter(os.getcwd()+"/"+params.OUTPUT_DIR+"/"+str(self.video_num)+".avi",
                                               fourcc, params.FPS, params.FRAME_SIZE)

                for image in self.rendered_images:
                    # cv2 wants 3 channels for color information
                    image = np.stack([image.T]*3, axis=-1).astype(np.uint8)
                    video_writer.write(image)

                video_writer.release()
                self.rendered_images=[]
                self.video_num+=1

        return state

    def act(self, brain):
        # exploit or explore
        if np.random.rand() < self.exploration:
            action = np.random.choice(params.NUM_ACTIONS)
        else:
            # use the brain to determine the best action for this state
            q_values = brain.predict_q(self.state)
            action = q_values.argmax(axis=1)
            action = action[0]

        # anneal exploration
        if self.exploration > params.FINAL_EXPLORATION:
            self.exploration -= params.EXPLORATION_STEP

        # we only allow a limited amount of repeated actions
        if action == self.last_action:
            self.repeat_action_counter += 1

            if self.repeat_action_counter > params.REPEAT_ACTION_MAX:
                action = np.random.choice(params.NUM_ACTIONS)
                self.last_action = action
                self.repeat_action_counter = 0
        else:
            self.last_action = action
            self.repeat_action_counter = 0

        new_state, reward, done = self.interact(action)
        reward *= params.REWARD_SCALE

        self.total_reward += reward

        from_state = self.state
        to_state = new_state

        if not done:
            self.state = new_state
        else:
            self.state = self.get_starting_state()
            print(self.total_reward)
            self.total_reward = 0

        if self.vis:
            rendered_image = from_state[:, :, -1]
            self.rendered_images.append(rendered_image)
            render_surf = pygame.surfarray.make_surface(rendered_image)
            self.window.blit(render_surf, (0, 0))

            self.clock.tick(10)
            pygame.display.update()

        return from_state, to_state, action, reward, done
