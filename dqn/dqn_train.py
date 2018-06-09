import numpy as np
import tensorflow as tf  # if tf is not imported first, it crashes :)
from tqdm import tqdm

print(tf.GRAPH_DEF_VERSION)  # and if I don't use it, autoformatting gets rid of it

import dqn.params as params
import environments.obstacle_car_graphical.params as env_params
from dqn.agent import PER_Agent
from dqn.brain import DQN_Brain
from dqn.memory import Priority_Memory
from environments.obstacle_car_graphical.environment import Environment

from util.loss_functions import huber_loss


vis = False
if vis:
    import pygame
    from pygame.locals import *

    pygame.init()
    clock = pygame.time.Clock()
    window = pygame.display.set_mode(env_params.screen_size)
    pygame.display.set_caption("observations")

memory = Priority_Memory()
brain = DQN_Brain(memory, loss=huber_loss)
env = Environment()
agent = PER_Agent(memory, brain, env)

for interaction in tqdm(range(params.TOTAL_INTERACTIONS), smoothing=1):

    # let the agent interact with the environment and memorize the result
    agent.act()

    if vis:
        frame = agent.env.render()
        surf = pygame.surfarray.make_surface(np.transpose(frame, axes=[1, 0, 2]))
        window.blit(surf, (0, 0))

        pygame.display.update()
        clock.tick(10)

    # fill the memory before training
    if interaction < params.REPLAY_START_SIZE:
        continue

    # train the network every N steps
    if interaction % params.TRAIN_SKIPS != 0:
        continue

    brain.train_once()

    # update the target network every N steps
    if interaction % params.TARGET_NETWORK_UPDATE_FREQ != 0:
        continue

    brain.update_target_model()



