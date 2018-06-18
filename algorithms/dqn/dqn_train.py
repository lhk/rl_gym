import numpy as np
import tensorflow as tf  # if tf is not imported first, it crashes :)
from tqdm import tqdm

print(tf.GRAPH_DEF_VERSION)  # and if I don't use it, autoformatting gets rid of it

import algorithms.dqn.params as params
import environments.obstacle_car.params as env_params
from algorithms.dqn.agent import Agent
from algorithms.dqn.memory import Priority_Memory
from environments.obstacle_car.environment_graphical import Environment_Graphical as Environment
from algorithms.dqn.brain import Brain
from algorithms.dqn.models import DQN_Model

from util.loss_functions import huber_loss

vis = False
if vis:
    import pygame

    pygame.init()
    clock = pygame.time.Clock()
    window = pygame.display.set_mode(env_params.screen_size)
    pygame.display.set_caption("observations")

memory = Priority_Memory(DQN_Model)
brain = Brain(DQN_Model, memory, loss=huber_loss, load_path=None)
env = Environment()
agent = Agent(memory, brain, env)

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
