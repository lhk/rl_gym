import tensorflow as tf  # if tf is not imported first, it crashes :)
from tqdm import tqdm

print(tf.GRAPH_DEF_VERSION)  # and if I don't use it, autoformatting gets rid of it

import dqn.params as params
from dqn.agent import ER_Agent
from dqn.brain import Dueling_Brain
from dqn.memory import Equal_Memory
from environments.obstacle_car_graphical.environment import Environment

from util.loss_functions import huber_loss

memory = Equal_Memory()
brain = Dueling_Brain(memory, loss=huber_loss)
env = Environment()
agent = ER_Agent(memory, brain, env)

for interaction in tqdm(range(params.TOTAL_INTERACTIONS), smoothing=1):

    # let the agent interact with the environment and memorize the result
    agent.act()

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
