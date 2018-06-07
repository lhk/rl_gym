import tensorflow as tf  # if tf is not imported first, it crashes :)
from tqdm import tqdm

print(tf.GRAPH_DEF_VERSION)  # and if I don't use it, autoformatting gets rid of it

import ddqn.params as params
from ddqn.agent import Agent
from ddqn.brain import Brain
from ddqn.memory import Memory
from environments.trailer_env.environment import Environment

from util.loss_functions import huber_loss

memory = Memory()
brain = Brain(memory, loss=huber_loss)
env = Environment()
agent = Agent(memory, brain, env)

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
