import numpy as np
import tensorflow as tf
print(tf.__version__)

np.random.seed(0)

from dqn.agent import Agent
from dqn.memory import Memory
from dqn.brain import Brain
import dqn.params as params

from tqdm import tqdm
from util.loss_functions import huber_loss

agent = Agent()
memory = Memory()
brain = Brain(loss=huber_loss)

for interaction in tqdm(range(params.TOTAL_INTERACTIONS), smoothing=1):

    # let the agent interact with the environment and memorize the result
    from_state, to_state, action, reward, done = agent.act(brain)
    memory.push(from_state, to_state, action, reward, done)

    # fill the memory before training
    if interaction < params.REPLAY_START_SIZE:
        continue

    # train the network every N steps
    if interaction % params.TRAIN_SKIPS != 0:
        continue

    batch = memory.sample()
    brain.train_on_batch(batch)
