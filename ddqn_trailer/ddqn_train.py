import numpy as np
import tensorflow as tf  # if tf is not imported first, it crashes :)
from tqdm import tqdm

print(tf.GRAPH_DEF_VERSION)  # and if I don't use it, autoformatting gets rid of it

import ddqn_trailer.params as params
from ddqn_trailer.agent import Agent
from ddqn_trailer.brain import Brain
from ddqn_trailer.memory import Memory

from util.loss_functions import huber_loss

agent = Agent(vis=True)
memory = Memory()
brain = Brain(loss=huber_loss)

for interaction in tqdm(range(params.TOTAL_INTERACTIONS), smoothing=1):

    # let the agent interact with the environment and memorize the result
    from_state, to_state, action, reward, done = agent.act(brain)

    # transform prediction error into priority and memorize observation
    error = brain.get_error((from_state, to_state, action, reward, done))
    priority = np.power(error + params.ERROR_BIAS, params.ERROR_POW)
    memory.push(from_state, to_state, action, reward, done, priority)

    # fill the memory before training
    if interaction < params.REPLAY_START_SIZE:
        continue

    # train the network every N steps
    if interaction % params.TRAIN_SKIPS != 0:
        continue

    # sample batch with priority as weight, train on it
    training_indices = memory.sample_indices()
    batch = memory[training_indices]
    brain.train_on_batch(batch)

    # calculate new priority and update memory
    error = brain.get_error(batch)
    priority = np.power(error + params.ERROR_BIAS, params.ERROR_POW)
    memory.update_priority(training_indices, priority)

    # update the target network every N steps
    if interaction % params.TARGET_NETWORK_UPDATE_FREQ != 0:
        continue

    brain.update_target()
