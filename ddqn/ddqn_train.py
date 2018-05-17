import numpy as np
import tensorflow as tf  # if tf is not imported first, it crashes :)
from tqdm import tqdm

print(tf.GRAPH_DEF_VERSION)  # and if I don't use it, autoformatting gets rid of it

import ddqn.params as params
from ddqn.agent import Agent
from ddqn.brain import Brain
from ddqn.memory import Memory

agent = Agent()
memory = Memory()
brain = Brain()

for interaction in tqdm(range(params.TOTAL_INTERACTIONS), smoothing=1):

    # use the brain to determine the best action for this state
    q_values = brain.predict_q(agent.state)
    best_action = q_values.argmax(axis=1)

    # let the agent interact with the environment and memorize the result
    from_state, to_state, action, reward, done = agent.act(best_action)

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

    batch = memory.sample()
    brain.train_on_batch(batch)

    # update the target network every N steps
    if interaction % params.TARGET_NETWORK_UPDATE_FREQ != 0:
        continue

    brain.update_target()

    if interaction % 100000 == 0:
        brain.model.save_weights("checkpoints/weights" + str(interaction) + ".hdf5")
