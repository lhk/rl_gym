import numpy as np
import tensorflow as tf

from dqn.agent import Agent
from dqn.memory import Memory
from dqn.brain import Brain
import dqn.params as params

from tqdm import tqdm

agent = Agent()
memory = Memory()
brain = Brain()

for interaction in tqdm(range(params.TOTAL_INTERACTIONS), smoothing=1):

    # use the brain to determine the best action for this state
    q_values = brain.predict_q(agent.state)
    best_action = q_values.argmax(axis=1)

    # let the agent interact with the environment and memorize the result
    from_state, to_state, action, reward, done = agent.act(best_action)
    memory.push(from_state, to_state, action, reward, done)

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