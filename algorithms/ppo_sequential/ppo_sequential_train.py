import numpy as np

np.seterr(all='raise')
np.random.seed(0)

from algorithms.ppo_sequential.agent import Agent
from algorithms.ppo_sequential.brain import Brain
from algorithms.ppo_sequential.fc_models import FullyConnectedModel
from algorithms.ppo_sequential.memory import Memory
import algorithms.ppo_sequential.params as params


memory = Memory()
brain = Brain(memory, FullyConnectedModel)
agent = Agent(brain, memory)

for update in params.NUM_UPDATES:
    # generate training data with the agent
    while len(memory) < params.MEM_SIZE:
        agent.act()

    # pop training data for brain
    training_data = memory.pop()

    # in this training data, we have value predictions, empirical rewards, etc
    # we can log metrics here

    # optimize brain on training data
    brain.optimize(training_data)