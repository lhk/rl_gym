# the architecture of optimizers and agents is taken from: https://github.com/jaara/AI-blog/blob/master/CartPole-a3c_doom.py
import numpy as np

np.seterr(all='raise')
np.random.seed(0)

import time, threading, os

from algorithms.ppo_mpi.agent import Agent
from algorithms.ppo_mpi.brain import Brain
from algorithms.ppo_mpi.fc_models import FullyConnectedModel
from algorithms.ppo_mpi.memory import Memory
import algorithms.ppo_mpi.params as params

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

Model = FullyConnectedModel

if rank == params.rank_brain:
    # set up the brain
    brain = Brain(Model)

    # listen to events
    while True:
        # first we check if we have received training data from the memory, this has highest priority
        if comm.iprobe(source=params.rank_memory, tag=params.message_batch):

            # we reset all agents
            # this prevents them from pushing observations to the memory
            for rank_agent in params.rank_agents:
                comm.isend(dest = rank_agent, tag = params.message_reset)

            # update the network
            training_batch = comm.recv(source=params.rank_memory, tag=params.message_batch)
            brain.optimize(training_batch)



        # the agents can request predictions
        for rank_agent in params.rank_agents:
            if comm.iprobe(source=rank_agent, tag = params.message_prediction):
                observation_state = comm.recv(source=rank_agent, tag=params.message_prediction)
                observation, state = observation_state
                prediction = brain.predict(observation, state)
                comm.send(prediction, dest=rank_agent)

if rank == params.rank_memory:
    # set up the memory
    memory = Memory(Model)

    # listen to events
    while True:
        # if enough training data is in the memory, we send a training batch to the brain
        if len(memory) >= params.NUM_BATCHES * params.BATCH_SIZE:
            training_batch = memory.pop()
            comm.send(dest = params.rank_memory, tag=params.message_batch)

        # collect data from every agent
        for rank_agent in params.rank_agents:
            if comm.iprobe(source=rank_agent, tag=params.message_observation):
                batch = comm.recv(source=rank_agent, tag=params.message_observation)
                memory.push(batch)

if rank in params.rank_agents:
    # set up an agent
    agent = Agent(comm, rank)
    while True:
        agent.run_one_episode()
