# the architecture of optimizers and agents is taken from: https://github.com/jaara/AI-blog/blob/master/CartPole-a3c_doom.py
import numpy as np

np.seterr(all='raise')
np.random.seed(0)

import time, threading, os

from algorithms.ppo_threading.agent import Agent
from algorithms.ppo_threading.brain import Brain
from algorithms.ppo_threading.fc_models import FullyConnectedModel
from algorithms.ppo_threading.memory import Memory
import algorithms.ppo_threading.params as params


class Optimizer(threading.Thread):

    def __init__(self, brain: Brain):
        threading.Thread.__init__(self)
        self.brain = brain
        self.stop = False

    def run(self):
        while not self.stop:
            brain.optimize()

from multiprocessing import Event

collect_data = Event()
collect_data.set()

memory = Memory(collect_data)
brain = Brain(memory, FullyConnectedModel, collect_data)

agents = [Agent(brain, memory, collect_data) for i in range(params.AGENTS)]
#agents.append(Agent(brain, memory, collect_data, reset_queue,  vis=True))  # one agent for the visualization

opts = [Optimizer(brain) for i in range(params.OPTIMIZERS)]

for o in opts:
    o.start()
for a in agents:
    a.start()

time.sleep(params.RUN_TIME)

for a in agents:
    a.stop = True
for a in agents:
    a.join()

for o in opts:
    o.stop = True
for o in opts:
    o.join()

brain.model.save(os.getcwd() + "/a3c_env/weights.hdf5")
