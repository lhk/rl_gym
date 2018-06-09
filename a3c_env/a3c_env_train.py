# the architecture of optimizers and agents is taken from: https://github.com/jaara/AI-blog/blob/master/CartPole-a3c_doom.py
import numpy as np

np.seterr(all='raise')
np.random.seed(0)

import time, threading, os

from a3c_env.agent import Agent
from a3c_env.brain import Brain
from a3c_env.memory import Memory
import a3c_env.params as params


class Optimizer(threading.Thread):

    def __init__(self, brain: Brain):
        threading.Thread.__init__(self)
        self.brain = brain
        self.stop = False

    def run(self):
        while not self.stop:
            brain.optimize()


memory = Memory()
brain = Brain(memory)

agents = [Agent(brain, memory) for i in range(params.AGENTS)]
#agents.append(Agent(brain, memory, vis=True)) # one agent for the visualization
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

brain.model.save(os.getcwd()+"/a3c_env/weights.hdf5")
