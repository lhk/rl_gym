# the architecture of optimizers and agents is taken from: https://github.com/jaara/AI-blog/blob/master/CartPole-a3c_trailer.py
import numpy as np

np.seterr(all='raise')
np.random.seed(0)

import time, threading

from a3c_trailer.agent import Agent
from a3c_trailer.brain import Brain
from a3c_trailer.memory import Memory
import a3c_trailer.params as params


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
