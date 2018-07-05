import numpy as np
np.seterr(all='raise')
np.random.seed(0)

from algorithms.ppo_sequential.agent import Agent
from algorithms.ppo_sequential.brain import Brain
from algorithms.policy_models.conv_models import ConvLSTMModel
from algorithms.policy_models.fc_models import FCRadialCar, FCCartPole
from algorithms.ppo_sequential.memory import Memory
import algorithms.ppo_sequential.params as params

# from environments.obstacle_car.environment import Environment_Graphical as Environment
from environments.obstacle_car.environment_vec import Environment_Vec as Environment
#from environments.openai_gym.environment import Environment

from colorama import Fore, Style
from tqdm import tqdm

Model = FCRadialCar
memory = Memory()
brain = Brain(Model)

vis = True
agent = Agent(brain, memory, Environment, vis=vis, vis_fps=500)
if vis:
    brain.load_weights()
    agent.reset()
    agent.reset_metadata()

while True:
    while len(memory) < params.MEM_SIZE:
        agent.act()
    memory.pop()