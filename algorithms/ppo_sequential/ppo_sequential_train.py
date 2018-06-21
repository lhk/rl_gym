import numpy as np

np.seterr(all='raise')
np.random.seed(0)

from algorithms.ppo_sequential.agent import Agent
from algorithms.ppo_sequential.brain import Brain
from algorithms.ppo_sequential.fc_models import FullyConnectedModel
from algorithms.ppo_sequential.memory import Memory
import algorithms.ppo_sequential.params as params
from colorama import Fore, Style
from tqdm import tqdm

memory = Memory()
brain = Brain(FullyConnectedModel)
agent = Agent(brain, memory)

for update in range(params.NUM_UPDATES):
    agent.reset()
    agent.reset_metadata()
    # generate training data with the agent
    pbar = tqdm(total=params.MEM_SIZE, dynamic_ncols=True)
    while len(memory) < params.MEM_SIZE:
        agent.act()
        pbar.update()
    pbar.close()

    # pop training data for brain
    training_data = memory.pop()

    # in this training data, we have value predictions, empirical rewards, etc
    # we can log metrics here
    (from_observations, from_states, to_observations, to_states, pred_policies, pred_values, actions, rewards,
     advantages,
     terminals, lengths) = training_data

    pred_policies = np.array(pred_policies).reshape((-1, params.NUM_ACTIONS))
    pred_values = np.array(pred_values).reshape((-1, 1))
    actions = np.vstack(actions).reshape((-1, params.NUM_ACTIONS))
    rewards = np.vstack(rewards)
    advantages = np.vstack(advantages).reshape((-1, 1))

    print(Fore.BLUE)
    print("average predicted value is {}".format(pred_values.mean()))
    print("average empirical reward is {}".format(rewards.mean()))
    print("average advantage is {}".format(advantages.mean()))
    print("advantage std is {}".format(advantages.std()))
    print(Style.RESET_ALL)

    # we also get information from the data collected by the agent
    print(Fore.GREEN)
    print("number of episodes is {}".format(agent.num_episodes))
    print("average reward over episode is {}".format(np.mean(agent.episode_rewards)))
    print(Style.RESET_ALL)

    # optimize brain on training data
    brain.optimize(training_data)
