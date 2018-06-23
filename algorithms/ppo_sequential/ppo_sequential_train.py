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
#brain.load_weights()
agent = Agent(brain, memory, Environment, vis=False)

#agent.reset()
#agent.reset_metadata()
for update in range(params.NUM_UPDATES):
    agent.reset()
    agent.reset_metadata()
    # generate training data with the agent
    pbar = tqdm(total=params.MEM_SIZE, desc="collecting observations")
    while len(memory) < params.MEM_SIZE:
        agent.act()
        pbar.update()
    pbar.close()

    # pop training data for brain
    training_data = memory.pop()

    #continue

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
