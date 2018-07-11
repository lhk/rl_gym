# Reinforcement learning

This repository collects implementations for common reinforcement learning algorithms.
So far I've implemented the following algorithms:
 - DQN with many varieties: DDQN, Dueling-Q-Learning, prioritized experience replay
 - A3C (threading based) 
 - PPO (sequential/threading and gpu/cpu) 
 
Currently I'm working on refactoring, there is a lot of duplicated code.
Initially, I wanted every algorithm to be self-contained: 
For example, the ddqn implementation should be one block of code, without external dependencies.
But most pieces of the code are very similar. And it would be nice to have a more modular setup,
to try dqn with a different targets (TD-lambda etc).
So far, I have merged all the different dqn versions into one package called dqn.
But I'm undecided if it will stay like this.

A small update on the refactoring: I decided to keep the following classes
 - Agent: Responsible for interacting with the environment, keeping track of rewards (TD-lambda) and feeding the memory
 - Memory: Stores the training samples. Can be a simple buffer (A3C) or more complex (priority based sampling for DQN)
 - Model: The function approximator for policy, values, Q-values, whatever you are using
 - Brain: Wrapper around the model, responsible for training and updates

The refactoring is mostly done now. A3C, PPO and DQN all follow the same design principles.
(Also note: Before the refactoring, I've trained the various algorithms against doom and atari. I'll do that again to check if something has been broken. TODO: remove this text after the tests have been successful :) )

Next to the refactoring, I've started to look at homebrew environments for reinforcement learning.
Such as a car which should navigate to an obstacle. The motivation is to learn about complexity of tasks.
For example: Many atari games give pretty immediate rewards (paddle missed the ball : -1).
In this car environment the first reward would come after quite a few timesteps. How hard is that actually ?
For the homebrewing, I've added a very small interface which wraps:
 - vizdoom
 - openai atari
 - homebrew car simulation

The following is a policy learned by PPO on the car simulation. The white dot is the car, it must navigate towards the green dot, while avoiding the red dots. This environment returns a list of positions in polar coordinates.
The repository contains lots of helper methods, for example to render the polar coordinate representation to a numpy array and export it as gif:
![homebrew](https://media.giphy.com/media/2vrzoHJa8C9qDhOWzY/giphy.gif)
