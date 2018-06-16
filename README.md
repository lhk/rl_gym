# Reinforcement learning

This repository collects implementations for common reinforcement learning algorithms.
So far I've implemented the following algorithms:
 - DQN with many varieties: DDQN, Dueling-Q-Learning, prioritized experience replay
 - A3C
 
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

The structure can be explained with PPO as an example: I'm working on implementing PPO, with a parallel architecture like A3C.
Each agent has it's own environment and they run episodes in parallel and push their observations to a buffer.
As opposed to the usual A3C where each agent updates its own network on CPU, here the network is allocated on the gpu.
It is represented by a single brain.
The brain is trained by optimizer threads. All they do is invoke the optimize() method on the brain again and again.
In this method the brain pops training data from the observation buffer and updates its model.

This sounds like an implementation of A3C. But it uses the clipped ratio surrogate loss of PPO.
My A3C code is extremely similar, but not yet refactored to move the Model out of the brain.

Eventually, I'll also refactor DQN to expose a model.

(Also note: Before the refactoring, I've trained the various algorithms against doom and atari. I'll do that again to check if something has been broken. TODO: remove this text after the tests have been successful :) )

Next to the refactoring, I've started to look at homebrew environments for reinforcement learning.
Such as a car which should navigate to an obstacle. The motivation is to learn about complexity of tasks.
For example: Many atari games give pretty immediate rewards (paddle missed the ball : -1).
In this car environment the first reward would come after quite a few timesteps. How hard is that actually ?
For the homebrewing, I've added a very small interface which wraps:
 - vizdoom
 - openai atari
 - homebrew car simulation

