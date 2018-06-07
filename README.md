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

Next to the refactoring, I've started to look at homebrew environments for reinforcement learning.
Such as a car which should navigate to an obstacle. The motivation is to learn about complexity of tasks.
For example: Many atari games give pretty immediate rewards (paddle missed the ball : -1).
In this car environment the first reward would come after quite a few timesteps. How hard is that actually ?
For the homebrewing, I've added a very small interface which wraps:
 - vizdoom
 - openai atari
 - homebrew car simulation
 
After the refactoring, I would like to move to A3C with priority experience replay. Intuitively the pieces should all be here.
I have agents that push to a memory and a brain that samples from the memory.
Is it enough to just plug in the memory with replay (and priority-based sampling) ?
The "sample efficient" A3C paper is next on the roadmap.