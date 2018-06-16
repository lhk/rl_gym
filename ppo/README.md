#PPO
After reading the enthusiastic openAI blogpost about ppo, I decided to implement this algorithm next.

As far as I can see, PPO is just a smart reformulation of the loss. They still work in combination with TD-lambda for the
value estimator and can make use of something like GAE for the advantage estimation.

So I'm reusing most of my A3C code, including the parallel architecture.

This algorithm is very much work in progress.