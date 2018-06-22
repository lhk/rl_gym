This is the interesting part of the repository, the rl algorithms :)

I have two families of algorithms: q learning and policy gradient methods.
The deep q learning approaches are all contained under dqn:
 - dqn
 - ddqn / dueling q learning
 - prioritized experience replay

For policy gradients, so far I have implemented a3c and ppo.
Here the code is much more fragmented. For ppo, there are three different codebases:
 - ppo_sequential: this one should be the most reliable and polished
 - ppo_threading: this one uses multiple parallel agent to get decorrelated training data.
   the idea is the same as for a3c. Please note: this only uses threading, not multiprocessing. So you get no speedup,
   just the decorrelated training data. And just collecting lots of observations + shuffling them seems to work just
   as well. So I think the easier structure of ppo_sequential is preferable.
 - ppo_msi: this one actually is parallel, based on mpi. While mpi4py is a fantastic interface on top of mpi, I haven't
   found a good way to debug this code, yet. And so far I don't see any speedups over ppo_sequential. Probably because
   I'm not using the buffer support of mpi. Every message has to be pickled and unpickled. That seems like a huuuuge
   overhead.
 - a3c_threading: a3c kind of breaks the naming scheme, a3c = asynchronous advantage actor-critic. So to add threading
   is a bit a duplicate of asynchronous.

The code for those policy gradient methods is extremely similar. Only small parts of the optimization step inside the
brain are different.
Maybe I'll refactor them to one shared codebase, as for dqn.

While the python threading will give no speedups on its own, I still think that the threading architecture could offer
a potentially huge optimization: collecting the observations before passing them to the brain would allow me to use
batches while collecting observations.
Ideally that would be paired with multiprocessing (or mpi4py) for full use both CPU and GPU. But for that I'll have to
find a better debugging setup.