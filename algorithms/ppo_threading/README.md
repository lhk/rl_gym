#PPO
This was my first ppo implementation, based on the A3C infrastructure I already had.
But synchronization between brain and agent threads is complicated.
And without true multiprocessing, the GIL means that this threading doesn't bring any speedups in any case.

It seems as if collecting enough training data and shuffling it, is sufficient to decorrelate training samples.
So for now, I'm abandonding this and keeping it only as a reference.