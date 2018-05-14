
# directory management:
# delete all previous memory maps
# and create dirs for checkpoints (if not present)
import os
import shutil
from tempfile import mkstemp


class Memory():

    def __init__(self):

        # creating a new memory, remove existing memory maps
        if os.path.exists(os.getcwd() + "/memory_maps/"):
            shutil.rmtree(os.getcwd() + "/memory_maps/")
        os.mkdir(os.getcwd() + "/memory_maps/")


        # from_state_memory = np.memmap(mkstemp(dir="memory_maps")[0], dtype=np.uint8, mode="w+", shape=(REPLAY_MEMORY_SIZE, *INPUT_SHAPE))
        # to_state_memory = np.memmap(mkstemp(dir="memory_maps")[0], dtype=np.uint8, mode="w+", shape=(REPLAY_MEMORY_SIZE, *INPUT_SHAPE))

        from_state_memory = np.empty(shape=(REPLAY_MEMORY_SIZE, *INPUT_SHAPE), dtype=np.uint8)
        to_state_memory = np.empty(shape=(REPLAY_MEMORY_SIZE, *INPUT_SHAPE), dtype=np.uint8)

        # these other parts of the memory consume only very little memory and can be kept in ram
        action_memory = np.empty(shape=(REPLAY_MEMORY_SIZE), dtype=np.uint8)
        reward_memory = np.empty(shape=(REPLAY_MEMORY_SIZE, 1), dtype=np.int16)
        terminal_memory = np.empty(shape=(REPLAY_MEMORY_SIZE, 1), dtype=np.bool)
        # action_memory = np.memmap(mkstemp(dir="memory_maps")[0], dtype=np.uint8, mode="w+", shape=(REPLAY_MEMORY_SIZE, 1))
        # reward_memory = np.memmap(mkstemp(dir="memory_maps")[0], dtype=np.float32, mode="w+", shape=(REPLAY_MEMORY_SIZE, 1))
        # terminal_memory = np.memmap(mkstemp(dir="memory_maps")[0], dtype=np.bool, mode="w+", shape=(REPLAY_MEMORY_SIZE, 1))
