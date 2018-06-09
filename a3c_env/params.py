# parameters for the training setup
RUN_TIME = 3600
NUM_EPISODES = 1000000
AGENTS = 32
OPTIMIZERS = 4
WAITING_TIME = 0.0001

# parameters for the agent
INITIAL_EXPLORATION = 0.1
FINAL_EXPLORATION = .05
FINAL_EXPLORATION_ACTION = 50000
EXPLORATION_STEP = (INITIAL_EXPLORATION - FINAL_EXPLORATION) / FINAL_EXPLORATION_ACTION

#params for the memory
MEM_SIZE = 10000

# parameters for the discount
NUM_STEPS = 90
GAMMA = 0.99
LAMBDA = 0.75
REWARD_SCALE = 1

# parameters for the neural network
FRAME_SIZE = (84, 84)
INPUT_SHAPE = (*FRAME_SIZE, 1)
NUM_ACTIONS = 4
MIN_BATCH = 32
MAX_BATCH = 5 * MIN_BATCH
RNN_SIZE = 126

# parameters for the training
LEARNING_RATE = 1e-4
DECAY = 0.99
LOSS_VALUE = .5
LOSS_ENTROPY = .02
GRADIENT_NORM_CLIP = 25.

# parameters to control tensorflow behaviour (and logging)
TF_ALLOW_GROWTH = True
TF_LOG_DEVICE_PLACEMENT = False
