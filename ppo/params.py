# parameters for the training setup
RUN_TIME = 3600 * 10
NUM_EPISODES = 1000000
AGENTS = 32
OPTIMIZERS = 1
WAITING_TIME = 0.0001

# parameters for the agent
INITIAL_EXPLORATION = 1.0
FINAL_EXPLORATION = .05
FINAL_EXPLORATION_ACTION = 10000
EXPLORATION_STEP = (INITIAL_EXPLORATION - FINAL_EXPLORATION) / FINAL_EXPLORATION_ACTION

# params for the memory
MEM_SIZE = 10000

# parameters for the discount
NUM_STEPS = 15
GAMMA = 0.99
LAMBDA = 0.8
REWARD_SCALE = 1

# parameters for the neural network
NUM_ACTIONS = 4
MIN_BATCH = 2
MAX_BATCH = 5 * MIN_BATCH

# parameters for the training
LEARNING_RATE = 7e-4
DECAY = 0.99
LOSS_VALUE = .5
LOSS_ENTROPY = .05
GRADIENT_NORM_CLIP = 10
RATIO_CLIP_VALUE = 0.1
NUM_UPDATES = 1000  # updates before we switch old and new policies

L2_REG_CONV = 1e-3
L2_REG_FULLY = 1e-3

# parameters to control tensorflow behaviour (and logging)
TF_ALLOW_GROWTH = True
TF_LOG_DEVICE_PLACEMENT = False
