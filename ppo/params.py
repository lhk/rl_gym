# parameters for the training setup
RUN_TIME = 3600 * 10
NUM_EPISODES = 1000000
AGENTS = 1
OPTIMIZERS = 1
WAITING_TIME = 0.0001

# parameters for the agent
INITIAL_EXPLORATION = 0
FINAL_EXPLORATION = 0
FINAL_EXPLORATION_ACTION = 10000
EXPLORATION_STEP = (INITIAL_EXPLORATION - FINAL_EXPLORATION) / FINAL_EXPLORATION_ACTION

# params for the memory
MEM_SIZE = 10000

# parameters for the discount
NUM_STEPS = 8
GAMMA = 0.99
LAMBDA = 0.99
REWARD_SCALE = 1

# parameters for the neural network
NUM_ACTIONS = 2

# parameters for the training
LEARNING_RATE = 5e-3
DECAY = 0.99
LOSS_VALUE = .5
LOSS_ENTROPY = .01
GRADIENT_NORM_CLIP = 100
RATIO_CLIP_VALUE = 0.2
NUM_UPDATES = 1  # updates before we switch old and new policies

NUM_BATCHES = 20
BATCH_SIZE = 32

L2_REG_CONV = 0# 1e-3
L2_REG_FULLY = 0# 1e-3

# parameters to control tensorflow behaviour (and logging)
TF_ALLOW_GROWTH = True
TF_LOG_DEVICE_PLACEMENT = False
