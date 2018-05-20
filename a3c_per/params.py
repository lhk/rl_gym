# parameters for the training setup
ENV_NAME = 'CartPole-v0'
RUN_TIME = 240**3
AGENTS = 8
OPTIMIZERS = 2
WAIT_ON_ACTION = 0.001

# parameters for the agent
INITIAL_EXPLORATION = 0.4
FINAL_EXPLORATION = .15
FINAL_EXPLORATION_ACTION = 75000
EXPLORATION_STEP = (INITIAL_EXPLORATION - FINAL_EXPLORATION) / FINAL_EXPLORATION_ACTION

# parameters for the discount
GAMMA = 0.99
NUM_STEPS = 8
GAMMA_N = GAMMA ** NUM_STEPS

# parameters for the neural network
NUM_STATE = 4
INPUT_SHAPE = (NUM_STATE,)
NUM_ACTIONS = 2
BATCH_SIZE = 32

# parameters for the training
LEARNING_RATE = 5e-4
DECAY = 0.99
LOSS_VALUE = .5
LOSS_ENTROPY = .01

# parameters to control tensorflow behaviour (and logging)
TF_ALLOW_GROWTH = True
TF_LOG_DEVICE_PLACEMENT = False

# parameters for the memory
REPLAY_MEMORY_SIZE = int(2 ** 16)
REPLAY_START_SIZE = int(2**14)
MEMORY_MAPPED = False
ERROR_BIAS = 0.01
ERROR_POW = 0.5