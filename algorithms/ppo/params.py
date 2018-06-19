# parameters for the training setup
RUN_TIME = 3600 * 10
NUM_EPISODES = 1000000
AGENTS = 32
OPTIMIZERS = 1
WAITING_TIME = 0.0001

# params for the memory
MEM_SIZE = 10000

# parameters for the discount
NUM_STEPS = 15
GAMMA = 0.99
LAMBDA = 0.9
REWARD_SCALE = 1

# parameters for the neural network
NUM_ACTIONS = 2

# parameters for the training
LEARNING_RATE = 5e-3
DECAY = 0.99
LOSS_VALUE = .5
LOSS_ENTROPY = .01
GRADIENT_NORM_CLIP = 20
RATIO_CLIP_VALUE = 0.15
NUM_UPDATES = 1  # updates before we switch old and new policies

NUM_BATCHES = 10
BATCH_SIZE = 32

L2_REG_CONV = 0  # 1e-3
L2_REG_FULLY = 0  # 1e-3

# parameters to control tensorflow behaviour (and logging)
TF_ALLOW_GROWTH = True
TF_LOG_DEVICE_PLACEMENT = False
