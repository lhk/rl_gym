# parameters for the training setup
RUN_TIME = 3600 * 10
NUM_EPISODES = 1000000
AGENTS = 24
OPTIMIZERS = 1
WAITING_TIME = 0.0001

# parameters for the discount
NUM_STEPS = 15
GAMMA = 0.99
LAMBDA = 0.9
REWARD_SCALE = 1

# parameters for the neural network
NUM_ACTIONS = 2
MIN_BATCH = 256
BATCH_SIZE = 32

# params for the memory
MEM_SIZE = 2*MIN_BATCH

# parameters for the training
LEARNING_RATE = 7e-4
DECAY = 0.99
LOSS_VALUE = .5
LOSS_ENTROPY = .01
GRADIENT_NORM_CLIP = 0.5
NUM_EPOCHS = 2

# parameters to control tensorflow behaviour (and logging)
TF_ALLOW_GROWTH = True
TF_LOG_DEVICE_PLACEMENT = False
