# parameters for the training setup
RUN_TIME = 3600 * 10
NUM_EPISODES = 1000000
AGENTS = 24
OPTIMIZERS = 1
WAITING_TIME = 0.0001

# parameters for the discount
NUM_STEPS = 10  # basically always run till episode end
GAMMA = 0.99
LAMBDA = 0.95
REWARD_SCALE = 1

# parameters for the neural network
NUM_ACTIONS = 4

# parameters for the training
LEARNING_RATE = 3e-4
DECAY = 0.99
LOSS_VALUE = .5
LOSS_ENTROPY = 1e-4
GRADIENT_NORM_CLIP = 20
RATIO_CLIP_VALUE = 0.15
VALUE_CLIP_RANGE = 0.15

L2_REG_CONV = 1e-3  # 1e-3
L2_REG_FULLY = 1e-3  # 1e-3
NUM_BATCHES = 40
BATCH_SIZE = 64
NUM_EPOCHS = 10  # number of times we iterate through the observed data

# params for the memory
MEM_SIZE = NUM_BATCHES * BATCH_SIZE


# parameters to control tensorflow behaviour (and logging)
TF_ALLOW_GROWTH = True
TF_LOG_DEVICE_PLACEMENT = False

# parameters for mpi
# the addresses in our topology
rank_brain = 0
rank_memory = 1
rank_agents =list(range(2,AGENTS + 2))

# the message types
message_prediction = 0
message_observation = 1
message_batch = 2
message_reset = 3