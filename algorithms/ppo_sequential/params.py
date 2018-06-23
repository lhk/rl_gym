# parameters for the discount
NUM_STEPS = 10  # basically always run till episode end
GAMMA = 0.99
LAMBDA = 0.95
REWARD_SCALE = 1

# parameters for the neural network
NUM_ACTIONS = 4

# parameters for the training
LEARNING_RATE = 1e-4
EPSILON = 1e-5
LOSS_VALUE = .5
LOSS_ENTROPY = 0 #1e-6
GRADIENT_NORM_CLIP = 0.5
RATIO_CLIP_VALUE = 0.15
VALUE_CLIP_RANGE = 0.15

NUM_BATCHES = 40
BATCH_SIZE = 64
NUM_EPOCHS = 10  # number of times we iterate through the observed data

# parameters for the setup
NUM_UPDATES = 1000

# params for the memory
MEM_SIZE = NUM_BATCHES * BATCH_SIZE

# parameters to control tensorflow behaviour (and logging)
TF_ALLOW_GROWTH = True
TF_LOG_DEVICE_PLACEMENT = False
