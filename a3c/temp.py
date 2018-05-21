from keras.layers import *
from keras.models import *
from keras.optimizers import *

in_layer = Input((10,))

out_layer_1 = Dense(1)(in_layer)
out_layer_2 = Dense(1)(in_layer)

model = Model(inputs=in_layer, outputs=[out_layer_1, out_layer_2])
model.compile(Adam(0.1), loss=["mse", "mse"])

model.train_on_batch(np.random.rand(1, 10), [np.random.rand(1, 1), np.random.rand(1, 1)])

print("here")
