import os
# For Mac User...
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np

import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses, utils, datasets

tf.random.set_seed(777)

print("Packge Loaded!")

EPOCHS = 10
BATCH_SIZE = 100
LEARNING_RATE = 0.01

# Data Loading
(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()
train_x, test_x = np.reshape(train_x/255., [-1, 784]), np.reshape(test_x/255., [-1, 784])
# 0 : digit < 5
# 1 : digit >= 5
train_y, test_y = np.greater_equal(train_y, 5)[..., np.newaxis], np.greater_equal(test_y, 5)[..., np.newaxis]


print("Train Data's Shape : ", train_x.shape, train_y.shape)
print("Test Data's Shape : ", test_x.shape, test_y.shape)


# Network Building
## Using Sequential
mlp = models.Sequential()
mlp.add(layers.Dense(1, activation='sigmoid'))

## Using Functional
# _input = layers.Input(shape=(784, ))
# layer = layers.Dense(10, activation='sigmoid')(_input)
# mlp = models.Model(inputs=_input, outputs=layer)

print("Network Built!")

mlp.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), loss=losses.binary_crossentropy, metrics=['accuracy'])

history = mlp.fit(train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(test_x, test_y))
