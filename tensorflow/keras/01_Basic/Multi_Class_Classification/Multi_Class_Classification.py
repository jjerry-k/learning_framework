import os
# For Mac User...
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses, utils, datasets

print("Packge Loaded!")


# Data Loading
(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()
train_x, test_x = np.reshape(train_x/255., [-1, 784]), np.reshape(test_x/255., [-1, 784])

print("Train Data's Shape : ", train_x.shape, train_y.shape)
print("Test Data's Shape : ", test_x.shape, test_y.shape)

# Network Building
## Using Sequential
mlp = models.Sequential()
mlp.add(layers.Dense(10, activation='softmax'))

## Using Functional
# _input = layers.Input(shape=(784, ))
# layer = layers.Dense(10, activation='softmax')(_input)
# mlp = models.Model(inputs=_input, outputs=layer)

print("Network Built!")

mlp.compile(optimizer=optimizers.Adam(), loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

history = mlp.fit(train_x, train_y, epochs=10, batch_size=16, validation_data=(test_x, test_y))

plt.plot(history.history['loss'], '.-')
plt.plot(history.history['val_loss'], '.-')
plt.legend(['train_loss', 'val_loss'], loc=0)
plt.show()

