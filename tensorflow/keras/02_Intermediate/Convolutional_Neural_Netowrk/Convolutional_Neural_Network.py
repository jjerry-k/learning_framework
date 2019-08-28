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
train_x, test_x = np.expand_dims(train_x/255., -1), np.expand_dims(test_x/255., -1)

print("Train Data's Shape : ", train_x.shape, train_y.shape)
print("Test Data's Shape : ", test_x.shape, test_y.shape)

# Network Building
## Using Sequential
cnn = models.Sequential()
cnn.add(layers.Conv2D(16, 3, activation='relu', input_shape=(28, 28, 1,)))
cnn.add(layers.MaxPool2D())
cnn.add(layers.Conv2D(32, 3, activation='relu'))
cnn.add(layers.MaxPool2D())
cnn.add(layers.Flatten())
cnn.add(layers.Dense(10, activation='softmax'))

## Using Functional
# _input = layers.Input(shape=(28, 28, 1, ))
# layer = layers.Conv2D(16, 3, activation='relu')(_input)
# layer = layers.MaxPool2D()(layer)
# layer = layers.Conv2D(32, 3, activation='relu')(layer)
# layer = layers.MaxPool2D()(layer)
# layer = layers.Flatten()(layer)
# layer = layers.Dense(10, activation='softmax')(layer)
# cnn = models.Model(inputs=_input, outputs=layer)

print("Network Built!")

# Compiling
cnn.compile(optimizer=optimizers.Adam(), loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])


# Training
history = cnn.fit(train_x, train_y, epochs=10, batch_size=16, validation_data=(test_x, test_y))


# Plotting Result
plt.plot(history.history['loss'], '.-')
plt.plot(history.history['val_loss'], '.-')
plt.legend(['train_loss', 'val_loss'], loc=0)
plt.show()

