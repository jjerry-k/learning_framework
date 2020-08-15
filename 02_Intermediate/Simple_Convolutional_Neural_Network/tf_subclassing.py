import os
# For Mac User...
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np

import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses, utils, datasets

tf.random.set_seed(777)

print("Packge Loaded!")
# %%
EPOCHS = 500
BATCH_SIZE = 128

# Data Loading
(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()
train_x, test_x = np.expand_dims(train_x/255., -1), np.expand_dims(test_x/255., -1)


print("Train Data's Shape : ", train_x.shape, train_y.shape)
print("Test Data's Shape : ", test_x.shape, test_y.shape)

train_ds = tf.data.Dataset.from_tensor_slices(
    (train_x, train_y)).shuffle(10000).batch(BATCH_SIZE)

test_ds = tf.data.Dataset.from_tensor_slices(
    (test_x, test_y)).shuffle(10000).batch(BATCH_SIZE)

# plt.plot(x, y, 'r.')
# plt.show()
print("Data Prepared!")

# %%
class SimpleConvolutionalNeuralNetwork(models.Model):
    def __init__(self):
        super(SimpleConvolutionalNeuralNetwork, self).__init__()
        self.conv1 = layers.Conv2D(16, 3, activation='relu', input_shape=(28, 28, 1,))
        self.pool1 = layers.MaxPool2D()
        self.conv2 = layers.Conv2D(32, 3, activation='relu')
        self.pool2 = layers.MaxPool2D()
        self.flat = layers.Flatten()
        self.dense = layers.Dense(10, activation='softmax')
    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flat(x)
        return self.dense(x)

# Create an instance of the model
model = SimpleConvolutionalNeuralNetwork()

loss_object = losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = optimizers.Adam()

# %%
for epoch in range(EPOCHS):
    epoch_loss = 0
    for batch_x, batch_y in train_ds:
        with tf.GradientTape() as tape:
            predictions = model(batch_x, training=True)
            loss = loss_object(batch_y, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss += loss
    print("{:5}|{:10.6f}".format(epoch+1, loss/(len(train_x)/BATCH_SIZE + 1)))