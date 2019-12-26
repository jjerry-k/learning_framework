import os
# For Mac User...
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import utils, datasets

print("Packge Loaded!")

# Data Loading
(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()
train_x, test_x = np.reshape(train_x/255., [-1, 784]), np.reshape(test_x/255., [-1, 784])
train_y, test_y = utils.to_categorical(train_y, 10), utils.to_categorical(test_y, 10)

print("Train Data's Shape : ", train_x.shape, train_y.shape)
print("Test Data's Shape : ", test_x.shape, test_y.shape)

# Set Network
hidden_node = 256
num_classes = 10

X = tf.placeholder(tf.float32, shape=[None, 784])

W = tf.Variable(tf.random.normal([784, hidden_node]))
b = tf.Variable(tf.zeros([hidden_node]))

h = tf.matmul(X, W)+b

Y = tf.placeholder(tf.float32, shape = [None, 10])
Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=h, labels=Y))
Optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(Loss)
Corr = tf.equal(tf.argmax(output,1), tf.argmax(Y,1)) 
Acc = tf.reduce_mean(tf.cast(Corr, tf.float32)) 

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# Training loop
print("Start Training !")
epochs = 500
batch_size = 100
steps = np.ceil(len(train_x)/batch_size)
for epoch in range(epochs):
    epoch_loss = 0
    epoch_acc = 0
    for step in range(0, len(train_x), batch_size):
        _, step_loss, step_acc = sess.run([Optimizer, Loss, Acc], 
                                          feed_dict={X:train_x[step:step+batch_size], Y:train_y[step:step+batch_size]})
        epoch_loss += step_loss
        epoch_acc += step_acc
    val_idx = np.random.choice(len(test_x), batch_size, replace=False)
    val_loss, val_acc = sess.run([Loss, Acc], feed_dict={X:test_x, Y:test_y})

    print("\nEpoch : ", epoch)
    print("Train Loss : ", epoch_loss/steps, " Train Accuracy : ", epoch_acc/steps)
    print("Validation Loss : ", val_loss, "Validation Accuracy : ", val_acc)