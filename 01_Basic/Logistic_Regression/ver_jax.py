import sys
sys.path.append('../../')
import time
import numpy as np
from jax import jit, grad
from jax.scipy.special import logsumexp
import jax.numpy as jnp
import jax.nn as nn
from matplotlib import pyplot as plt
from utils import jax_dataset

EPOCHS = 10
BATCH_SIZE = 100
LEARNING_RATE = 0.01

def init_random_params(scale, layer_sizes, rng=np.random.RandomState(0)):
    return [(scale * rng.randn(m, n), scale * rng.randn(n))
            for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

def predict(params, inputs):
    outputs = jnp.dot(inputs, params[0][0]) + params[0][1]
    return nn.sigmoid(outputs)

def loss(params, batch):
    inputs, targets = batch
    preds = predict(params, inputs)
    return -jnp.mean(jnp.log(preds) * targets)
    
def accuracy(params, batch):
    inputs, targets = batch
    predicted_class = jnp.greater_equal(predict(params, inputs), 0.5)
    return jnp.mean(predicted_class == targets)

layer_sizes = [784, 1]
param_scale= 1


train_images, train_labels, test_images, test_labels = jax_dataset.mnist()
train_labels= np.greater_equal(np.argmax(train_labels, axis=1)[..., np.newaxis], 0.5).astype(np.float32)
test_labels = np.greater_equal(np.argmax(test_labels, axis=1)[..., np.newaxis], 0.5).astype(np.float32)

num_train = train_images.shape[0]
num_complete_batches, leftover = divmod(num_train, BATCH_SIZE)
num_batches = num_complete_batches + bool(leftover)
# %%
def data_stream():
    rng = np.random.RandomState(0)
    while True:
        perm = rng.permutation(num_train)
        for i in range(num_batches):
            batch_idx = perm[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            yield train_images[batch_idx], train_labels[batch_idx]
batches = data_stream()

@jit
def update(params, batch):
    grads = grad(loss)(params, batch)
    return [(w - LEARNING_RATE * dw, b - LEARNING_RATE * db)
            for (w, b), (dw, db) in zip(params, grads)]

params = init_random_params(param_scale, layer_sizes)
for epoch in range(EPOCHS):
    start_time = time.time()
    loss_val = 0
    for _ in range(num_batches):
        batch_data = next(batches)
        loss_val += loss(params, batch_data)
        params = update(params, batch_data)
    epoch_time = time.time() - start_time

    train_acc = accuracy(params, (train_images, train_labels))
    test_acc = accuracy(params, (test_images, test_labels))
    print(f"Epoch: {epoch+1}, Loss: {loss_val/num_batches}, Elapsed time: {epoch_time:0.2f} sec")
    print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))