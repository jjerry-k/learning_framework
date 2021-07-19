# %%
import sys
sys.path.append('../../')
import time
import numpy as np
import numpy.random as npr
import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad
from jax import nn, lax
from jax import random

from jax.experimental import stax, optimizers

from utils import jax_dataset
key = random.PRNGKey(1)
# %%
param_scale = 0.1
step_size = 0.001
num_epochs = 10
batch_size = 128

train_images, train_labels, test_images, test_labels = jax_dataset.mnist()
train_images = np.reshape(train_images, [-1, 1, 28, 28])
test_images = np.reshape(test_images, [-1, 1, 28, 28])
num_train = train_images.shape[0]
num_complete_batches, leftover = divmod(num_train, batch_size)
num_batches = num_complete_batches + bool(leftover)

# %%
def data_stream():
    rng = npr.RandomState(0)
    while True:
        perm = rng.permutation(num_train)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size:(i + 1) * batch_size]
            yield train_images[batch_idx], train_labels[batch_idx]

batches = data_stream()

init_fun, net = stax.serial(
    stax.Conv(16, (3, 3), (1, 1), padding="SAME"), 
    stax.Relu, 
    stax.MaxPool((2, 2), (2, 2), padding="SAME"),
    stax.Conv(32, (3, 3), (1, 1), padding="SAME"), 
    stax.Relu, 
    stax.MaxPool((2, 2), (2, 2), padding="SAME"),
    stax.Flatten,
    stax.Dense(10),
    stax.LogSoftmax
)

_, params = init_fun(key, (64, 1, 28, 28))

def loss(params, batch):
	inputs, targets = batch
	preds = net(params, inputs)
	return -jnp.mean(jnp.sum(preds * targets, axis=1))

def accuracy(params, batch):
	inputs, targets = batch
	target_class = jnp.argmax(targets, axis=1)
	predicted_class = jnp.argmax(predict(params, inputs), axis=1)
	return jnp.mean(predicted_class == target_class)

# %%
opt_init, opt_update, get_params = optimizers.adam(step_size)
opt_state = opt_init(params)

@jit
def update(params, batch, opt_state):
	value, grads = value_and_grad(loss)(params, batch)
	opt_state = opt_update(0, grads, opt_state)
	return get_params(opt_state), opt_state, value

# %%
for epoch in range(num_epochs):
	start_time = time.time()
	loss_val = 0
	for _ in range(num_batches):
		batch_data = next(batches)
		loss_val += loss(params, batch_data)
		params = update(params, batch_data, opt_state)
	epoch_time = time.time() - start_time

	train_acc = accuracy(params, (train_images, train_labels))
	test_acc = accuracy(params, (test_images, test_labels))
	print(f"Epoch: {epoch+1}, Loss: {loss_val/num_batches}, Elapsed time: {epoch_time:0.2f} sec")
	print("Training set accuracy {}".format(train_acc))
	print("Test set accuracy {}".format(test_acc))