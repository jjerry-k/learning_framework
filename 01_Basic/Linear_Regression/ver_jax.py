# %%
import time
import numpy as np
from jax import jit, grad
from jax.scipy.special import logsumexp
import jax.numpy as jnp
from matplotlib import pyplot as plt
# %%
def init_random_params(scale, layer_sizes, rng=np.random.RandomState(0)):
    return [(scale * rng.randn(m, n), scale * rng.randn(n))
            for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

def predict(params, inputs):
    outputs = jnp.dot(inputs, params[0][0]) + params[0][1]
    return outputs

def loss(params, batch):
    inputs, targets = batch
    preds = predict(params, inputs)
    return jnp.mean(jnp.sum((preds - targets)**2, axis=1))
    # return -jnp.mean(jnp.sum(preds - targets, axis=1))
    
def accuracy(params, batch):
    inputs, targets = batch
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(predict(params, inputs), axis=1)
    return jnp.mean(predicted_class == target_class)
# %%
layer_sizes = [1, 1]
param_scale= 1
step_size= 0.01
num_epochs= 10
batch_size= 256
# %%
x = np.random.normal(0.0, 0.55, (10000, 1))
y = x * 0.1 + 0.3 + np.random.normal(0.0, 0.03, (10000,1))
plt.plot(x, y, 'r.')
plt.show()
# %%
num_train = x.shape[0]
num_complete_batches, leftover = divmod(num_train, batch_size)
num_batches = num_complete_batches + bool(leftover)
# %%
def data_stream():
    rng = np.random.RandomState(0)
    while True:
        perm = rng.permutation(num_train)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size:(i + 1) * batch_size]
            yield x[batch_idx], y[batch_idx]
batches = data_stream()
@jit
def update(params, batch):
    grads = grad(loss)(params, batch)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]
# %% 
params = init_random_params(param_scale, layer_sizes)
for epoch in range(num_epochs):
    start_time = time.time()
    for _ in range(num_batches):
        batch_data = next(batches)
        params = update(params, batch_data)
    epoch_time = time.time() - start_time
    y_ = x*params[0][0] + params[0][1]
    if (epoch + 1)%2 == 0:
        plt.plot(x, y, 'r.')
        plt.plot(x, y_, 'b-')
        plt.show()
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))