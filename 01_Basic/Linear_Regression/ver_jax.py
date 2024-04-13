import time
import numpy as np
from jax import jit, grad
from jax.scipy.special import logsumexp
import jax.numpy as jnp

EPOCHS = 500
LEARNING_RATE = 0.05


W = 0.1
B = 0.3

x = np.random.normal(0.0, 0.55, (10000, 1))
y = x * W + B + np.random.normal(0.0, 0.03, (10000,1))

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

layer_sizes = [1, 1]
param_scale= 1

@jit
def update(params, batch):
    grads = grad(loss)(params, batch)
    return [(w - LEARNING_RATE * dw, b - LEARNING_RATE * db)
            for (w, b), (dw, db) in zip(params, grads)]

params = init_random_params(param_scale, layer_sizes)
for epoch in range(EPOCHS):
    start_time = time.time()
    params = update(params, (x, y))
    epoch_time = time.time() - start_time
    y_ = x*params[0][0] + params[0][1]
    if (epoch == 0) or ((epoch+1) % 100 == 0):
        print(f"Epoch: {epoch+1} Loss: {loss(params, (x, y))}")

# After Training, check parameters
print(f"Real W: {W}, Predict W: {params[0][0].item():.3f}")
print(f"Real B: {B}, Predict B: {params[0][1].item():.3f}")