import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, losses, optimizers

EPOCHS = 500
LEARNING_RATE = 0.05

W = 0.1
B = 0.3

x = np.random.normal(0.0, 0.55, (10000, 1))
y = x * W + B + np.random.normal(0.0, 0.03, (10000,1))

model = models.Sequential()
model.add(layers.Dense(1))

loss = losses.MeanSquaredError()
optimizer = optimizers.SGD(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss=loss)

history = model.fit(x, y, epochs=EPOCHS, batch_size=10000, verbose=0)

param = model.weights
print(f"Real W: {W}, Predict W: {param[0].numpy().item():.3f}")
print(f"Real B: {B}, Predict B: {param[1].numpy().item():.3f}")