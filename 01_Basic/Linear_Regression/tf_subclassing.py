import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, losses, optimizers

EPOCHS = 500
LEARNING_RATE = 0.05

W = 0.1
B = 0.3

x = np.random.normal(0.0, 0.55, (10000, 1))
y = x * W + B + np.random.normal(0.0, 0.03, (10000,1))

class LinearRegression(models.Model):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.d = layers.Dense(1)

    def call(self, x):
        return self.d(x)

# Create an instance of the model
model = LinearRegression()

loss_object = losses.MeanSquaredError()

optimizer = optimizers.SGD(learning_rate=LEARNING_RATE)

for epoch in range(500):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_object(y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if (epoch == 0) or ((epoch+1) % 100 == 0):
        print(f"Epoch: {epoch+1} Loss: {loss}")

param = model.weights
print(f"Real W: {W}, Predict W: {param[0].numpy().item():.3f}")
print(f"Real B: {B}, Predict B: {param[1].numpy().item():.3f}")