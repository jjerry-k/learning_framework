# %% 
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, losses, optimizers
# from matplotlib import pyplot as plt
tf.random.set_seed(777)
print("Package Loaded!")
# %%
EPOCHS = 500
BATCH_SIZE = 128

x = np.random.normal(0.0, 0.55, (10000, 1))
y = x * 0.1 + 0.3 + np.random.normal(0.0, 0.03, (10000,1))

train_ds = tf.data.Dataset.from_tensor_slices(
    (x, y)).shuffle(10000).batch(BATCH_SIZE)

# plt.plot(x, y, 'r.')
# plt.show()
print("Data Prepared!")

# %%
class LinearRegression(models.Model):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.d = layers.Dense(1, input_shape=(1, ))

    def call(self, x):
        return self.d(x)

# Create an instance of the model
model = LinearRegression()

loss_object = losses.MeanSquaredError()

optimizer = optimizers.Adam()

# %%
for epoch in range(EPOCHS):
    for batch_x, batch_y in train_ds:
        with tf.GradientTape() as tape:
            predictions = model(batch_x, training=True)
            loss = loss_object(batch_y, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    print("{:5}|{:10.6f}".format(epoch+1, loss))