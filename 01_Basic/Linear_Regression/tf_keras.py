import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from matplotlib import pyplot as plt
tf.compat.v1.set_random_seed(777)

x = np.random.normal(0.0, 0.55, (10000, 1))
y = x * 0.1 + 0.3 + np.random.normal(0.0, 0.03, (10000,1))
                     
# plt.plot(x, y, 'r.')
# plt.show()

model = models.Sequential()
model.add(layers.Dense(1, input_shape=(1, )))

model.compile(optimizer='sgd', loss='mse')

history = model.fit(x, y, epochs=500, batch_size=1024)