import mxnet as mx
import numpy as np
from matplotlib import pyplot as plt
import logging

mx.random.seed(42)

x = np.random.normal(0.0, 0.55, (10000, 1))
y = x * 0.1 + 0.3 + np.random.normal(0.0, 0.03, (10000,1))

# Set the compute machine
ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
                     
plt.plot(x, y, 'r.')
plt.show()

batch_size = 100

X = mx.sym.Variable('data')
Y = mx.symbol.Variable('lin_reg_label')
fully_connected_layer = mx.sym.FullyConnected(data=X, name='fc1', num_hidden = 1)
lro = mx.sym.LinearRegressionOutput(data=fully_connected_layer, label=Y, name="lro")

train_iter = mx.io.NDArrayIter(x, y, batch_size, shuffle=True, label_name='lin_reg_label')
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
# create a trainable module on compute context
model = mx.mod.Module(
    symbol = lro ,
    data_names=['data'],
    label_names = ['lin_reg_label']# network structure
)
model.fit(train_iter,  # train data
              optimizer='sgd',  # use SGD to train
              optimizer_params={'learning_rate':0.1},  # use fixed learning rate
              eval_metric='mse',  # report accuracy during training
              batch_end_callback = mx.callback.Speedometer(batch_size, 100), # output progress for each 100 data batches
              num_epoch=10)  # train for at most 10 dataset passes