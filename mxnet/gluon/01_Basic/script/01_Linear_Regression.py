# %%
import os
import numpy as np
from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn
from matplotlib import pyplot as plt
print("Load Package!")
# %%
x = np.random.normal(0.0, 0.55, (10000, 1))
y = x * 0.1 + 0.3 + np.random.normal(0.0, 0.03, (10000,1))

# %%
net = nn.Sequential()
net.add(nn.Dense(1))

net.initialize(init=init.Xavier())

cross_entropy = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
#%%
print("Setting Done!")

batch_size = 100
tot_iter = len(x) // batch_size
print("Start Training!")
for epoch in range(10):
    train_loss, train_acc, valid_acc = 0., 0., 0.
    #tic = time.time()
    # forward + backward
    for iter in range(tot_iter):
        idx = np.random.choice(len(x), batch_size, replace=False)
        with autograd.record():
            output = net(nd.array(x[idx]))
            loss = cross_entropy(output, nd.array(y[idx]))
        loss.backward()
        # update parameters
        trainer.step(batch_size)
        train_loss += loss.mean().asscalar()
    test_y = net.forward(nd.array(x)).asnumpy()
    print("Epoch : %d, loss : "%(epoch+1), train_loss/batch_size)
#     plt.plot(x, y, 'r.')
#     plt.plot(x, test_y, 'b-')
#     plt.show()
test_y = net.forward(nd.array(x)).asnumpy()
plt.plot(x, y, 'r.')
plt.plot(x, test_y, 'b-')
plt.show()
