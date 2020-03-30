import os
import numpy as np
from mxnet import nd, gluon, init, autograd
from mxnet.gluon.data.vision import datasets
from mxnet.gluon import nn
from matplotlib import pyplot as plt
print("Load Package!")

train_raw_data = datasets.MNIST(train=True)
val_raw_data = datasets.MNIST(train=False)

train_data = {}
train_data['data'] = np.array([i[0].asnumpy() for i in train_raw_data])
train_data['label'] = np.array([i[1] for i in train_raw_data])
#train_data['label'] = np.array([np.eye(1, 10, k=i[1]).squeeze(axis=0) for i in train_raw_data])

print(train_data['data'].shape)
print(train_data['label'].shape)

val_data = {}
val_data['data'] = np.array([i[0].asnumpy() for i in val_raw_data])
val_data['label'] = np.array([i[1] for i in val_raw_data])
#val_data['label'] = np.array([np.eye(1, 10, k=i[1]).squeeze(axis=0) for i in val_raw_data])

print(val_data['data'].shape)
print(val_data['label'].shape)

# %%
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'), nn.Dense(10, activation='sigmoid'))

net.initialize(init=init.Xavier())

cross_entropy = gluon.loss.SoftmaxCELoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
#%%
print("Setting Done!")

batch_size = 100
tot_iter = len(train_data['data']) // batch_size
print("Start Training!")
for epoch in range(10):
    train_loss, train_acc, valid_acc = 0., 0., 0.
    #tic = time.time()
    # forward + backward
    for iter in range(tot_iter):
        idx = np.random.choice(len(train_data['data']), batch_size, replace=False)
        with autograd.record():
            output = net(nd.array(np.reshape(train_data['data'][idx], (batch_size, -1))))
            loss = cross_entropy(output, nd.array(train_data['label'][idx]))
        loss.backward()
        # update parameters
        trainer.step(batch_size)
        train_loss += loss.mean().asscalar()
        
    val_idx = np.random.choice(len(val_data['data']), 100, replace=False)
    output = nd.argmax(net(nd.array(np.reshape(val_data['data'][val_idx], (batch_size, -1)))), axis = 1).asnumpy()
    acc = np.mean(output == val_data['label'][val_idx])

    print("Epoch : %d, loss : %f, val_acc : %f"%(epoch+1, train_loss/batch_size, acc))