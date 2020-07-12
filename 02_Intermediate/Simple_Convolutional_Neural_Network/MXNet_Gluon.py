# %%
import os
import numpy as np
import mxnet as mx
from mxnet import io, nd, gluon, init, autograd
from mxnet.gluon.data.vision import datasets
from mxnet.gluon import nn, data
from matplotlib import pyplot as plt
from multiprocessing import cpu_count
CPU_COUNT = cpu_count()
print("Package Loaded!")

# %%
train_raw_data = datasets.MNIST(train=True)
val_raw_data = datasets.MNIST(train=False)

train_data = {}
train_data['data'] = np.array([i[0].asnumpy() for i in train_raw_data])
train_data['label'] = np.array([i[1] for i in train_raw_data])

print(train_data['data'].shape)
print(train_data['label'].shape)

val_data = {}
val_data['data'] = np.array([i[0].asnumpy() for i in val_raw_data])
val_data['label'] = np.array([i[1] for i in val_raw_data])

print(val_data['data'].shape)
print(val_data['label'].shape)

# %%
net = nn.Sequential()
net.add(
    nn.Conv2D(16, (3, 3), (1, 1), (1, 1), activation='relu'),
    nn.MaxPool2D((2, 2), (2, 2)),
    nn.Conv2D(32, (3, 3), (1, 1), (1, 1), activation='relu'),
    nn.MaxPool2D((2, 2), (2, 2)),
    nn.Flatten(),
    nn.Dense(10, activation='sigmoid')
    )

gpus = mx.test_utils.list_gpus()
ctx =  [mx.gpu()] if gpus else [mx.cpu()]

net.initialize(init=init.Xavier(), ctx=ctx)

cross_entropy = gluon.loss.SoftmaxCELoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
print("Setting Done!")

# %%

epochs=100
batch_size=16

class DataIterLoader():
    def __init__(self, X, Y, batch_size=1, shuffle=True, ctx=mx.cpu()):
        self.data_iter = io.NDArrayIter(data=gluon.utils.split_and_load(np.transpose(X, [0, 3, 1, 2]), ctx_list=ctx, batch_axis=0), 
                                        label=gluon.utils.split_and_load(Y, ctx_list=ctx, batch_axis=0), 
                                        batch_size=batch_size, shuffle=shuffle)
        self.len = len(X)

    def __iter__(self):
        self.data_iter.reset()
        return self

    def __next__(self):
        batch = self.data_iter.__next__()
        assert len(batch.data) == len(batch.label) == 1
        data = batch.data[0]
        label = batch.label[0]
        return data, label

train_loader = DataIterLoader(train_data['data'], train_data['label'], batch_size, ctx=ctx)
validation_loader = DataIterLoader(val_data['data'], val_data['label'], batch_size, ctx=ctx)

print("Start Training!")
for epoch in range(epochs):
    train_loss, train_acc, valid_loss, valid_acc = 0., 0., 0., 0.
    #tic = time.time()
    # forward + backward
    for step, (batch_img, batch_lab) in enumerate(train_loader):
        
        with autograd.record():
            output = net(batch_img)
            loss = cross_entropy(output, batch_lab)
        loss.backward()
        # update parameters
        trainer.step(batch_size)
        correct = nd.argmax(output, axis = 1).asnumpy()
        acc = np.mean(correct == batch_lab.asnumpy())
        train_loss += loss.mean().asnumpy()
        train_acc += acc

    for idx, (val_img, val_lab) in enumerate(validation_loader):
        output = net(val_img)
        loss = cross_entropy(output, val_lab)
        correct = nd.argmax(output, axis = 1).asnumpy()
        acc = np.mean(correct == val_lab.asnumpy())
        valid_loss += loss.asnumpy().mean()
        valid_acc += acc
        
    print("Epoch : %d, loss : %f, acc : %f, val_loss : %f,  val_acc : %f"%(epoch+1, train_loss/(step+1), train_acc/(step+1), valid_loss/(idx+1), valid_acc/(idx+1)))