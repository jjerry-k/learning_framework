import sys
sys.path.append('../../')

from mlx import nn
from mlx import core as mx
from mlx import optimizers as optim
import numpy as np
np.random.seed(777)
mx.random.seed(777)

from utils import mlx_dataset

train_images, train_labels, test_images, test_labels = mlx_dataset.mnist()
train_labels = np.greater_equal(train_labels, 5).astype(float)[:, np.newaxis]
test_labels = np.greater_equal(test_labels, 5).astype(float)[:, np.newaxis]

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 1)

    def __call__(self, x):
        x = self.linear(x)
        return x

def loss_fn(model, x, y):
    output = model(mx.array(x))
    tgt = mx.array(y)
    # print(output.shape, tgt.shape)
    return mx.mean(nn.losses.binary_cross_entropy(output, tgt))

def eval_fn(x, y):
    return mx.mean(mx.greater_equal(mx.sigmoid(model(x)), 0.5) == y)

def batch_iterate(batch_size, x, y):
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        yield x[ids], y[ids]
num_epochs = 10
batch_size = 100

model  = Model()
mx.eval(model.parameters())

learning_rate = 0.01

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
optimizer = optim.SGD(learning_rate=learning_rate)

for epoch in range(num_epochs):
    avg_loss = 0
    for i, (batch_x, batch_y) in enumerate(batch_iterate(batch_size, train_images, train_labels)):
        
        
        loss, grads = loss_and_grad_fn(model, batch_x, batch_y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        avg_loss += loss
        
        if (i+1)%100 == 0 :
            print("Epoch : ", epoch+1, "Iteration : ", i+1, " Loss : ", avg_loss.item()/(i+1))
    accuracy = eval_fn(mx.array(test_images), mx.array(test_labels))
    print(f"Epoch: {epoch+1}, Loss: {avg_loss.item()/(i+1):.3f}, Accuracy: {accuracy.item():.3f}")