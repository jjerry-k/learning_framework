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

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 256)
        self.linear2 = nn.Linear(256, 10)

    def __call__(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

def loss_fn(model, x, y):
    x = mx.array(x)
    y = mx.array(y)
    return mx.mean(nn.losses.cross_entropy(model(x), y))

def eval_fn(x, y):
    return mx.mean(mx.argmax(model(x), axis=1) == y)

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
optimizer = optim.Adam(learning_rate=learning_rate)

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