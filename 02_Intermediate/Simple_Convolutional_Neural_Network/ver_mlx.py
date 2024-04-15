import sys
sys.path.append('../../')

from mlx import nn
from mlx import core as mx
from mlx import optimizers as optim
import numpy as np
np.random.seed(777)
mx.random.seed(777)

from utils import mlx_dataset

EPOCHS = 5
BATCH_SIZE = 256
LEARNING_RATE = 0.01

train_images, train_labels, test_images, test_labels = mlx_dataset.mnist()
train_images = train_images.reshape([-1, 28, 28, 1])
test_images = test_images.reshape([-1, 28, 28, 1])

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, 10)

    def __call__(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
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


model  = Model()
mx.eval(model.parameters())

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
optimizer = optim.Adam(learning_rate=LEARNING_RATE)

for epoch in range(EPOCHS):
    avg_loss = 0
    for i, (batch_x, batch_y) in enumerate(batch_iterate(BATCH_SIZE, train_images, train_labels)):
        loss, grads = loss_and_grad_fn(model, batch_x, batch_y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        avg_loss += loss
        
        if (i+1)%100 == 0 :
            print("Epoch : ", epoch+1, "Iteration : ", i+1, " Loss : ", avg_loss.item()/(i+1))
    accuracy = eval_fn(mx.array(test_images), mx.array(test_labels))
    print(f"Epoch: {epoch+1}, Loss: {avg_loss.item()/(i+1):.3f}, Accuracy: {accuracy.item():.3f}")