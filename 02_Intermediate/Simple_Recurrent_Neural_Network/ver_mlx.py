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
LEARNING_RATE = 0.001

train_images, train_labels, test_images, test_labels = mlx_dataset.mnist()
train_images = train_images.reshape([-1, 28, 28])
test_images = test_images.reshape([-1, 28, 28])

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)
    def __call__(self, x):
        h0 = mx.zeros((self.num_layers, x.shape[0], self.hidden_size))
        x, _ = self.rnn(x, h0)  
        x = self.fc(x[:, -1, :])
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


model  = Model(28, 128, 2, 10)
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