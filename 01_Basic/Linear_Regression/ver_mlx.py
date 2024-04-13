from mlx import nn
from mlx import core as mx
from mlx import optimizers as optim
import numpy as np

EPOCHS = 500
LEARNING_RATE = 0.05

W = 0.1
B = 0.3

x = np.random.normal(0.0, 0.55, (10000, 1))
y = x * W + B + np.random.normal(0.0, 0.03, (10000, 1))

x_mx_arr = mx.array(x)
y_mx_arr = mx.array(y)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def __call__(self, x):
        x = self.linear(x)
        return x

def loss_fn(model, X, y):
    return mx.mean(nn.losses.mse_loss(model(X), y))

def eval_fn(model, X, y):
    return mx.mean((model(X) - y)**2)

model  = Model()

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
optimizer = optim.SGD(learning_rate=LEARNING_RATE)

for epoch in range(EPOCHS):
    loss, grads = loss_and_grad_fn(model, x_mx_arr, y_mx_arr)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    accuracy = eval_fn(model, x_mx_arr, y_mx_arr)
    if (epoch == 0) or ((epoch+1) % 100 == 0):
        print(f"Epoch: {epoch+1}: Loss: {loss.item()}")
        
param = (model.linear.weight, model.linear.bias)
print(f"Real W: {W}, Predict W: {param[0].item():.3f}")
print(f"Real B: {B}, Predict B: {param[1].item():.3f}")