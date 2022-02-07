import torch
from torch import nn
from torch import optim
import numpy as np

W = 0.1
B = 0.3

x = np.random.normal(0.0, 0.55, (10000, 1))
y = x * W + B + np.random.normal(0.0, 0.03, (10000,1))

x_data = torch.Tensor(x)
y_data = torch.Tensor(y)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, X):
        X = self.linear(X)
        return X

model = Model()
criterion  = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

# Training loop
for epoch in range(500):
    y_pred = model.forward(x_data)

    loss = criterion(y_pred, y_data)
    
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch ==0 :
        print("Epoch : ", epoch+1, " Loss : ", loss.data.numpy())
    
    elif (epoch+1) % 100 == 0 :
        print("Epoch : ", epoch+1, " Loss : ", loss.data.numpy())
        
# After Training, check parameters
param = list(model.parameters())
print(f"Real W: {W}, Predict W: {param[0].item()}")
print(f"Real B: {B}, Predict B: {param[1].item()}")
# print(param[0].data.numpy())
# print(param[1].data.numpy())
