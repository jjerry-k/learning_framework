import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt

x = np.random.normal(0.0, 0.55, (10000, 1))
y = x * 0.1 + 0.3 + np.random.normal(0.0, 0.03, (10000,1))
                     
#plt.plot(x, y, 'r.')
#plt.show()

x_data = Variable(torch.Tensor(x))
y_data = Variable(torch.Tensor(y))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1,1)

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
    
    
    if epoch ==0 :
        print("Epoch : ", epoch+1, " Loss : ", loss.data.numpy())
        #y_dis = model.forward(x_data).data.numpy()
        #plt.plot(x, y, 'r.')
        #plt.plot(x, y_dis, 'b.')
        #plt.show()
        
    elif (epoch+1) % 100 == 0 :
        print("Epoch : ", epoch+1, " Loss : ", loss.data.numpy())
        #y_dis = model.forward(x_data).data.numpy()
        #plt.plot(x, y, 'r.')
        #plt.plot(x, y_dis, 'b.')
        #plt.show()
        
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# After Training, check parameters
param = list(model.parameters())
print(len(param))
print(param[0].data.numpy())
print(param[1].data.numpy())
