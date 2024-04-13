import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import numpy as np

EPOCHS = 10
BATCH_SIZE = 100
LEARNING_RATE = 0.01

# MNIST dataset
mnist_train = datasets.MNIST(root="../../data",
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
print("Downloading Train Data Done ! ")

mnist_test = datasets.MNIST(root="../../data",
                            train=False,
                            transform=transforms.ToTensor(),
                            download=True)
print("Downloading Test Data Done ! ")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# our model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(784,1)

    def forward(self, X):
        X = self.linear(X)
        return X

model = Model().to(device)

criterion  = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

data_iter = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    avg_loss = 0
    total_batch = len(mnist_train)//BATCH_SIZE
    for i, (batch_img, batch_lab) in enumerate(data_iter):

        # 0 : digit < 5
        # 1 : digit >= 5
        X = batch_img.view(-1, 28*28).to(device)

        # To use BCEWithLogitsLoss
        # 1. Target tensor must be same as predict result's size 
        # 2. Target tensor's type must be Float
        Y = batch_lab.unsqueeze(dim=1) 
        Y = Y.type(torch.FloatTensor).to(device) 
        Y[Y>=5] = 1
        Y[Y<5] = 0
        

        y_pred = model.forward(X)
        loss = criterion(y_pred, Y)
        # Zero gradients, perform a backward pass, and update the weights.

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss

        if (i+1)%100 == 0 :
            print("Epoch : ", epoch+1, "Iteration : ", i+1, " Loss : ", avg_loss.data.cpu().numpy()/(i+1))
    print("Epoch : ", epoch+1, " Loss : ", avg_loss.data.cpu().numpy()/(i+1))
print("Training Done !")