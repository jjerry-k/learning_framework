import os
from tqdm import tqdm

import cv2 as cv
import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms, datasets, utils

# Device Configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# MNIST dataset
train_dataset = datasets.MNIST(root='../../../data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)
print("Downloading Train Data Done ! ")

val_dataset = datasets.MNIST(root='../../../data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)
print("Downloading Test Data Done ! ")

# Build network
class build_CAE(nn.Module):
    def __init__(self, input_features=1):
        super(build_CAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_features, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, input_features, 4, 2, 1),
            nn.Sigmoid()
        )

        self.init_weights(self.encoder)
        self.init_weights(self.decoder)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

net = build_CAE(input_features=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr = 0.0001)

epochs=10
batch_size=256

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=2)

print("Iteration maker Done !")

# Training Network

for epoch in range(epochs):
    net.train()
    avg_loss = 0
    
    with tqdm(total=len(train_loader)) as t:
        t.set_description(f'[{epoch+1}/{epochs}]')
        for i, (batch_img, batch_lab) in enumerate(train_loader):
            
            X = batch_img.to(device)
            
            optimizer.zero_grad()
            y_pred = net.forward(X)
            loss = criterion(y_pred, X)
            
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            
            t.set_postfix({"loss": f"{loss.item():05.3f}"})
            t.update()

    net.eval()
    with tqdm(total=len(val_loader)) as t:
        t.set_description(f'[{epoch+1}/{epochs}]')
        with torch.no_grad():
            val_loss = 0
            for i, (batch_img, batch_lab) in enumerate(val_loader):
                
                X = batch_img.to(device)
                
                y_pred = net(X)
                val_loss += criterion(y_pred, X)
                t.set_postfix({"val_loss": f"{val_loss.item()/(i+1):05.3f}"})
                t.update()

            val_loss /= len(val_loader)
            
    print(f"Epoch : {epoch+1}, Loss : {(avg_loss/len(train_loader)):.3f}, Val Loss : {val_loss.item():.3f}")

print("Training Done !")