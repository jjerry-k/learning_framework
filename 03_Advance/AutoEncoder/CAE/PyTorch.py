# Importing Modules
import random
from tqdm import tqdm

import numpy as np

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader 

from torchvision import datasets
from torchvision import transforms

from matplotlib import pyplot as plt

# Device Configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Set randomness
seed = 777
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set hyperparameter
epochs= 10
batch_size= 256

# MNIST dataset
mnist_train = datasets.MNIST(root='../../../data/',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
print("Downloading Train Data Done ! ")

mnist_test = datasets.MNIST(root='../../../data/',
                            train=False,
                            transform=transforms.ToTensor(),
                            download=True)
print("Downloading Test Data Done ! ")

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=2)

# Defining Model
class BuildCAE(nn.Module):
    def __init__(self, input_features=1):
        super(BuildCAE, self).__init__()

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

model = BuildCAE(input_features=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)

for epoch in range(epochs):
    model.train()
    avg_loss = 0
    
    with tqdm(total=len(train_loader)) as t:
        t.set_description(f'[{epoch+1}/{epochs}]')
        for i, (batch_img, batch_lab) in enumerate(train_loader):
            
            X = batch_img.to(device)
            
            optimizer.zero_grad()
            y_pred = model.forward(X)
            loss = criterion(y_pred, X)
            
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            
            t.set_postfix({"loss": f"{loss.item():05.3f}"})
            t.update()

    model.eval()
    with tqdm(total=len(val_loader)) as t:
        t.set_description(f'[{epoch+1}/{epochs}]')
        with torch.no_grad():
            val_loss = 0
            for i, (batch_img, batch_lab) in enumerate(val_loader):
                
                X = batch_img.to(device)
                
                y_pred = model(X)
                val_loss += criterion(y_pred, X)
                t.set_postfix({"val_loss": f"{val_loss.item()/(i+1):05.3f}"})
                t.update()

            val_loss /= len(val_loader)
            
    print(f"Epoch : {epoch+1}, Loss : {(avg_loss/len(train_loader)):.3f}, Val Loss : {val_loss.item():.3f}")

print("Training Done !")