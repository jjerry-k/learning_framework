# Importing Modules
import random
from tqdm import tqdm

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
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

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2)

# Defining Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(100, 256)
        self.bnorm1 = nn.BatchNorm1d(256)
        self.linear2 = nn.Linear(256, 512)
        self.bnorm2 = nn.BatchNorm1d(512)
        self.linear3 = nn.Linear(512, 784)
        
    def forward(self, X):
        X = F.leaky_relu(self.bnorm1(self.linear1(X)), negative_slope=0.03)
        X = F.leaky_relu(self.bnorm2(self.linear2(X)), negative_slope=0.03)
        X = torch.sigmoid(self.linear3(X))
        return X
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(784, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 1)
    
    def forward(self, X):
        X = F.leaky_relu(self.linear1(X), negative_slope=0.03)
        X = F.leaky_relu(self.linear2(X), negative_slope=0.03)
        X = torch.sigmoid(self.linear3(X))
        return X

G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.MSELoss()
d_optimizer = optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = optim.Adam(G.parameters(), lr=0.0002)

# Helper Function
def plot_generator(G, num=10):
    z = torch.randn(num, 100).to(device)
    
    generated = G.forward(z).cpu().detach()
    plt.figure(figsize=(8, 2))
    for i in range(num):
        plt.subplot(1, num, i+1)
        plt.imshow(generated[i].view(28, 28), cmap=plt.cm.gray)
        plt.axis('off')
    plt.show()

plot_generator(G)

# Training loop
for epoch in range(epochs):
    G.train()
    D.train()
    avg_g_loss = 0
    avg_d_loss = 0
    
    with tqdm(total=len(train_loader)) as t:
        t.set_description(f'[{epoch+1}/{epochs}]')
        for i, (batch_img, _) in enumerate(train_loader):
            
            tmp_batch_siae = batch_img.shape[0]

            X = batch_img.view(tmp_batch_siae, -1).to(device)
        
            real_lab = torch.ones(tmp_batch_siae, 1).to(device)
            
            fake_lab = torch.zeros(tmp_batch_siae, 1).to(device)

            # Training Discriminator
            D_pred = D.forward(X)
            d_loss_real = criterion(D_pred, real_lab)
            real_score = D_pred
            
            z = torch.randn(tmp_batch_siae, 100).to(device)
            
            fake_images = G.forward(z)
            G_pred = D.forward(fake_images)
            d_loss_fake = criterion(G_pred, fake_lab)
            fake_score = G_pred
            
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            
            # Training Generator
            z = torch.randn(tmp_batch_siae, 100).to(device)
            fake_images = G.forward(z)
            G_pred = D.forward(fake_images)
            g_loss = criterion(G_pred, real_lab)
            
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            avg_g_loss += g_loss.item()
            avg_d_loss += d_loss.item()

            
            t.set_postfix({
                            "D loss": f"{d_loss.item():05.3f}", 
                            "G loss": f"{g_loss.item():05.3f}"
                            })
            t.update()

            
    print(f"Epoch : {epoch+1}, D Loss : {avg_d_loss/len(train_loader):.3f}, G Loss : {avg_g_loss/len(train_loader):.3f}")
    plot_generator(G)
print("Training Done !")

# Sample
num_sample = 64
z = torch.randn(num_sample, 100).to(device)

generated = G.forward(z).cpu().detach()
plt.figure(figsize=(8, 8))
for i in range(num_sample):
    plt.subplot(8, 8, i+1)
    plt.imshow(generated[i].view(28, 28), cmap=plt.cm.gray)
    plt.axis('off')
plt.show()