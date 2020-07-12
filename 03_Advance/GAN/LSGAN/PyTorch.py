#%%
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import os
import numpy as np
from matplotlib import pyplot as plt
#%%
def find_data_dir():
    data_path = 'data'
    while os.path.exists(data_path) != True:
        data_path = '../' + data_path
        
    return data_path
#%%
# MNIST dataset
mnist_train = datasets.MNIST(root='../',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)
print("Downloading Train Data Done ! ")

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# our model
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

batch_size = 100

data_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=1)
#%%
def plot_generator(num = 10):
    z = torch.randn(num, 100).to(device)
    
    test_g = G.forward(z)
    plt.figure(figsize=(8, 2))
    for i in range(num):
        plt.subplot(1, num, i+1)
        plt.imshow(test_g[i].view(28, 28).data.cpu().numpy(), cmap=plt.cm.gray)
        plt.axis('off')
    plt.show()
    

print("Iteration maker Done !")

# Training loop
for epoch in range(100):
    avg_loss = 0
    total_batch = len(mnist_train) // batch_size
    for i, (batch_img, _) in enumerate(data_iter):
        
        X = batch_img.view(batch_size, -1).to(device)
        
        real_lab = torch.ones(batch_size, 1).to(device)
        
        fake_lab = torch.zeros(batch_size, 1).to(device)
        
        # Training Discriminator
        D_pred = D.forward(X)
        d_loss_real = criterion(D_pred, real_lab)
        real_score = D_pred
        
        z = torch.randn(batch_size, 100).to(device)
        
        fake_images = G.forward(z)
        G_pred = D.forward(fake_images)
        d_loss_fake = criterion(G_pred, fake_lab)
        fake_score = G_pred
        
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        
        # Training Generator
        z = torch.randn(batch_size, 100).to(device)
        fake_images = G.forward(z)
        G_pred = D.forward(fake_images)
        g_loss = criterion(G_pred, real_lab)
        
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        
        if (i+1)%200 == 0 :
            print("Epoch : ", epoch+1, "Iteration : ", i+1, "G_loss : ", g_loss.data.cpu().numpy(), "D_loss : ", d_loss.data.cpu().numpy())
    plot_generator()
        
        
torch.save(G.state_dict(), './trained/LSGAN/sd_gen')
torch.save(D.state_dict(), './trained/LSGAN/sd_dis')

torch.save(G, './trained/LSGAN/gen.pt')
torch.save(D, './trained/LSGAN/dis.pt')