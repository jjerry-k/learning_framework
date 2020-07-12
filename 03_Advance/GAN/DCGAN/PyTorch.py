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

def find_data_dir():
    data_path = 'data'
    while os.path.exists(data_path) != True:
        data_path = '../' + data_path
        
    return data_path

# MNIST dataset

transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
])

mnist_train = datasets.MNIST(root=find_data_dir(),
                          train=True,
                          transform=transform,
                          download=True)
print("Downloading Train Data Done ! ")

mnist_test = datasets.MNIST(root=find_data_dir(),
                         train=False,
                         transform=transform,
                         download=True)
print("Downloading Test Data Done ! ")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# our model
class Generator(nn.Module):
    def __init__(self, n_z=100, d=128):
        super(Generator, self).__init__() 
        self.deconv1 = nn.ConvTranspose2d(n_z, d*8, 4, 1, 0),
        self.bnorm1 = nn.BatchNorm2d(d*8),
        
        self.devonc2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1),
        self.bnorm2 = nn.BatchNorm2d(d*4),
        
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1),
        self.bnorm3 = nn.BatchNorm2d(d*2),
        
        self.deconv4 = nn.ConvTranspose2d(d*2, 1, 4, 2, 1),
        
                    
    def forward(self, X):
        X = F.relu(self.bnorm1(self.deconv1(X)))
        X = F.relu(self.bnorm2(self.deconv2(X)))
        X = F.relu(self.bnorm3(self.deconv3(X)))
        X = F.tanh(self.deconv4(X))
        return X
    
class Discriminator(nn.Module):
    def __init__(self, d=128):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1),
        self.bnorm1 = nn.BatchNorm2d(d),
        
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1),
        self.bnorm2 = nn.BatchNorm2d(d*2),
        
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1),
        self.bnorm3 = nn.BatchNorm2d(d*4),
        
        self.conv4 = nn.Conv2d(d*4, 1, 4, 1, 0),
        
    def forward(self, X):
        X = F.leaky_relu(self.bnorm1(self.conv1(X)), negative_slope=0.003)
        X = F.leaky_relu(self.bnorm2(self.conv2(X)), negative_slope=0.003)
        X = F.leaky_relu(self.bnorm3(self.conv3(X)), negative_slope=0.003)
        X = torch.sigmoid(self.conv1(X))
        return X

G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

batch_size = 100

data_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=1)

def plot_generator(num = 10):
    z = torch.randn(num, 100, 1, 1).to(device)
    
    test_g = G.forward(z)
    plt.figure(figsize=(8, 2))
    for i in range(num):
        plt.subplot(1, num, i+1)
        plt.imshow(test_g[i].view(32, 32).data.cpu().numpy(), cmap=plt.cm.gray)
        plt.axis('off')
    plt.show()
    

print("Iteration maker Done !")
history = {}
history['g_loss']=[]
history['d_loss']=[]
# Training loop
for epoch in range(10):
    avg_loss = 0
    total_batch = len(mnist_train) // batch_size
    
    for i, (batch_img, _) in enumerate(data_iter):
        
        # Preparing train data
        X = batch_img.to(device)
        
        real_lab = torch.ones(batch_size, 1).to(device)
        
        fake_lab = torch.zeros(batch_size, 1).to(device)
        
        
        # Training Discriminator
        D_pred = D.forward(X)
        d_loss_real = criterion(D_pred.view(-1, 1), real_lab)
        real_score = D_pred
        
        z = torch.randn(batch_size, 100, 1, 1).to(device)
        
        fake_images = G.forward(z)
        G_pred = D.forward(fake_images)
        d_loss_fake = criterion(G_pred.view(-1, 1), fake_lab)
        fake_score = G_pred
        
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        
        # Training Generator
        z = torch.randn(batch_size, 100, 1, 1).to(device)
        fake_images = G.forward(z)
        G_pred = D.forward(fake_images)
        
        g_loss = criterion(G_pred.view(-1, 1), real_lab)
        
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        
        history['g_loss'].append(g_loss.data.cpu().numpy())
        history['d_loss'].append(d_loss.data.cpu().numpy())
        
        if (i+1)%100 == 0 :
            print("Epoch : ", epoch+1, "Iteration : ", i+1, "G_loss : ", g_loss.data.cpu().numpy(), "D_loss : ", d_loss.data.cpu().numpy())
    plot_generator()
    
torch.save(G.state_dict(), './trained/DCGAN/sd_gen')
torch.save(D.state_dict(), './trained/DCGAN/sd_dis')

torch.save(G, './trained/DCGAN/gen.pt')
torch.save(D, './trained/DCGAN/dis.pt')

plt.figure(figsize=(8,4))
plt.plot(history['g_loss'], 'r-')
plt.plot(history['d_loss'], 'b-')
plt.legend(['g_loss', 'd_loss'], loc=1)
plt.show()