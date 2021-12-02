# %%
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

# %%
SAVE_PATH = "../../../data"
URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
file_name = URL.split("/")[-1]
data = datasets.utils.download_and_extract_archive(URL, SAVE_PATH)
PATH = os.path.join(SAVE_PATH, "flower_photos")

category_list = [i for i in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, i)) ]
print(category_list)

num_classes = len(category_list)
img_size = 128

def read_img(path, img_size):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (img_size, img_size))
    return img

imgs_tr = []
labs_tr = []

imgs_val = []
labs_val = []

for i, category in enumerate(category_list):
    path = os.path.join(PATH, category)
    imgs_list = os.listdir(path)
    print("Total '%s' images : %d"%(category, len(imgs_list)))
    ratio = int(np.round(0.05 * len(imgs_list)))
    print("%s Images for Training : %d"%(category, len(imgs_list[ratio:])))
    print("%s Images for Validation : %d"%(category, len(imgs_list[:ratio])))
    print("=============================")

    imgs = [read_img(os.path.join(path, img),img_size) for img in imgs_list]
    labs = [i]*len(imgs_list)

    imgs_tr += imgs[ratio:]
    labs_tr += labs[ratio:]
    
    imgs_val += imgs[:ratio]
    labs_val += labs[:ratio]

imgs_tr = np.array(imgs_tr)/255.
labs_tr = np.array(labs_tr)

imgs_val = np.array(imgs_val)/255.
labs_val = np.array(labs_val)

print(imgs_tr.shape, labs_tr.shape)
print(imgs_val.shape, labs_val.shape)

# %%
# Build network

class Conv_Block(nn.Module):
    def __init__(self, input_feature, output_feature, ksize=3, strides=1, padding=1):
        super(Conv_Block, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(input_feature, output_feature, ksize, strides, padding),
            nn.BatchNorm2d(output_feature),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class Depthwise_Separable_Block(nn.Module):
    def __init__(self, input_feature, output_feature, ksize=3, strides=1, padding=1):
        super(Depthwise_Separable_Block, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(input_feature, input_feature, ksize, strides, padding, groups=input_feature),
            nn.Conv2d(input_feature, output_feature, 1),
            nn.BatchNorm2d(output_feature),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class Residual_Block(nn.Module):
    def __init__(self, input_feature, intermediate_feature, output_feature):
        super(Residual_Block, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(input_feature, output_feature, 1, 2),
            nn.BatchNorm2d(output_feature)
        )
        self.block2 = nn.Sequential(
            Depthwise_Separable_Block(input_feature, intermediate_feature),
            nn.BatchNorm2d(intermediate_feature),
            nn.ReLU(True),

            Depthwise_Separable_Block(intermediate_feature, output_feature),
            nn.BatchNorm2d(output_feature),
            
            nn.MaxPool2d(3, 2, 1)
        )

    def forward(self, x):
        return self.block1(x) + self.block2(x)

class Middle_Flow(nn.Module):
    def __init__(self, features):
        super(Middle_Flow, self).__init__()

        self.block = nn.Sequential(
            nn.ReLU(True),
            Depthwise_Separable_Block(features, features),
            nn.BatchNorm2d(features),
            
            nn.ReLU(True),
            Depthwise_Separable_Block(features, features),
            nn.BatchNorm2d(features),

            nn.ReLU(True),
            Depthwise_Separable_Block(features, features),
            nn.BatchNorm2d(features)
        )

    def forward(self, x):
        return self.block(x) + x

class Build_Xception(nn.Module):
    def __init__(self, input_channel= 3, num_classes=1000):
        super(Build_Xception, self).__init__()

        self.stem = nn.Sequential(
            Conv_Block(input_channel, 32, 3, 2, 1),
            Conv_Block(32, 64)
        )

        self.entry_block = nn.Sequential(
            Residual_Block(64, 128, 128),
            Residual_Block(128, 256, 256),
            Residual_Block(256, 728, 728)
        )

        self.middle_block = nn.Sequential(
            *[Middle_Flow(728) for _ in range(8)]
        )

        self.exit_block = nn.Sequential(
            Residual_Block(728, 728, 1024),
            Depthwise_Separable_Block(1024, 1536),
            Depthwise_Separable_Block(1536, 2048)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(2048, num_classes)

        self.init_weights(self.stem)
        self.init_weights(self.entry_block)
        self.init_weights(self.middle_block)
        self.init_weights(self.exit_block)
        self.init_weights(self.classifier)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.stem(x)
        x = self.entry_block(x)
        x = self.middle_block(x)
        x = self.exit_block(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

net = Build_Xception(input_channel=imgs_tr.shape[-1], num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# %%
epochs=100
batch_size=16

class CustomDataset(Dataset):
    def __init__(self, train_x, train_y): 
        self.len = len(train_x) 
        self.x_data = torch.tensor(np.transpose(train_x, [0, 3, 1, 2]), dtype=torch.float)
        self.y_data = torch.tensor(train_y, dtype=torch.long) 

    def __getitem__(self, index): 
        return self.x_data[index], self.y_data[index] 

    def __len__(self): 
        return self.len
        
train_dataset = CustomDataset(imgs_tr, labs_tr) 
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

val_dataset = CustomDataset(imgs_val, labs_val) 
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

print("Iteration maker Done !")

# %%
# Training Network

for epoch in range(epochs):
    net.train()
    avg_loss = 0
    avg_acc = 0
    
    with tqdm(total=len(train_loader)) as t:
        t.set_description(f'[{epoch+1}/{epochs}]')
        total = 0
        correct = 0
        for i, (batch_img, batch_lab) in enumerate(train_loader):
            X = batch_img.to(device)
            Y = batch_lab.to(device)

            optimizer.zero_grad()

            y_pred = net.forward(X)

            loss = criterion(y_pred, Y)
            
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

            _, predicted = torch.max(y_pred.data, 1)
            total += Y.size(0)
            correct += (predicted == Y).sum().item()
            
            t.set_postfix({"loss": f"{loss.item():05.3f}"})
            t.update()
        acc = (100 * correct / total)

    net.eval()
    with tqdm(total=len(val_loader)) as t:
        t.set_description(f'[{epoch+1}/{epochs}]')
        with torch.no_grad():
            val_loss = 0
            total = 0
            correct = 0
            for i, (batch_img, batch_lab) in enumerate(val_loader):
                X = batch_img.to(device)
                Y = batch_lab.to(device)
                y_pred = net(X)
                val_loss += criterion(y_pred, Y)
                _, predicted = torch.max(y_pred.data, 1)
                total += Y.size(0)
                correct += (predicted == Y).sum().item()
                t.set_postfix({"val_loss": f"{val_loss.item()/(i+1):05.3f}"})
                t.update()

            val_loss /= total
            val_acc = (100 * correct / total)
            
    print(f"Epoch : {epoch+1}, Loss : {(avg_loss/len(train_loader)):.3f}, Acc: {acc:.3f}, Val Loss : {val_loss.item():.3f}, Val Acc : {val_acc:.3f}")

print("Training Done !")