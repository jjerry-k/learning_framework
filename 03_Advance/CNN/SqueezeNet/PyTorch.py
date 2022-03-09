# Importing Modules
import os
import random
from tqdm import tqdm

import numpy as np
from PIL import Image

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader 

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
epochs= 5
batch_size= 16
img_size= 192

# Dataset
class FlowerDataset(Dataset):
    def __init__(self, data_dir, transform):
        IMG_FORMAT = ["jpg", "jpeg", "bmp", "png", "tif", "tiff"]
        self.filelist = []
        self.classes = sorted(os.listdir(data_dir))
        for root, _, files in os.walk(data_dir):
            if not len(files): continue
            files = [os.path.join(root, file) for file in files if file.split(".")[-1] in IMG_FORMAT]
            self.filelist += files
        # self.filelist = self.filelist[:64]
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.filelist)

    def __getitem__(self, idx):

        image = Image.open(self.filelist[idx]).convert("RGB")
        image = self.transform(image)
        label = self.filelist[idx].split('/')[-2]
        label = self.classes.index(label)
        return image, label

transform = transforms.Compose([
                                transforms.Resize((img_size, img_size)), transforms.ToTensor()
                                ])
train_dataset = FlowerDataset(os.path.join("../../../data/flower_photos/train"), transform)
val_dataset = FlowerDataset(os.path.join("../../../data/flower_photos/validation"), transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Defining Model
class Conv_Block(nn.Module):
    def __init__(self, input_feature, output_feature, ksize=3, strides=1, padding=1, use_relu=True, use_bn=False):
        super(Conv_Block, self).__init__()
        
        layer_list = []

        layer_list.append(nn.Conv2d(input_feature, output_feature, ksize, strides, padding))
        
        if use_bn:
            layer_list.append(nn.BatchNorm2d(output_feature))

        if use_relu:
            layer_list.append(nn.ReLU(True))

        self.block = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.block(x)

class Fire_Module(nn.Module):
    def __init__(self, input_feature, squ, exp_1x1, exp_3x3, use_bn=False):
        super(Fire_Module, self).__init__()

        self.squeeze = Conv_Block(input_feature, squ, 1, 1, 0)

        self.expand_1x1 = Conv_Block(squ, exp_1x1, 1, 1, 0, False)
        self.expand_3x3 = Conv_Block(squ, exp_3x3, 3, 1, 1, False)

        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.squeeze(x)

        exp_1x1 = self.expand_1x1(out)
        exp_3x3 = self.expand_3x3(out)

        expand = torch.cat([exp_1x1, exp_3x3], dim=1)

        out = self.relu(expand)
        return out

class SqueezeNet(nn.Module):
    def __init__(self, input_channel=3, num_classes=1000):
        super(SqueezeNet, self).__init__()

        self.Stem = Conv_Block(input_channel, 96, 7, 2, 3)

        layer_list = []
        layer_list.append(Fire_Module(96, 16, 64, 64))
        layer_list.append(Fire_Module(128, 16, 64, 64))
        layer_list.append(Fire_Module(128, 32, 128, 128))
        layer_list.append(nn.MaxPool2d(3, 2))

        layer_list.append(Fire_Module(256, 32, 128, 128))
        layer_list.append(Fire_Module(256, 48, 192, 192))
        layer_list.append(Fire_Module(384, 48, 192, 192))
        layer_list.append(Fire_Module(384, 64, 256, 256))
        layer_list.append(nn.MaxPool2d(3, 2))

        layer_list.append(Fire_Module(512, 64, 256, 256))       

        self.Main_Block = nn.Sequential(*layer_list)

        self.Classifier = nn.Sequential(
            nn.Dropout(0.5), 
            Conv_Block(512, num_classes, 1, 1, 0),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.Stem(x)
        x = self.Main_Block(x)
        x = self.Classifier(x)
        return x

model = SqueezeNet(input_channel=3, num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training
for epoch in range(epochs):
    model.train()
    avg_loss = 0
    avg_acc = 0
    
    with tqdm(total=len(train_loader)) as t:
        t.set_description(f'[{epoch+1}/{epochs}]')
        total = 0
        correct = 0
        for i, (batch_img, batch_lab) in enumerate(train_loader):
            X = batch_img.to(device)
            Y = batch_lab.to(device)

            y_pred = model.forward(X)

            loss = criterion(y_pred, Y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

            _, predicted = torch.max(y_pred.data, 1)
            total += Y.size(0)
            correct += (predicted == Y).sum().item()
            
            t.set_postfix({"loss": f"{avg_loss/(i+1):05.3f}"})
            t.update()
        acc = (100 * correct / total)

    model.eval()
    with tqdm(total=len(val_loader)) as t:
        t.set_description(f'[{epoch+1}/{epochs}]')
        with torch.no_grad():
            val_loss = 0
            total = 0
            correct = 0
            for i, (batch_img, batch_lab) in enumerate(val_loader):
                X = batch_img.to(device)
                Y = batch_lab.to(device)
                y_pred = model(X)
                val_loss += criterion(y_pred, Y)
                _, predicted = torch.max(y_pred.data, 1)
                total += Y.size(0)
                correct += (predicted == Y).sum().item()
                t.set_postfix({"val_loss": f"{val_loss.item()/(i+1):05.3f}"})
                t.update()

            val_loss /= len(val_loader)
            val_acc = (100 * correct / total)
            
    print(f"Epoch : {epoch+1}, Loss : {(avg_loss/len(train_loader)):.3f}, Acc: {acc:.3f}, Val Loss : {val_loss.item():.3f}, Val Acc : {val_acc:.3f}\n")

print("Training Done !")