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

class BuildXception(nn.Module):
    def __init__(self, input_channel= 3, num_classes=1000):
        super(BuildXception, self).__init__()

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
        x = self.stem(x)
        x = self.entry_block(x)
        x = self.middle_block(x)
        x = self.exit_block(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = BuildXception(input_channel=3, num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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