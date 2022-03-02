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
class Inception_Module_A(nn.Module):
    def __init__(self, input_feature, 
                filters_b1, filters_b2_1, filters_b2_2, 
                filters_b3_1, filters_b3_2, filters_b3_3, filters_b4):
        super(Inception_Module_A, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b1, 1), 
            nn.BatchNorm2d(filters_b1),
            nn.ReLU(True))

        self.branch2 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b2_1, 1),
            nn.BatchNorm2d(filters_b2_1),
            nn.ReLU(True),
            nn.Conv2d(filters_b2_1, filters_b2_2, 3, 1, 1),
            nn.BatchNorm2d(filters_b2_2),
            nn.ReLU(True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b3_1, 1),
            nn.BatchNorm2d(filters_b3_1),
            nn.ReLU(True),
            nn.Conv2d(filters_b3_1, filters_b3_2, 3, 1, 1),
            nn.BatchNorm2d(filters_b3_2),
            nn.ReLU(True),
            nn.Conv2d(filters_b3_2, filters_b3_3, 3, 1, 1),
            nn.BatchNorm2d(filters_b3_3),
            nn.ReLU(True)
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            nn.Conv2d(input_feature, filters_b4, 1),
            nn.BatchNorm2d(filters_b4),
            nn.ReLU(True)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class Inception_Module_B(nn.Module):
    def __init__(self, input_feature, 
                filters_b1, filters_b2_1, filters_b2_2, filters_b2_3, 
                filters_b3_1, filters_b3_2, filters_b3_3, filters_b3_4, filters_b3_5, 
                filters_b4):
        super(Inception_Module_B, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b1, 1), 
            nn.BatchNorm2d(filters_b1),
            nn.ReLU(True))

        self.branch2 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b2_1, 1),
            nn.BatchNorm2d(filters_b2_1),
            nn.ReLU(True),
            nn.Conv2d(filters_b2_1, filters_b2_2, (7, 1), 1, (3, 0)),
            nn.BatchNorm2d(filters_b2_2),
            nn.ReLU(True),
            nn.Conv2d(filters_b2_2, filters_b2_3, (1, 7), 1, (0, 3)),
            nn.BatchNorm2d(filters_b2_3),
            nn.ReLU(True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b3_1, 1),
            nn.BatchNorm2d(filters_b3_1),
            nn.ReLU(True),
            nn.Conv2d(filters_b3_1, filters_b3_2, (7, 1), 1, (3, 0)),
            nn.BatchNorm2d(filters_b3_2),
            nn.ReLU(True),
            nn.Conv2d(filters_b3_2, filters_b3_3, (1, 7), 1, (0, 3)),
            nn.BatchNorm2d(filters_b3_3),
            nn.ReLU(True),
            nn.Conv2d(filters_b3_3, filters_b3_4, (7, 1), 1, (3, 0)),
            nn.BatchNorm2d(filters_b3_4),
            nn.ReLU(True),
            nn.Conv2d(filters_b3_4, filters_b3_5, (1, 7), 1, (0, 3)),
            nn.BatchNorm2d(filters_b3_5),
            nn.ReLU(True)
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            nn.Conv2d(input_feature, filters_b4, 1),
            nn.BatchNorm2d(filters_b4),
            nn.ReLU(True)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)

class Inception_Module_C(nn.Module):
    def __init__(self, input_feature, 
                filters_b1, filters_b2_1, filters_b2_2, filters_b2_3, 
                filters_b3_1, filters_b3_2, filters_b3_3, filters_b3_4, 
                filters_b4):
        super(Inception_Module_C, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b1, 1), 
            nn.BatchNorm2d(filters_b1),
            nn.ReLU(True))

        self.branch2_block_1 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b2_1, 1),
            nn.BatchNorm2d(filters_b2_1),
            nn.ReLU(True)
        )

        self.branch2_block_2_1 = nn.Sequential(
            nn.Conv2d(filters_b2_1, filters_b2_2, (1, 3), 1, (0, 1)),
            nn.BatchNorm2d(filters_b2_2),
            nn.ReLU(True)
        )

        self.branch2_block_2_2 = nn.Sequential(
            nn.Conv2d(filters_b2_1, filters_b2_3, (3, 1), 1, (1, 0)),
            nn.BatchNorm2d(filters_b2_3),
            nn.ReLU(True)
        )

        self.branch3_block_1 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b3_1, 1),
            nn.BatchNorm2d(filters_b3_1),
            nn.ReLU(True),
            nn.Conv2d(filters_b3_1, filters_b3_2, 3, 1, 1),
            nn.BatchNorm2d(filters_b3_2),
            nn.ReLU(True)
        )

        self.branch3_block_2_1 = nn.Sequential(
            nn.Conv2d(filters_b3_2, filters_b3_3, (1, 3), 1, (0, 1)),
            nn.BatchNorm2d(filters_b3_3),
            nn.ReLU(True)
        )
        
        self.branch3_block_2_2 = nn.Sequential(
            nn.Conv2d(filters_b3_2, filters_b3_4, (3, 1), 1, (1, 0)),
            nn.BatchNorm2d(filters_b3_4),
            nn.ReLU(True)
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            nn.Conv2d(input_feature, filters_b4, 1),
            nn.BatchNorm2d(filters_b4),
            nn.ReLU(True)
        )

    def forward(self, x):
        block1 = self.branch1(x)
        
        block2 = self.branch2_block_1(x)
        block2 = torch.cat([self.branch2_block_2_1(block2), self.branch2_block_2_2(block2)], dim=1)

        block3 = self.branch3_block_1(x)
        block3 = torch.cat([self.branch3_block_2_1(block3), self.branch3_block_2_2(block3)], dim=1)

        block4 = self.branch4(x)

        return torch.cat([block1, block2, block3, block4], dim=1)


class Grid_Reduction_1(nn.Module):
    def __init__(self, input_feature, filters_b1, 
                filters_b2_1, filters_b2_2, filters_b2_3):
        super(Grid_Reduction_1, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b1, 3, 2),
            nn.BatchNorm2d(filters_b1),
            nn.ReLU(True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b2_1, 1),
            nn.BatchNorm2d(filters_b2_1),
            nn.ReLU(True),
            nn.Conv2d(filters_b2_1, filters_b2_2, 3, 1, 1),
            nn.BatchNorm2d(filters_b2_2),
            nn.ReLU(True),
            nn.Conv2d(filters_b2_2, filters_b2_3, 3, 2),
            nn.BatchNorm2d(filters_b2_3),
            nn.ReLU(True)
        )

        self.branch3 = nn.Sequential(
            nn.MaxPool2d(3, 2)
        )
    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)

class Grid_Reduction_2(nn.Module):
    def __init__(self, input_feature, filters_b1_1, filters_b1_2, 
                filters_b2_1, filters_b2_2, filters_b2_3, filters_b2_4):
        super(Grid_Reduction_2, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b1_1, 1),
            nn.BatchNorm2d(filters_b1_1),
            nn.ReLU(True),
            nn.Conv2d(filters_b1_1, filters_b1_2, 3, 2),
            nn.BatchNorm2d(filters_b1_2),
            nn.ReLU(True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b2_1, 1),
            nn.BatchNorm2d(filters_b2_1),
            nn.ReLU(True),
            nn.Conv2d(filters_b2_1, filters_b2_2, (1, 7), 1, (0, 3)),
            nn.BatchNorm2d(filters_b2_2),
            nn.ReLU(True),
            nn.Conv2d(filters_b2_2, filters_b2_3, (7, 1), 1, (3, 0)),
            nn.BatchNorm2d(filters_b2_3),
            nn.ReLU(True),
            nn.Conv2d(filters_b2_3, filters_b2_4, 3, 2),
            nn.BatchNorm2d(filters_b2_4),
            nn.ReLU(True)
        )

        self.branch3 = nn.Sequential(
            nn.MaxPool2d(3, 2)
        )
    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)

class Auxiliary_Classifier(nn.Module):
    def __init__(self, input_feature, num_classes):
        super(Auxiliary_Classifier, self).__init__()
        self.block = nn.Sequential(
            nn.AvgPool2d(5, 3, 1),
            nn.Conv2d(input_feature, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.block(x)

        
class BuildInceptionV3(nn.Module):
    def __init__(self, input_channel=3, num_classes=1000):
        super(BuildInceptionV3, self).__init__()

        self.Stem = nn.Sequential(
            nn.Conv2d(input_channel, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            
            nn.Conv2d(64, 80, 1),
            nn.BatchNorm2d(80),
            nn.ReLU(True),
            nn.Conv2d(80, 192, 3, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2)
        )
        
        self.inception1 = Inception_Module_A(192, 64, 48, 64, 64, 96, 96, 64)
        self.inception2 = Inception_Module_A(288, 64, 48, 64, 64, 96, 96, 64)
        self.inception3 = Inception_Module_A(288, 64, 48, 64, 64, 96, 96, 64)
        self.grid_reduction1 = Grid_Reduction_1(288, 384, 64, 96, 96)    

        self.inception4 = Inception_Module_B(768, 192, 128, 128, 192, 128, 128, 128, 128, 192, 192)
        self.inception5 = Inception_Module_B(768, 192, 160, 160, 192, 160, 160, 160, 160, 192, 192)
        self.inception6 = Inception_Module_B(768, 192, 160, 160, 192, 160, 160, 160, 160, 192, 192)
        self.inception7 = Inception_Module_B(768, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192)
        self.aux = Auxiliary_Classifier(768, num_classes)

        self.grid_reduction2 = Grid_Reduction_2(768, 192, 320, 192, 192, 192, 192)
        self.inception8 = Inception_Module_C(1280, 320, 384, 384, 384, 448, 384, 384, 384, 192)
        self.inception9 = Inception_Module_C(2048, 320, 384, 384, 384, 448, 384, 384, 384, 192)

        self.Classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(2048, num_classes)
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.Stem(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.grid_reduction1(x)
        
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.inception6(x)
        x = self.inception7(x)
        aux = self.aux(x)

        x = self.grid_reduction2(x)
        x = self.inception8(x)
        x = self.inception9(x)
        x = self.Classifier(x)
        return x, aux

model = BuildInceptionV3(input_channel=3, num_classes=5).to(device)
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

            y_pred, aux = model.forward(X)

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
                y_pred, aux = model(X)
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