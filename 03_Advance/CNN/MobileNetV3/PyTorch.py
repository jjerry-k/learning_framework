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
batch_size= 8
img_size= 128

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
class Hard_Sigmoid(nn.Module):
    def __init__(self, inplace=False):
        super(Hard_Sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return torch.max(torch.zeros_like(x), torch.min(torch.ones_like(x), x * 0.2 + 0.5))

class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace
        self.relu6 = nn.ReLU6(inplace)
    def forward(self, x):
        return x * self.relu6(x + 3.) / 6.

class ConvBlock(nn.Module):
    def __init__(self, input_feature, output_feature, ksize=3, strides=1, padding=1, use_hs=True):
        super(ConvBlock, self).__init__()
        Act = HardSwish if use_hs else nn.ReLU6
        self.block = nn.Sequential(
            nn.Conv2d(input_feature, output_feature, ksize, strides, padding),
            nn.BatchNorm2d(output_feature),
            Act(True)
            )

    def forward(self, x):
        return self.block(x)

class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, input_feature, output_feature, ksize=3, strides=1, padding=1, alpha=1, use_se=True, use_hs=True):
        super(DepthwiseSeparableBlock, self).__init__()
        
        self.use_se = use_se

        Act = HardSwish if use_hs else nn.ReLU6

        self.depthwise = nn.Sequential(
            nn.Conv2d(input_feature, input_feature, ksize, strides, padding, groups=input_feature),
            nn.BatchNorm2d(input_feature),
            Act(True)
        )

        self.pointhwise = nn.Sequential(
            nn.Conv2d(input_feature, int(output_feature*alpha), 1),
            nn.BatchNorm2d(int(output_feature*alpha)),
            Act(True)
        )

        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(input_feature, input_feature, 1, 1),
                nn.ReLU(True),
                nn.Conv2d(input_feature, input_feature, 1, 1),
                Hard_Sigmoid(True)
            )

    def forward(self, x):
        out = self.depthwise(x)
        if self.use_se:
            out = out * self.se(out)
        out = self.pointhwise(x)
        return out
            
class InvertedResidualBlock(nn.Module):
    def __init__(self, input_feature, expansion, output_feature, strides=1, alpha=1, use_se=True, use_hs=True):
        super(InvertedResidualBlock, self).__init__()
        
        self.stride = strides

        self.intermediate_featrue = int(input_feature*expansion)
        
        self.output_feature = output_feature

        self.alpha = alpha
        
        self.block = nn.Sequential(
            ConvBlock(input_feature, self.intermediate_featrue, 1, 1, 0, use_hs),
            DepthwiseSeparableBlock(self.intermediate_featrue, self.output_feature, 3, strides, 1, self.alpha, use_se, use_hs)
        )
    
    def forward(self, x):
        output = self.block(x)
        if self.stride==1 and self.intermediate_featrue == int(self.output_feature*self.alpha):
            return x + output
        return output

class MobileNetV3(nn.Sequential):
    def __init__(self, input_channel=3, num_classes=1000, alpha=1):
        super(MobileNetV3, self).__init__()

        self.Stem = ConvBlock(input_channel, 16, 3, 2, 1)

        layer_list = []

        layer_list.append(InvertedResidualBlock(16, 1, 16, 1, alpha, use_se=False, use_hs=False))

        layer_list.append(InvertedResidualBlock(16, 4, 24, 2, alpha, use_se=False, use_hs=False))
        layer_list.append(InvertedResidualBlock(24, 3, 24, 1, alpha, use_se=False, use_hs=False))

        layer_list.append(InvertedResidualBlock(24, 3, 40, 2, alpha, use_hs=False))
        layer_list.append(InvertedResidualBlock(40, 3, 40, 1, alpha, use_hs=False))
        layer_list.append(InvertedResidualBlock(40, 3, 40, 1, alpha, use_hs=False))

        layer_list.append(InvertedResidualBlock(40, 6, 80, 2, alpha, use_se=False))
        layer_list.append(InvertedResidualBlock(80, 2.5, 80, 1, alpha, use_se=False))
        layer_list.append(InvertedResidualBlock(80, 2.3, 80, 1, alpha, use_se=False))
        layer_list.append(InvertedResidualBlock(80, 2.3, 80, 1, alpha, use_se=False))
        layer_list.append(InvertedResidualBlock(80, 6, 112, 1, alpha))
        layer_list.append(InvertedResidualBlock(112, 6, 112, 1, alpha))
        
        layer_list.append(InvertedResidualBlock(112, 6, 160, 2, alpha))
        layer_list.append(InvertedResidualBlock(160, 6, 160, 1, alpha))
        layer_list.append(InvertedResidualBlock(160, 6, 160, 1, alpha))

        layer_list.append(ConvBlock(160, 960, 1, 1, 0))

        self.Main_Block = nn.Sequential(*layer_list)

        self.Classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(960, 1280),
            HardSwish(True),
            nn.Linear(1280, num_classes)
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

model = MobileNetV3(input_channel=3, num_classes=5, alpha=1).to(device)
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