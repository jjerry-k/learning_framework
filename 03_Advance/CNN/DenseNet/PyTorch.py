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
class DenseLayer(nn.Module):
    def __init__(self, input_feature, growth_rate):
        super(DenseLayer, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(input_feature),
            nn.ReLU(True),
            nn.Conv2d(input_feature, growth_rate * 4, 1),
            nn.BatchNorm2d(growth_rate * 4),
            nn.ReLU(True),
            nn.Conv2d(growth_rate * 4, growth_rate, 3, padding=1)
        )

    def forward(self, x):
        new_features = self.block(x)
        return torch.cat([x, new_features], dim=1)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, input_feature, growth_rate):
        super(DenseBlock, self).__init__()

        layer_list = []
        for i in range(num_layers):
            layer_list.append(DenseLayer(input_feature + (i * growth_rate), growth_rate))

        self.block = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.block(x)
            
class Transitionlayer(nn.Module):
    def __init__(self, input_feature, reduction):
        super(Transitionlayer, self).__init__()

        self.block = nn.Sequential(
            nn.BatchNorm2d(input_feature), 
            nn.ReLU(True),
            nn.Conv2d(input_feature, int(input_feature * reduction), kernel_size=1),
            nn.AvgPool2d(2, 2)
        )

    def forward(self, x):
        return self.block(x)

class DenseNet(nn.Module):
    def __init__(self, input_channel=3, num_classes=1000, num_blocks=121, growth_rate=32):
        super(DenseNet, self).__init__()

        blocks_dict = {
        121: [6, 12, 24, 16],
        169: [6, 12, 32, 32], 
        201: [6, 12, 48, 32], 
        264: [6, 12, 64, 48]
    }

        assert num_blocks in  blocks_dict.keys(), "Number of layer must be in %s"%blocks_dict.keys()

        self.Stem = nn.Sequential(
            nn.ZeroPad2d(3),
            nn.Conv2d(input_channel, 64, 7, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ZeroPad2d(1),
            nn.MaxPool2d(3, 2)
        )
        
        layer_list = []
        num_features = 64
        
        for idx, layers in enumerate(blocks_dict[num_blocks]):
            layer_list.append(DenseBlock(layers, num_features, growth_rate))
            num_features = num_features + (layers * growth_rate)
            if idx != 3:
                layer_list.append(Transitionlayer(num_features, 0.5))
                num_features = int(num_features * 0.5)

        self.Main_Block = nn.Sequential(*layer_list)

        self.Classifier = nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(num_features, num_classes)
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

model = DenseNet(input_channel=3, num_classes=5, num_blocks=121, growth_rate=32).to(device)
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