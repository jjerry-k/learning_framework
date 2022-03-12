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
epochs= 10
batch_size= 16
img_size= 224

# Dataset
class HorseDataset(Dataset):
    def __init__(self, data_dir, transform):
        IMG_FORMAT = ["jpg", "jpeg", "bmp", "png", "tif", "tiff"]
        self.img_root = os.path.join(data_dir, "jpg")
        self.lab_root = os.path.join(data_dir, "gt")

        self.img_list = [file for file in sorted(os.listdir(self.img_root)) if file.split(".")[-1] in IMG_FORMAT]
        self.lab_list = [file for file in sorted(os.listdir(self.lab_root)) if file.split(".")[-1] in IMG_FORMAT]
        
        for img, lab in zip(self.img_list, self.lab_list):
            img_filename = img.split(".")[0]
            lab_filename = lab.split(".")[0]
            if img_filename != lab_filename: 
                raise RuntimeError(f"File name NOT same {img} {lab}")

        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.img_list)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.img_root, self.img_list[idx])).convert("RGB")
        image = self.transform(image)
        label = Image.open(os.path.join(self.lab_root, self.lab_list[idx])).convert("L")
        label = self.transform(label).squeeze()
        return image, label

transform = transforms.Compose([
                                transforms.Resize((img_size, img_size)), transforms.ToTensor()
                                ])
train_dataset = HorseDataset(os.path.join("../../../data/horses/train"), transform)
val_dataset = HorseDataset(os.path.join("../../../data/horses/validation"), transform)

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

class Upconv_Block(nn.Module):
    def __init__(self, input_feature, output_feature, ksize=3, strides=1, padding=1):
        super(Upconv_Block, self).__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(input_feature, output_feature, ksize, strides, padding),
            nn.BatchNorm2d(output_feature),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class BuildDeconvNet(nn.Module):
    """
    Input size : 224 x 224
    """
    def __init__(self, input_channel= 3, num_classes=1000):
        super(BuildDeconvNet, self).__init__()

        self.en_block_1 = nn.Sequential(
            Conv_Block(input_channel, 64, 3, 1, 1),
            Conv_Block(64, 64, 3, 1, 1)
        )

        self.en_block_2 = nn.Sequential(
            Conv_Block(64, 128, 3, 1, 1),
            Conv_Block(128, 128, 3, 1, 1)
        )

        self.en_block_3 = nn.Sequential(
            Conv_Block(128, 256, 3, 1, 1),
            Conv_Block(256, 256, 3, 1, 1),
            Conv_Block(256, 256, 3, 1, 1)
        )

        self.en_block_4 = nn.Sequential(
            Conv_Block(256, 512, 3, 1, 1),
            Conv_Block(512, 512, 3, 1, 1),
            Conv_Block(512, 512, 3, 1, 1)
        )

        self.en_block_5 = nn.Sequential(
            Conv_Block(512, 512, 3, 1, 1),
            Conv_Block(512, 512, 3, 1, 1),
            Conv_Block(512, 512, 3, 1, 1)
        )

        self.en_block_5 = nn.Sequential(
            Conv_Block(512, 512, 3, 1, 1),
            Conv_Block(512, 512, 3, 1, 1),
            Conv_Block(512, 512, 3, 1, 1)
        )

        self.fc_block = nn.Sequential(
            Conv_Block(512, 4096, 7, 1, 0),
            Conv_Block(4096, 4096, 1, 1, 0),
            Upconv_Block(4096, 512, 7, 1, 0)
        )

        self.de_block_5 = nn.Sequential(
            Upconv_Block(512, 512, 3, 1, 1),
            Upconv_Block(512, 512, 3, 1, 1),
            Upconv_Block(512, 512, 3, 1, 1)
        )

        self.de_block_4 = nn.Sequential(
            Upconv_Block(512, 512, 3, 1, 1),
            Upconv_Block(512, 512, 3, 1, 1),
            Upconv_Block(512, 256, 3, 1, 1)
        )

        self.de_block_3 = nn.Sequential(
            Upconv_Block(256, 256, 3, 1, 1),
            Upconv_Block(256, 256, 3, 1, 1),
            Upconv_Block(256, 128, 3, 1, 1)
        )

        self.de_block_2 = nn.Sequential(
            Upconv_Block(128, 128, 3, 1, 1),
            Upconv_Block(128, 64, 3, 1, 1)
        )

        self.de_block_1 = nn.Sequential(
            Upconv_Block(64, 64, 3, 1, 1),
            Upconv_Block(64, 64, 3, 1, 1)
        )

        self.classification = nn.Conv2d(64, num_classes, 1)

        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

        self.unpool = nn.MaxUnpool2d(2, 2)

    def forward(self, x):

        x = self.en_block_1(x)
        x, idx_1 = self.pool(x)
        x = self.en_block_2(x)
        x, idx_2 = self.pool(x)
        x = self.en_block_3(x)
        x, idx_3 = self.pool(x)
        x = self.en_block_4(x)
        x, idx_4 = self.pool(x)
        x = self.en_block_5(x)
        x, idx_5 = self.pool(x)
        x = self.fc_block(x)
        x = self.unpool(x, idx_5)
        x = self.de_block_5(x)
        x = self.unpool(x, idx_4)
        x = self.de_block_4(x)
        x = self.unpool(x, idx_3)
        x = self.de_block_3(x)
        x = self.unpool(x, idx_2)
        x = self.de_block_2(x)
        x = self.unpool(x, idx_1)
        x = self.de_block_1(x)
        x = self.classification(x)

        return x

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss()

    def forward(self, inputs, targets):
        return self.nll_loss(nn.functional.log_softmax(inputs, dim=1), targets)

model = BuildDeconvNet(input_channel=3, num_classes=2).to(device)
criterion = CrossEntropyLoss2d()
optimizer = optim.Adam(model.parameters(), lr=0.001)  

# Training

def calc_iou(pred, label):
    # pred: [B, H, W]
    # label: [B, H, W]

    intersection = (pred * label).sum(dim=(1, 2))
    union = (pred + label).sum(dim=(1, 2)) - intersection
    iou = (intersection + 1e-5)/(union + 1e-5)
    return iou

for epoch in range(epochs):
    model.train()
    avg_loss = 0
    avg_iou = 0

    with tqdm(total=len(train_loader)) as t:
        t.set_description(f'[{epoch+1}/{epochs}]')
        total = 0
        iou = 0
        for i, (batch_img, batch_lab) in enumerate(train_loader):
            X = batch_img.to(device)
            Y = batch_lab.type(torch.LongTensor).to(device)
            y_pred = model.forward(X)
            loss = criterion(y_pred, Y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            
            predicted = torch.argmax(y_pred, dim=1)
            total += Y.size(0)
            iou += calc_iou(predicted, Y).mean().item()

            t.set_postfix({"loss": f"{avg_loss/(i+1):05.3f}"})
            t.update()
        avg_iou += iou

    model.eval()
    with tqdm(total=len(val_loader)) as t:
        t.set_description(f'[{epoch+1}/{epochs}]')
        with torch.no_grad():
            val_loss = 0
            val_iou = 0
            total = 0
            for i, (batch_img, batch_lab) in enumerate(val_loader):
                X = batch_img.to(device)
                Y = batch_lab.type(torch.LongTensor).to(device)
                y_pred = model(X)
                val_loss += criterion(y_pred, Y)
                predicted = torch.argmax(y_pred, dim=1)
                total += Y.size(0)
                val_iou += calc_iou(predicted, Y).mean()
                t.set_postfix({"val_loss": f"{val_loss.item()/(i+1):05.3f}"})
                t.update()
            val_loss /= len(val_loader)
            val_iou /= len(val_loader)
            
    print(f"Epoch : {epoch+1}, Loss : {(avg_loss/len(train_loader)):.3f}, IoU: {avg_iou/len(train_loader):.3f}, Val Loss : {val_loss.item():.3f}, Val IoU : {val_iou.item():.3f}\n")

print("Training Done !")