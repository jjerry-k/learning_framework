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
    def __init__(self, in_channel, output_channel, ksize, stride, padding, activation=True, use_bn=False):
        super(Conv_Block, self).__init__()

        layer_list = []

        layer_list.append(nn.Conv2d(in_channel, output_channel, ksize, stride, padding))
        
        if use_bn:
            layer_list.append(nn.BatchNorm2d(output_channel))
        
        if activation:
            layer_list.append(nn.ReLU(True))

        self.block = nn.Sequential(*layer_list)
    
    def forward(self, x):
        return self.block(x)

class Squeeze_Excitation_Module(nn.Module):
    def __init__(self, features, reduction_ratio):
        super(Squeeze_Excitation_Module, self).__init__()

        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(features, features//reduction_ratio, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(features//reduction_ratio, features, 1, 1),
            nn.Sigmoid()   
        )

    def forward(self, x):
        return x * self.block(x)

class SE_Residual_Block(nn.Module):
    def __init__(self, in_channel, output_channel, stride, reduction_ratio, use_bn=False, 
                use_proj=False, proj_ksize=3, proj_pad=1):
        super(SE_Residual_Block, self).__init__()
        
        self.use_proj = use_proj

        layer_list = []

        layer_list.append(Conv_Block(in_channel, output_channel//4, 1, 1, 0, True, use_bn))
        layer_list.append(Conv_Block(output_channel//4, output_channel//4, 3, stride, 1, True, use_bn))
        layer_list.append(Conv_Block(output_channel//4, output_channel, 1, 1, 0, False, use_bn))

        layer_list.append(Squeeze_Excitation_Module(output_channel, reduction_ratio))

        self.block = nn.Sequential(*layer_list)

        self.proj = Conv_Block(in_channel, output_channel, proj_ksize, stride, proj_pad, False, use_bn)

        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.block(x)
    
        proj = self.proj(x) if self.use_proj else x

        return self.relu(out + proj)

class build_seresnet(nn.Module):
    def __init__(self, input_channel= 3, num_classes=1000):
        super(build_seresnet, self).__init__()

        layer_list = []

        layer_list.append(Conv_Block(input_channel, 64, 3, 2, 0, True, True))
        layer_list.append(Conv_Block(64, 64, 3, 1, 1, True, True))
        layer_list.append(Conv_Block(64, 128, 3, 1, 1, True, True))
        layer_list.append(nn.MaxPool2d(3, 2))

        layer_list.append(SE_Residual_Block(128, 256, 1, 16, True, True, 1, 0))
        for _ in range(2):
            layer_list.append(SE_Residual_Block(256, 256, 1, 16, True))

        layer_list.append(SE_Residual_Block(256, 512, 2, 16, True, True, 3, 1))
        for _ in range(3):
            layer_list.append(SE_Residual_Block(512, 512, 1, 16, True))
    
        layer_list.append(SE_Residual_Block(512, 1024, 2, 16, True, True, 3, 1))
        for _ in range(5):
            layer_list.append(SE_Residual_Block(1024, 1024, 1, 16, True))
        
        layer_list.append(SE_Residual_Block(1024, 2048, 2, 16, True, True, 3, 1))
        for _ in range(2):
            layer_list.append(SE_Residual_Block(2048, 2048, 1, 16, True))

        self.Stem = nn.Sequential(*layer_list)

        self.Classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(2048, num_classes)
        )

        self.init_weights(self.Stem)
        self.init_weights(self.Classifier)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.Stem(x)
        x = self.Classifier(x)
        return x

net = build_seresnet(input_channel=imgs_tr.shape[-1], num_classes=5).to(device)
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