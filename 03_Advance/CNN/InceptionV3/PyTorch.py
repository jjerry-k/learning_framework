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

class Inception_Module_A(nn.Module):
    def __init__(self, input_feature, 
                filters_b1, filters_b2_1, filters_b2_2, 
                filters_b3_1, filters_b3_2, filters_b3_3, filters_b4):
        super(Inception_Module_A, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b1, 1), 
            nn.ReLU(True))

        self.branch2 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b2_1, 1),
            nn.ReLU(True),
            nn.Conv2d(filters_b2_1, filters_b2_2, 3, 1, 1),
            nn.ReLU(True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b3_1, 1),
            nn.ReLU(True),
            nn.Conv2d(filters_b3_1, filters_b3_2, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(filters_b3_2, filters_b3_3, 3, 1, 1),
            nn.ReLU(True)
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            nn.Conv2d(input_feature, filters_b4, 1),
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
            nn.ReLU(True))

        self.branch2 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b2_1, 1),
            nn.ReLU(True),
            nn.Conv2d(filters_b2_1, filters_b2_2, (7, 1), 1, (3, 0)),
            nn.ReLU(True),
            nn.Conv2d(filters_b2_2, filters_b2_3, (1, 7), 1, (0, 3)),
            nn.ReLU(True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b3_1, 1),
            nn.ReLU(True),
            nn.Conv2d(filters_b3_1, filters_b3_2, (7, 1), 1, (3, 0)),
            nn.ReLU(True),
            nn.Conv2d(filters_b3_2, filters_b3_3, (1, 7), 1, (0, 3)),
            nn.ReLU(True),
            nn.Conv2d(filters_b3_3, filters_b3_4, (7, 1), 1, (3, 0)),
            nn.ReLU(True),
            nn.Conv2d(filters_b3_4, filters_b3_5, (1, 7), 1, (0, 3)),
            nn.ReLU(True)
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            nn.Conv2d(input_feature, filters_b4, 1),
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
            nn.ReLU(True))

        self.branch2_block_1 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b2_1, 1),
            nn.ReLU(True)
        )

        self.branch2_block_2_1 = nn.Sequential(
            nn.Conv2d(filters_b2_1, filters_b2_2, (1, 3), 1, (0, 1)),
            nn.ReLU(True)
        )

        self.branch2_block_2_2 = nn.Sequential(
            nn.Conv2d(filters_b2_1, filters_b2_3, (3, 1), 1, (1, 0)),
            nn.ReLU(True)
        )

        self.branch3_block_1 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b3_1, 1),
            nn.ReLU(True),
            nn.Conv2d(filters_b3_1, filters_b3_2, 3, 1, 1),
            nn.ReLU(True)
        )

        self.branch3_block_2_1 = nn.Sequential(
            nn.Conv2d(filters_b3_2, filters_b3_3, (1, 3), 1, (0, 1)),
            nn.ReLU(True)
        )
        
        self.branch3_block_2_2 = nn.Sequential(
            nn.Conv2d(filters_b3_2, filters_b3_4, (3, 1), 1, (1, 0)),
            nn.ReLU(True)
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, 1, 1),
            nn.Conv2d(input_feature, filters_b4, 1),
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
            nn.ReLU(True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b2_1, 1),
            nn.ReLU(True),
            nn.Conv2d(filters_b2_1, filters_b2_2, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(filters_b2_2, filters_b2_3, 3, 2),
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
            nn.ReLU(True),
            nn.Conv2d(filters_b1_1, filters_b1_2, 3, 2),
            nn.ReLU(True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(input_feature, filters_b2_1, 1),
            nn.ReLU(True),
            nn.Conv2d(filters_b2_1, filters_b2_2, (1, 7), 1, (0, 3)),
            nn.ReLU(True),
            nn.Conv2d(filters_b2_2, filters_b2_3, (7, 1), 1, (3, 0)),
            nn.ReLU(True),
            nn.Conv2d(filters_b2_3, filters_b2_4, 3, 2),
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
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.block(x)

        
class Build_InceptionV3(nn.Module):
    def __init__(self, input_channel=3, num_classes=1000):
        super(Build_InceptionV3, self).__init__()

        self.Stem = nn.Sequential(
            nn.Conv2d(input_channel, 32, 3, 2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            
            nn.Conv2d(64, 80, 1),
            nn.ReLU(True),
            nn.Conv2d(80, 192, 3, 1),
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

        self.init_weights(self.Stem)
        self.init_weights(self.inception1)
        self.init_weights(self.inception3)
        self.init_weights(self.grid_reduction1)
        self.init_weights(self.inception4)
        self.init_weights(self.inception5)
        self.init_weights(self.inception6)
        self.init_weights(self.inception7)
        self.init_weights(self.aux)
        self.init_weights(self.grid_reduction2)
        self.init_weights(self.inception8)
        self.init_weights(self.inception9)
        self.init_weights(self.classifier)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        
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

net = Build_InceptionV3(input_channel=imgs_tr.shape[-1], num_classes=5).to(device)
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