# %%
import os, torch
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms, datasets, utils

# Device Configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
SAVE_PATH = "../../../data"
URL = 'https://www.robots.ox.ac.uk/~vgg/data/bicos/data/horses.tar'

file_name = URL.split("/")[-1]
data = datasets.utils.download_and_extract_archive(URL, SAVE_PATH)

PATH = os.path.join(SAVE_PATH, 'horses')

PATH_img = os.path.join(PATH, 'jpg')
PATH_lab = os.path.join(PATH, 'gt')

img_list = sorted(os.listdir(PATH_img))
lab_list = sorted(os.listdir(PATH_lab))


img_size = 224

def read_img(path, img_size, mode='rgb'):
    mode_dict = {"rgb":cv.COLOR_BGR2RGB, 
            "gray":cv.COLOR_BGR2GRAY}
    
    img = cv.imread(path)
    img = cv.cvtColor(img, mode_dict[mode])
    img = cv.resize(img, (img_size, img_size))
    return img

print("Total images : %d"%(len(img_list)))
print("Total labels : %d"%(len(lab_list)))

imgs = np.array([read_img(os.path.join(PATH_img, i), img_size, 'rgb') for i in img_list])/255.
labs = np.greater(np.array([read_img(os.path.join(PATH_lab, i), img_size, 'gray') for i in lab_list])/255., 0.5)

ratio = int(len(img_list)*0.05)

imgs_tr = imgs[ratio:]
labs_tr = labs[ratio:]

imgs_val = imgs[:ratio]
labs_val = labs[:ratio]

print("Training images : %d"%(len(imgs_tr)))
print("Training labels : %d"%(len(labs_tr)))

print("Validation images : %d"%(len(imgs_val)))
print("Validation labels : %d"%(len(labs_val)))

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

class Build_DeconvNet(nn.Module):
    """
    Input size : 224 x 224
    """
    def __init__(self, input_channel= 3, num_classes=1000):
        super(Build_DeconvNet, self).__init__()

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

num_classes = 2
deconvnet = Build_DeconvNet(input_channel=imgs_tr.shape[-1], num_classes=num_classes).to(device)
criterion = CrossEntropyLoss2d()
optimizer = optim.Adam(deconvnet.parameters(), lr=0.001)

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
    avg_loss = 0
    avg_acc = 0
    total_batch = train_dataset.len // batch_size
    for i, (batch_img, batch_lab) in enumerate(train_loader):
        X = batch_img.to(device)
        Y = batch_lab.to(device)

        optimizer.zero_grad()

        y_pred = deconvnet.forward(X)

        loss = criterion(y_pred, Y)
        
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()

        if (i+1)%20 == 0 :
            print("Epoch : ", epoch+1, "Iteration : ", i+1, " Loss : ", loss.item())

    with torch.no_grad():
        val_loss = 0
        total = 0
        correct = 0
        for (batch_img, batch_lab) in val_loader:
            X = batch_img.to(device)
            Y = batch_lab.to(device)
            y_pred = deconvnet(X)
            val_loss += criterion(y_pred, Y)
            _, predicted = torch.max(y_pred.data, 1)
            total += Y.size(0)
        val_loss /= total

    print("Epoch : ", epoch+1, " Loss : ", (avg_loss/total_batch), " Val Loss : ", val_loss.item())
    num_plot=4
    shuffle_idx = np.random.choice(val_dataset.len, num_plot, replace=False)
    In = X.cpu().numpy()[shuffle_idx].transpose(0, 2, 3, 1)
    predicted = predicted.cpu().numpy()[shuffle_idx]
    plt.figure(figsize=(10, 4))
    for i in range(num_plot):
        plt.subplot(2, num_plot, i+1)
        plt.imshow(In[i])
        plt.axis("off")
        plt.subplot(2, num_plot, i+1+num_plot)
        plt.imshow(predicted[i], cmap='gray')
    plt.show()

print("Training Done !")