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
    '''(Conv, ReLU) * 2'''
    def __init__(self, in_ch, out_ch, pool=None):
        super(Conv_Block, self).__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(out_ch, out_ch, 3, padding=1),
                  nn.ReLU(inplace=True)]
        
        if pool:
            layers.insert(0, nn.MaxPool2d(2, 2))
        
        self.conv = nn.Sequential(*layers)
            

    def forward(self, x):
        x = self.conv(x)
        return x


class Upconv_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Upconv_Block, self).__init__()

        self.upconv = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        
        self.conv = Conv_Block(in_ch, out_ch)

    def forward(self, x1, x2):
        # x1 : unpooled feature
        # x2 : encoder feature
        x1 = self.upconv(x1)
        x1 = nn.UpsamplingBilinear2d(x2.size()[2:])(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class Build_UNet(nn.Module):
    def __init__(self, input_channel=3, num_classes=5):
        super(Build_UNet, self).__init__()
        self.conv1 = Conv_Block(input_channel, 64)
        self.conv2 = Conv_Block(64, 128, pool=True)
        self.conv3 = Conv_Block(128, 256, pool=True)
        self.conv4 = Conv_Block(256, 512, pool=True)
        self.conv5 = Conv_Block(512, 1024, pool=True)
        
        self.unconv4 = Upconv_Block(1024, 512)
        self.unconv3 = Upconv_Block(512, 256)
        self.unconv2 = Upconv_Block(256, 128)
        self.unconv1 = Upconv_Block(128, 64)
        
        self.prediction = nn.Conv2d(64, num_classes, 1)
        
    def forward(self, x):
        en1 = self.conv1(x) #/2
        en2 = self.conv2(en1) #/4
        en3 = self.conv3(en2) #/8
        en4 = self.conv4(en3) #/16
        en5 = self.conv5(en4) 
        
        de4 = self.unconv4(en5, en4) # /8
        de3 = self.unconv3(de4, en3) # /4
        de2 = self.unconv2(de3, en2) # /2
        de1 = self.unconv1(de2, en1) # /1
        
        output = self.prediction(de1)
        return output

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss()

    def forward(self, inputs, targets):
        return self.nll_loss(nn.functional.log_softmax(inputs, dim=1), targets)

num_classes = 2
unet = Build_UNet(input_channel=imgs_tr.shape[-1], num_classes=num_classes).to(device)
criterion = CrossEntropyLoss2d()
optimizer = optim.Adam(unet.parameters(), lr=0.001)    

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

        y_pred = unet.forward(X)

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
            y_pred = unet(X)
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