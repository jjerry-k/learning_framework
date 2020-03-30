# %%
import os, torch
import cv2 as cv
import numpy as np
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
img_size = 150

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

class Hard_Sigmoid(nn.Module):
    def __init__(self, inplace=False):
        super(Hard_Sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return torch.max(torch.zeros_like(x), torch.min(torch.ones_like(x), x * 0.2 + 0.5))

class Hard_Swish(nn.Module):
    def __init__(self, inplace=False):
        super(Hard_Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

class Conv_Block(nn.Module):
    def __init__(self, input_feature, output_feature, ksize=3, strides=1, padding=1, use_hs=True):
        super(Conv_Block, self).__init__()
        Act = Hard_Swish if use_hs else nn.ReLU6
        self.block = nn.Sequential(
            nn.Conv2d(input_feature, output_feature, ksize, strides, padding),
            nn.BatchNorm2d(output_feature),
            Act(True)
            )

    def forward(self, x):
        return self.block(x)

class Depthwise_Separable_Block(nn.Module):
    def __init__(self, input_feature, output_feature, ksize=3, strides=1, padding=1, alpha=1, use_se=True, use_hs=True):
        super(Depthwise_Separable_Block, self).__init__()
        
        self.use_se = use_se

        Act = Hard_Swish if use_hs else nn.ReLU6

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
            
class Inverted_Residual_Block(nn.Module):
    def __init__(self, input_feature, expansion, output_feature, strides=1, alpha=1, use_se=True, use_hs=True):
        super(Inverted_Residual_Block, self).__init__()
        
        self.stride = strides

        self.intermediate_featrue = int(input_feature*expansion)
        
        self.output_feature = output_feature

        self.alpha = alpha
        
        self.block = nn.Sequential(
            Conv_Block(input_feature, self.intermediate_featrue, 1, 1, 0, use_hs),
            Depthwise_Separable_Block(self.intermediate_featrue, self.output_feature, 3, strides, 1, self.alpha, use_se, use_hs)
        )
    
    def forward(self, x):
        output = self.block(x)
        if self.stride==1 and self.intermediate_featrue == int(self.output_feature*self.alpha):
            return x + output
        return output

class Build_MobileNetV3(nn.Sequential):
    def __init__(self, input_channel=3, num_classes=1000, alpha=1):
        super(Build_MobileNetV3, self).__init__()

        self.Stem = Conv_Block(input_channel, 16, 3, 2, 1)

        layer_list = []

        layer_list.append(Inverted_Residual_Block(16, 1, 16, 1, 1, use_se=False, use_hs=False))

        layer_list.append(Inverted_Residual_Block(16, 4, 24, 2, 1, use_se=False, use_hs=False))
        layer_list.append(Inverted_Residual_Block(24, 3, 24, 1, 1, use_se=False, use_hs=False))

        layer_list.append(Inverted_Residual_Block(24, 3, 40, 2, 1, use_hs=False))
        layer_list.append(Inverted_Residual_Block(40, 3, 40, 1, 1, use_hs=False))
        layer_list.append(Inverted_Residual_Block(40, 3, 40, 1, 1, use_hs=False))

        layer_list.append(Inverted_Residual_Block(40, 6, 80, 2, 1, use_se=False))
        layer_list.append(Inverted_Residual_Block(80, 2.5, 80, 1, 1, use_se=False))
        layer_list.append(Inverted_Residual_Block(80, 2.3, 80, 1, 1, use_se=False))
        layer_list.append(Inverted_Residual_Block(80, 2.3, 80, 1, 1, use_se=False))
        layer_list.append(Inverted_Residual_Block(80, 6, 112, 1, 1))
        layer_list.append(Inverted_Residual_Block(112, 6, 112, 1, 1))
        
        layer_list.append(Inverted_Residual_Block(112, 6, 160, 2, 1))
        layer_list.append(Inverted_Residual_Block(160, 6, 160, 1, 1))
        layer_list.append(Inverted_Residual_Block(160, 6, 160, 1, 1))

        layer_list.append(Conv_Block(160, 960, 1, 1, 0))

        self.Main_Block = nn.Sequential(*layer_list)

        self.Classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(960, 1280),
            Hard_Swish(True),
            nn.Linear(1280, num_classes)
        )
        
    def forward(self, x):
        x = self.Stem(x)
        x = self.Main_Block(x)
        x = self.Classifier(x)
        return x

mobilenetv3 = Build_MobileNetV3(input_channel=imgs_tr.shape[-1], num_classes=5, alpha=1).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mobilenetv3.parameters(), lr=0.001)

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

        y_pred = mobilenetv3.forward(X)

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
            y_pred = mobilenetv3(X)
            val_loss += criterion(y_pred, Y)
            _, predicted = torch.max(y_pred.data, 1)
            total += Y.size(0)
            correct += (predicted == Y).sum().item()
        val_loss /= total
        val_acc = (100 * correct / total)

    print("Epoch : ", epoch+1, " Loss : ", (avg_loss/total_batch), " Val Loss : ", val_loss.item(), "Val Acc : ", val_acc)

print("Training Done !")
