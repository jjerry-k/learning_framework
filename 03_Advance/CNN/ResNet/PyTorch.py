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

class Residual_block(nn.Module):
    def __init__(self, in_channel, output_channel, strides=1, use_branch=True):
        super(Residual_block, self).__init__()

        self.branch1 = lambda x: x
        if use_branch:
            self.branch1 = nn.Conv2d(in_channel, output_channel, 1, strides)
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, output_channel//4, 1, strides),
            nn.BatchNorm2d(output_channel//4),
            nn.ReLU(True),
            nn.Conv2d(output_channel//4, output_channel//4, 3, 1, padding=1),
            nn.BatchNorm2d(output_channel//4),
            nn.ReLU(True),
            nn.Conv2d(output_channel//4, output_channel, 1, 1),
            nn.BatchNorm2d(output_channel),        
        )

        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.branch2(x)
        out = self.relu(out + self.branch1(x))

        return out

class build_resnet(nn.Module):
    def __init__(self, input_channel= 3, num_classes=1000, num_layer=16):
        super(build_resnet, self).__init__()

        blocks_dict = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3], 
        152: [3, 8, 36, 3]
        }

        num_channel_list = [256, 512, 1024, 2048]

        assert num_layer in  blocks_dict.keys(), "Number of layer must be in %s"%blocks_dict.keys()

        self.stem = nn.Sequential(
            nn.ZeroPad2d((3,3)),
            nn.Conv2d(input_channel, 64, 7, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )

        layer_list = []

        input_features = 64

        for idx, num_iter in enumerate(blocks_dict[num_layer]):
            for j in range(num_iter):
                if j==0:
                    layer_list.append(Residual_block(input_features, num_channel_list[idx], strides=2))
                else:
                    layer_list.append(Residual_block(input_features, num_channel_list[idx], use_branch=False))
                input_features = num_channel_list[idx]
        self.main_net = nn.Sequential(*layer_list)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(input_features, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.main_net(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

resnet = build_resnet(input_channel=imgs_tr.shape[-1], num_classes=5, num_layer=50).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.001)

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

        y_pred = resnet.forward(X)

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
            y_pred = resnet(X)
            val_loss += criterion(y_pred, Y)
            _, predicted = torch.max(y_pred.data, 1)
            total += Y.size(0)
            correct += (predicted == Y).sum().item()
        val_loss /= total
        val_acc = (100 * correct / total)

    print("Epoch : ", epoch+1, " Loss : ", (avg_loss/total_batch).item(), " Val Loss : ", val_loss, "Val Acc : ", val_acc)

print("Training Done !")