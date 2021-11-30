import os
import time
import cv2 as cv
import numpy as np
from tqdm import tqdm
import torch
from torchvision import transforms, datasets, utils
from torch.utils.data import Dataset, DataLoader 

PATH = "../data/flower_photos"

category_list = [i for i in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, i)) ]
print(category_list)

num_classes = len(category_list)
img_size = 128
batch_size = 32

class CustomDataset(Dataset):
    def __init__(self, train_x, train_y): 
        self.len = len(train_x) 
        self.x_data = torch.tensor(np.transpose(train_x, [0, 3, 1, 2]), dtype=torch.float)
        self.y_data = torch.tensor(train_y, dtype=torch.long) 

    def __getitem__(self, index): 
        return self.x_data[index], self.y_data[index] 

    def __len__(self): 
        return self.len

batch_size = 32

train_dataset = CustomDataset()
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

validation_dataset = CustomDataset()
validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

with tqdm(total=len(train_loader)) as t:
    t.set_description(f'Train Loader')
    for i, (batch_img, batch_lab) in enumerate(train_loader):
        time.sleep(0.1)
        t.set_postfix({"Train data shape": f"{batch_img.shape} {batch_lab.shape}"})
        t.update()

with tqdm(total=len(validation_loader)) as t:
    t.set_description(f'Validation Loader')
    for i, (batch_img, batch_lab) in enumerate(validation_loader):
        time.sleep(0.1)
        t.set_postfix({"Validation data shape": f"{batch_img.shape} {batch_lab.shape}"})
        t.update()