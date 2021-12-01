import os
import time
import cv2 as cv
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torchvision import transforms, datasets, utils
from torch.utils.data import Dataset, DataLoader 

PATH = "../data/flower_photos"
IMG_FORMAT = ["jpg", "jpeg", "tif", "tiff", "bmp", "png"]

category_list = [i for i in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, i)) ]
print(category_list)

num_classes = len(category_list)
img_size = 128
batch_size = 32

transform = transforms.Compose([
                                transforms.Resize([img_size, img_size]), 
                                transforms.ToTensor()
                                ])

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform):
        
        self.filelist = []
        self.classes = sorted(os.listdir(data_dir))
        for root, sub_dir, files in os.walk(data_dir):
            if not len(files): continue
            files = [os.path.join(root, file) for file in files if file.split(".")[-1].lower() in IMG_FORMAT]
            self.filelist += files
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.filelist)

    def __getitem__(self, idx):

        image = Image.open(self.filelist[idx])
        image = self.transform(image)
        label = self.filelist[idx].split('/')[-2]
        label = self.classes.index(label)
        return image, label

train_dataset = CustomDataset(os.path.join(PATH, "train"), transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

validation_dataset = CustomDataset(os.path.join(PATH, "validation"), transform)
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