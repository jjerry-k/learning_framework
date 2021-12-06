import os
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models import *

PATH = "../../datasets/facades"
IMG_FORMAT = ["jpg", "jpeg", "tif", "tiff", "bmp", "png"]

img_size = 256
batch_size = 32

transform = transforms.Compose([
                                transforms.Resize([img_size, img_size]), 
                                transforms.ToTensor()
                                ])

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform):
        
        self.filelist = []
        self.classes = sorted(os.listdir(data_dir))
        for root, _, files in os.walk(data_dir):
            if not len(files): continue
            files = [os.path.join(root, file) for file in files if file.split(".")[-1].lower() in IMG_FORMAT]
            self.filelist += files
        self.transform = transform

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):

        image = Image.open(self.filelist[idx])
        min_side = min(image.size)
        max_side = max(image.size)
        dom_a = image.crop((0, 0, min_side, min_side))
        dom_b = image.crop((min_side, 0, max_side, min_side))
        dom_a = self.transform(dom_a)
        dom_b = self.transform(dom_b)
        return dom_a, dom_b

dataset = CustomDataset(PATH, transform)

loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
