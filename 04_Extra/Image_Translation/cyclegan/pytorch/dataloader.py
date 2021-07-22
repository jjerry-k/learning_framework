import random
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class BaseDataset(Dataset):

    def __init__(self, data_dir, transform, type):
        self.data_dir = data_dir

        self.A_path = os.path.join(self.data_dir, f"{type}A")
        self.A_list = os.listdir(self.A_path)

        self.B_path = os.path.join(self.data_dir, f"{type}B")
        self.B_list = os.listdir(self.B_path)

        self.transform = transform

    def __len__(self):
        return max(len(self.A_list), len(self.B_list))

    def __getitem__(self, idx):
        A_idx = random.randint(0, len(self.A_list)-1)
        B_idx = random.randint(0, len(self.B_list)-1)

        A_image = Image.open(os.path.join(self.A_path, self.A_list[A_idx]))
        if len(A_image.getbands()) > 3:
            A_image = A_image.convert("RGB")

        B_image = Image.open(os.path.join(self.B_path, self.B_list[B_idx]))
        if len(B_image.getbands()) > 3:
            B_image = B_image.convert("RGB")
        
        if self.transform:
            A_image = self.transform(A_image)
            B_image = self.transform(B_image)

        return A_image, B_image

def CustomDataloader(transform=None, **kwargs):
    input_size = (kwargs['input_size'], kwargs['input_size'])
    dataloaders = {}
    for split in ['train', 'test']:
        transform_list = [transforms.Resize(input_size), transforms.ToTensor()]
        dl = DataLoader(BaseDataset(kwargs['path'], transforms.Compose(transform_list), split), batch_size=kwargs['batch_size'], num_workers=kwargs['num_workers'], drop_last=True)
        dataloaders[split] = dl
    return dataloaders