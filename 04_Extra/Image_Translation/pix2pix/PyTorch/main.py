import os
import time
import torch
from tqdm import tqdm
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

PATH = "../../datasets/night2day"
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

generator = Generator_Encoder_Decoder(A_channel=3, B_channel=3, num_features=64).to(device)
discriminator = Discriminator(A_channel=3, B_channel=3, num_features=64, n_layers=1).to(device)

gan_loss = nn.BCELoss()
l1_loss = nn.L1Loss()
l1_lambda = 10
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)

with tqdm(total=len(loader)) as t:
    t.set_description(f'Loader')
    for i, (batch_img_a, batch_img_b) in enumerate(loader):
        # time.sleep(0.1)
        batch_img_a = batch_img_a.to(device)
        batch_img_b = batch_img_b.to(device)

        gen_b = generator(batch_img_a)
        dis_pred_real = discriminator(torch.cat([batch_img_a, batch_img_b], dim=1))
        dis_pred_fake = discriminator(torch.cat([batch_img_a, gen_b], dim=1).detach())

        # Training Discriminator
        real_lab = torch.ones_like(dis_pred_real).to(device)
        fake_lab = torch.zeros_like(dis_pred_fake).to(device)

        dis_loss_real = gan_loss(dis_pred_real, real_lab)
        dis_loss_fake = gan_loss(dis_pred_fake, fake_lab)

        dis_loss = dis_loss_real + dis_loss_fake
        d_optimizer.zero_grad()
        dis_loss.backward()
        d_optimizer.step()
        
        # Training Generator
        gen_l1_loss = l1_loss(gen_b, batch_img_b)
        dis_pred_fake = discriminator(torch.cat([batch_img_a, gen_b], dim=1))
        dis_loss_real = gan_loss(dis_pred_fake, real_lab)
        gen_loss = dis_loss_real + l1_lambda*gen_l1_loss
        g_optimizer.zero_grad()
        gen_loss.backward()
        g_optimizer.step()

        # Logger
        t.set_postfix({"Generator loss": f"{gen_loss.item():.3f}", "Discriminator loss": f"{dis_loss.item():.3f}"})
        t.update()