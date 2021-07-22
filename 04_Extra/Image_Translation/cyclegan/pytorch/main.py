# %%
import itertools
from tqdm import tqdm

import torch
from torch import nn

from dataloader import *
from models import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# %%
# =================
# Set Configuration
# =================
PATH= "../../datasets/summer2winter_yosemite"
INPUTSIZE= 256
BATCHSIZE= 16
NUMWORKER= 2

EPOCHS = 5
BATCH_SIZE = 16
LR = 1e-5

# %%
# =================
# Data Processing
# =================
dataloaders = CustomDataloader(None, input_size=INPUTSIZE, path=PATH, batch_size=BATCHSIZE, num_workers=NUMWORKER)

# %%
# =================
# Set Models
# =================
G_AtoB = Generator(3, 3, 64, "IN", 2, 5)
G_BtoA = Generator(3, 3, 64, "IN", 2, 5)

D_AtoB = Discriminator(3, 64, "BN", 4)
D_BtoA = Discriminator(3, 64, "BN", 4)

GANLoss = nn.BCELoss()
CycleLoss = torch.nn.L1Loss()
IdentityLoss = torch.nn.L1Loss()

optimizer_G_AtoB = torch.optim.Adam(G_AtoB.parameters(), lr=LR)
optimizer_G_BtoA = torch.optim.Adam(G_BtoA.parameters(), lr=LR)
optimizer_D_AtoB = torch.optim.Adam(D_AtoB.parameters(), lr=LR)
optimizer_D_BtoA = torch.optim.Adam(D_BtoA.parameters(), lr=LR)

# %%
# =================
# Training Loop
# =================
for epoch in range(EPOCHS):
    
    print(f"[{epoch+1}/{EPOCHS}]")
    
    G_AtoB.train()
    G_BtoA.train()
    D_AtoB.train()
    D_BtoA.train()
    with tqdm(len(dataloaders['train'])) as t:
        t.set_description(f"Training Phase")
        for step, (a_img, b_img) in enumerate(dataloaders['train']):
            
            fake_B = G_AtoB(a_img)
            recon_A = G_BtoA(fake_B)

            fake_A = G_BtoA(B_img)
            recon_B = G_AtoB(fake_A)
            
            #  Code

            t.set_postfix({"GAN Loss": step, "Cycle Loss": step, "Identity Loss": step})
            t.update()

        
    G_AtoB.eval()
    G_BtoA.eval()
    D_AtoB.eval()
    D_BtoA.eval()
    with tqdm(len(dataloaders['test'])) as t:
        t.set_description(f"Test Phase")
        with torch.no_grad():

            for step, (a_img, b_img) in enumerate(dataloaders['test']):
                
                fake_B = G_AtoB(a_img)
                recon_A = G_BtoA(fake_B)

                fake_A = G_BtoA(B_img)
                recon_B = G_AtoB(fake_A)
                
                # Code

                t.set_postfix({"Val GAN Loss": step, "Val Cycle Loss": step, "Val Identity Loss": step})
                t.update()

    print("")
# %%
