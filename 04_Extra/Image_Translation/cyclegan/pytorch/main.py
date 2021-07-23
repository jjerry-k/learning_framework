# %%
import itertools
from tqdm import tqdm

import torch
from torch import nn

from dataloader import *
from models import *
from helper import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# %%
# =================
# Set Configuration
# =================
PATH= "../../datasets/summer2winter_yosemite"
INPUTSIZE= 256
BATCHSIZE= 4
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
G_AtoB = Generator(3, 3, 64, "IN", 2, 3).to(device)
G_BtoA = Generator(3, 3, 64, "IN", 2, 3).to(device)

D_A = Discriminator(3, 64, "BN", 3).to(device)
D_B = Discriminator(3, 64, "BN", 3).to(device)

GANLoss = nn.BCEWithLogitsLoss()
CycleLoss = torch.nn.L1Loss()
IdentityLoss = torch.nn.L1Loss()

optimizer_G_AtoB = torch.optim.Adam(G_AtoB.parameters(), lr=LR)
optimizer_G_BtoA = torch.optim.Adam(G_BtoA.parameters(), lr=LR)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=LR)
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=LR)

# %%
# =================
# Training Loop
# =================
for epoch in range(EPOCHS):
    
    print(f"[{epoch+1}/{EPOCHS}]")
    
    G_AtoB.train()
    G_BtoA.train()
    D_B.train()
    D_A.train()
    with tqdm(len(dataloaders['train'])) as t:

        t.set_description(f"Training Phase")
        for step, (a_img, b_img) in enumerate(dataloaders['train']):
            a_img = a_img.to(device)
            b_img = b_img.to(device)

            fake_B = G_AtoB(a_img)
            recon_A = G_BtoA(fake_B)
            idt_A = G_AtoB(b_img)

            fake_A = G_BtoA(b_img)
            recon_B = G_AtoB(fake_A)
            idt_B = G_BtoA(a_img)

            # Training Generator
            set_requires_grad([D_A, D_B], False)
            
            fake_A_pred = D_A(fake_A)
            fake_A_label = torch.ones_like(fake_A_pred)
            loss_fake_A = GANLoss(fake_A_pred, fake_A_label)
            cycle_A = CycleLoss(a_img, recon_A)
            identity_A = IdentityLoss(b_img, idt_A)
            loss_A = loss_fake_A + 10*(cycle_A + 0.5*identity_A)

            fake_B_pred = D_A(fake_B)
            fake_B_label = torch.ones_like(fake_B_pred)
            loss_fake_B = GANLoss(fake_B_pred, fake_B_label)
            cycle_B = CycleLoss(b_img, recon_B)
            identity_B = IdentityLoss(a_img, idt_B)
            loss_B = loss_fake_B + 10*(cycle_B + 0.5*identity_B)

            Gen_loss = loss_A + loss_B
            
            optimizer_G_AtoB.zero_grad()
            optimizer_G_BtoA.zero_grad()
            Gen_loss.backward()
            # loss_A.backward()
            # loss_B.backward()
            optimizer_G_AtoB.step()
            optimizer_G_BtoA.step()

            # Training Discriminator
            set_requires_grad([D_A, D_B], True)
            real_B_pred = D_B(b_img)
            real_B_label = torch.ones_like(real_B_pred)
            loss_real_B = GANLoss(real_B_pred, real_B_label)
            fake_B_pred = D_B(fake_B.detach())
            fake_B_label = torch.zeros_like(fake_B_pred)
            loss_fake_B = GANLoss(fake_B_pred, fake_B_label)
            loss_D_B = (loss_real_B + loss_fake_B) * 0.5
            optimizer_D_B.zero_grad()
            loss_D_B.backward()
            optimizer_D_B.step()

            real_A_pred = D_A(b_img)
            real_A_label = torch.ones_like(real_A_pred)
            loss_real_A = GANLoss(real_A_pred, real_A_label)
            fake_A_pred = D_A(fake_A.detach())
            fake_A_label = torch.zeros_like(fake_A_pred)
            loss_fake_A = GANLoss(fake_A_pred, fake_A_label)
            loss_D_A = (loss_real_A + loss_fake_A) * 0.5
            optimizer_D_A.zero_grad()
            loss_D_A.backward()
            optimizer_D_A.step()


            t.set_postfix(
                {"GAN Loss": (loss_D_A.item() + loss_D_B.item()) * 0.5, 
                "Cycle Loss": (cycle_A.item() + cycle_B.item()) * 0.5, 
                "Identity Loss": (identity_A.item() + identity_B.item()) * 0.5})
            t.update()

        
    G_AtoB.eval()
    G_BtoA.eval()
    D_B.eval()
    D_A.eval()
    
    with tqdm(len(dataloaders['test'])) as t:
        t.set_description(f"Test Phase")
        with torch.no_grad():

            for step, (a_img, b_img) in enumerate(dataloaders['test']):
                a_img = a_img.to(device)
                b_img = b_img.to(device)
                
                fake_B = G_AtoB(a_img)
                recon_A = G_BtoA(fake_B)

                fake_A = G_BtoA(b_img)
                recon_B = G_AtoB(fake_A)
                
                # Code

                t.set_postfix({"Val GAN Loss": step, "Val Cycle Loss": step, "Val Identity Loss": step})
                t.update()

    print("")
# %%
