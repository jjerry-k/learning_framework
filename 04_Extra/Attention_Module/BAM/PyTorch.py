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

class BAM(nn.Module):
    def __init__(self, input_feature, reduction_ratio=16, dilation=4):
        super(BAM, self).__init__()
        
        inter_feature = input_feature//reduction_ratio

        self.attention_ch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(input_feature, inter_feature, ksize=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(inter_feature, input_feature, ksize=(1, 1)),
            nn.BatchNorm2d(input_feature)
        )
        
        self.attention_sp = nn.Sequential(
            nn.Conv2d(input_feature, inter_feature, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(inter_feature, inter_feature, kernel_size=3, stride=1, padding=1, dilation=dilation),
            nn.ReLU(True),
            nn.Conv2d(inter_feature, inter_feature, kernel_size=3, stride=1, padding=1, dilation=dilation),
            nn.ReLU(True),
            nn.Conv2d(inter_feature, 1, kernel_size=1),
            nn.ReLU(True),
            nn.BatchNorm2d(1)

        )

        self.act = nn.Sigmoid()

    def forward(self, x):

        att_ch = self.attention_ch(x)

        att_sp = self.attention_sp(x)

        att = self.act(att_ch + att_sp)

        return x + (x*att)