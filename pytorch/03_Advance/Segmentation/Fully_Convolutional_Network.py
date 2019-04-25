# import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument("--content", help="Path of Content Image", type=str)
# parser.add_argument("--style", help="Path of Style Image", type=str)
# parser.add_argument("--scale", help="Scaling Factor", type=float, default=1.0)
# parser.add_argument("--steps", help="Steps of Training", type=int, default=2000)
# args = parser.parse_args()

import torch
import torch.nn as nn
import torch.optim as opti
from torch.autograd import Variable

import PIL
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from keras.utils import Progbar
import torchvision.transforms as transforms
import torchvision.models as models

print("Loading Packages!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_block(in_channel, out_channel, num_iter):
    network = []
    network.append(nn.Conv2d(in_channel, out_channel, 3, padding=1))
    network.append(nn.ReLU(True))
    for iter in range(num_iter-1):
        network.append(nn.Conv2d(out_channel, out_channel, 3, padding=1))
        network.append(nn.ReLU(True))
    return nn.Sequential(*network)

class FCN_8():
    def __init__(self, num_classes=10, version=8):
        super(FCN_8, self).__init__()
        self.num_classes = 10
        self.block1 = make_block(3, 64, 2)
        self.pool1 = nn.MaxPool2d(2, 2) # /2
        self.block2 = make_block(64, 128, 2)
        self.pool2 = nn.MaxPool2d(2, 2) # /4
        self.block3 = make_block(128, 256, 3)
        self.pool3 = nn.MaxPool2d(2, 2) # /8
        self.block4 = make_block(256, 512, 3)
        self.pool4 = nn.MaxPool2d(2, 2) # /16
        self.block5 = make_block(512, 512, 3)
        self.pool5 = nn.MaxPool2d(2, 2) # /32

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.score = nn.Conv2d(4096, num_classes, 1)
        self.dropout = nn.Dropout(0.5, True)

        self.upsample1 = nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1) # /16
        self.upsample2 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1) # /8
        self.upsample3 = nn.ConvTranspose2d(256, num_classes, 16, stride=8, padding=4) # /1

        self.relu = nn.ReLU(True)

    def forward(self, x):
        init_input = x

        block1 = self.pool1(self.block1(x))
        block2 = self.pool2(self.block2(block1))
        block3 = self.pool3(self.block3(block2))
        block4 = self.pool4(self.block4(block3))
        block5 = self.pool5(self.block5(block4))

        upsample1 = self.relu(self.upsample1(block5)) + block4
        upsample2 = self.relu(self.upsample2(upsample1)) + block3
        upsample3 = self.relu(self.upsample3(upsample2))

        return upsample3
