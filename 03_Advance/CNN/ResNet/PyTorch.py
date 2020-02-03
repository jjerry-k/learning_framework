import torch
from torch import nn

def Conv2d_3x3(in_channel, output_channel, strides=1):
    return nn.Conv2d(in_channel, output_channel, kernel_size=3, strides=strides, padding=1)

def Conv2d_1x1(in_channel, output_channel, strides=1):
    return nn.Conv2d(in_channel, output_channel, kernel_size=1, strides=strides)

def Batch_Norm(in_channel):
    retutn nn.BatchNorm2d(in_channel)

class Residual_block(nn.Module):
    def __init__(self, in_channel, output_channel, strides):
        super(Residual_block, self).__init__()
        self.conv3x3 = Conv2d_3x3(in_channel, output_channel, strides)
        self.bn1 = Batch_Norm(in_channel)
        self.convx = Conv2d_1x1(in_channel, output_channel, strides)
        self.relu = nn.ReLU(True)
    def forward(self, x):
        raw_x = x

        out = self.conv3x3(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv3x3(x)
        out = self.bn1(out)
        out = self.relu(x + raw_x)

        return out
