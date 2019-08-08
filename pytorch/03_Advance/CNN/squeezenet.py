import torch
from torch import nn

class Fire_.module(nn.Sequential):
    def __init__(self, input_feature, squeeze, expand):
        super(Fire_module, self).__init__()
        self.conv = nn.Conv2d(input_feature, squeeze, 1)
        self.act = nn.ReLU(True)
        self.e_l = nn.Conv2d(input_feature, squeeze, 1)
        self.e_r = nn.Conv2d(input_feature, squeeze, 3, padding=1)
def forward(self, x):
    x = self.act(self.conv(x))
    e_l = self.act(self.e_l(x))
    e_r = self.act(self.e_r(x))
    return torch.cat([e_l, e_r], dim=1)
