import torch
from torch import nn

def basic_conv(in_ch, out_ch, ksize=3, pad='same'):
    assert ksize%2 == 1, "Please use ksize of odd number."

    if pad=='same':
        pad = (ksize-1)//2
    elif pad=='valid':
        pad = 0

    return nn.Conv2d(in_ch, out_ch, kernel_size=ksize, stride=1, padding=pad)


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        layers = [
            basic_conv(3, 64, 9, 'valid'), 
            nn.ReLU(inplace=True), 
            basic_conv(64, 32, 1, 'valid'), 
            nn.ReLU(inplace=True),
            basic_conv(32, 3, 5, 'valid')]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        out = self.net(x)
        return out