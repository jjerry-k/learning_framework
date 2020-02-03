import torch
from torch import nn

def basic_conv(in_ch, out_ch, ksize=3, pad='same'):
    assert ksize%2 == 1, "Please use ksize of odd number."

    if pad=='same':
        pad = (ksize-1)//2
    elif pad=='valid':
        pad = 0

    return nn.Conv2d(in_ch, out_ch, kernel_size=ksize, stride=1, padding=pad)


class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()

        layers = [
            basic_conv(3, 64, 3, 'same'), 
            nn.ReLU(inplace=True)]

        for _ in range(1, 19):
            layers.append(basic_conv(64, 64, 3, 'same'))
            layers.append(nn.ReLU(inplace=True))

        layers.append(layers.append(basic_conv(64, 3, 3, 'same')))

        self.net = nn.Sequential(*layers)
    def forward(self, x):
        out = self.net(x)
        return out + x


vdsr = VDSR()
print(vdsr)