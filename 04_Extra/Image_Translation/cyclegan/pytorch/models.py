import torch
from torch import nn
from torch.nn import functional as F


class NormLayer(nn.Module):
    def __init__(self, features, norm_type='IN'):
        super(NormLayer, self).__init__()
        
        if norm_type == "BN":
            self.norm = nn.BatchNorm2d(features, affine=True, track_running_stats=True)
        elif norm_type == "IN":
            self.norm = nn.InstanceNorm1d(features, affine=False, track_running_stats=False)
        elif norm_type == "None":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Please input 'norm_type', ['BN', 'IN', 'None']")

    def forward(self, x):
        x = self.norm(x)
        return x

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, features, norm_type, n_downsampling, n_blocks):
        super(Generator, self).__init__()
        
        layers = [
            nn.ReflectionPad2d(3), 
            nn.Conv2d(in_channels, features, kernel_size=7, padding=0),
            NormLayer(features, norm_type),
            nn.ReLU(True)
        ]

        # Downsampling
        for i in range(n_downsampling):
            multiply = 2 ** i
            prev_channels = features * multiply
            new_channels = prev_channels * 2
            layers.append(nn.Conv2d(prev_channels, new_channels, kernel_size=3, stride=2, padding=1))
            layers.append(NormLayer(new_channels, norm_type))
            layers.append(nn.ReLU(True))
        
        # Residual BlockS
        multiply = 2 ** n_downsampling
        curr_channels = features * multiply
        for i in range(n_blocks):
            # layers.append(ResidualBlock(curr_channels, norm_type))
            pass
        
        # Upsampling
        for i in range(n_downsampling):
            prev_channels = features * (2 ** (n_downsampling - i))
            curr_channels = int(prev_channels /2)
            layers.append(nn.ConvTranspose2d(prev_channels, curr_channels, 
                                             kernel_size=3, stride=2, padding=1, 
                                             output_padding=1))
            layers.append(NormLayer(curr_channels, norm_type))
            layers.append(nn.ReLU(True))
        
        layers.append(nn.ReflectionPad2d(3))
        layers.append(nn.Conv2d(features, out_channels, kernel_size=7, padding=0))
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels, features, norm_type, n_blocks):
        super(Discriminator, self).__init__()
        """
        C64-C128-C256-C512
        """
        layers = [
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        new_channels = features * 2
        prev_channels = features
        for i in range(1, n_layers):
            layers.append(nn.Conv2d(prev_channels, new_channels, kernel_size=4, stride=2, padding=1))
            layers.append(NormLayer(new_channels, norm_type))
            layers.append(nn.LeakyReLU(0.2, True))
            prev_channels = new_channels
            new_channels = min(features * (2**(i+1)), 512)

        layers.append(prev_channels, 1, kernel_size=4, stride=1, padding=1)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x