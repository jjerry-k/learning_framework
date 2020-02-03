import torch
from torch import nn

cfg = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

def Make_VGG(structure):
    layers = []
    input_feature = 3
    for layer in structure:
        if layer == 'M':
            layers.append(nn.MaxPool2d(2, 2))
        else:
            conv = nn.Conv2d(input_feature, layer, 3, padding=1)
            layers.append(conv)
            layers.append(nn.ReLU(True))
            input_feature = layer
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, cfg, num_class):
        super(VGG, self).__init__()
        self.vgg = Make_VGG(cfg)
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )
    def forward(self, x):
        x = self.vgg(x)
        x = self.avgpool(x)
        x = x.view(x.size[0], -1)
        x = self.classifier(x)
        return x
