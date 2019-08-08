import torch
from torch import nn

class DenseLayer(nn.Sequential):
    def __init__(self, input_feature, growth_rate, bn_size):
        super(DenseLayer, self).__init__()
        self.add_module('BN1', nn.BatchNorm2d(input_feature)),
        self.add_module('Act1', nn.ReLU(True)),
        self.add_module('Conv1', nn.Conv2d(input_feature, growth_rate * 4, 1)),
        self.add_module('BN2', nn.BatchNorm2d(growth_rate * 4)),
        self.add_module('Act2', nn.ReLU(True)),
        self.add_module('Conv2', nn.Conv2d(growth_rate * 4, growth_rate, 3, padding=1))
    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], dim=1)

class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, input_feature, growth_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(input_feature + i * growth_rate, growth_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class Transition_layer(nn.Sequential):
    def __init__(self, input_feature, reduction):
        self.add_module('BN', nn.BatchNorm2d(input_feature)),
        self.add_module('Act', nn.ReLU(True)),
        self.add_module('Conv', nn.Conv2d(input_feature, input_feature * reduction, kernel_size=1)),
        self.add_module('Pool', nn.AvgPool2d(2, 2))
