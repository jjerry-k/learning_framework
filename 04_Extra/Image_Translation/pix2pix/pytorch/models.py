import torch
from torch import nn
from torch.nn import functional as F

class Encoding_Block(nn.Module):
    def __init__(self, in_channel=3, output_channel=32, ksize=4, strides=2, padding=1, use_act=True, use_bn=True):
        super(Encoding_Block, self).__init__()
        
        layer_list = []
        if use_act:
            layer_list.append(nn.LeakyReLU(0.2, inplace=True))
        layer_list.append(nn.Conv2d(in_channel, output_channel, ksize, strides, padding))
        if use_bn:
            layer_list.append(nn.BatchNorm2d(output_channel))
        
        self.module = nn.Sequetial(*layer_list)

    def forward(self, x):
        return self.module(x)

class Decoding_Block(nn.Module):
    def __init__(self, in_channel=3, output_channel=32, ksize=4, strides=2, padding=1, use_bn=True):
        super(Decoding_Block, self).__init__()

        layer_list = []
        layer_list.append(nn.ReLU(inplace=True))
        layer_list.append(nn.ConvTranspose2d(in_channel, output_channel, ksize, strides, padding))
        if use_bn:
            layer_list.append(nn.BatchNorm2d(output_channel))
        
        self.module = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.module(x)

class Generator_Encoder_Decoder(nn.Module):
    def __init__(self, A_channel=3, B_channel=3, num_features=64):
        super(Generator_Encoder_Decoder, self).__init__()
        layer_list = []

        layer_list.append(Encoding_Block(A_channel, num_features, use_act=False, use_bn=False))
        prev_features = num_features
        
        for i in range(1, 8):
            output_channel = min(num_features * (i+1), 512)
            layer_list.append(Encoding_Block(prev_features, output_channel))
            prev_features = output_channel
        
        for i in range(3):
            layer_list.append(Decoding_Block(prev_features, prev_features))
            layer_list.append(nn.Dropout(0.5))

        for i in range(4):
            output_channel = prev_features // (i+1)
            layer_list.append(Decoding_Block(prev_features, output_channel))
            prev_features = output_channel

        layer_list.append(Decoding_Block(prev_features, B_channel, use_bn=False))
        layer_list.append(nn.Tanh())

        self.net = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.net(x)