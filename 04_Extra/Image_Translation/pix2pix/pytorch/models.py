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
        
        for i in range(1, 7):
            output_channel = min(num_features * (2**(i+1)), 512)
            layer_list.append(Encoding_Block(prev_features, output_channel))
            prev_features = output_channel
        layer_list.append(Encoding_Block(prev_features, prev_features, use_bn=False))

        for i in range(3):
            layer_list.append(Decoding_Block(prev_features, prev_features))
            layer_list.append(nn.Dropout(0.5))

        for i in range(4):
            output_channel = prev_features // (2**(i+1))
            layer_list.append(Decoding_Block(prev_features, output_channel))
            prev_features = output_channel

        layer_list.append(Decoding_Block(prev_features, B_channel, use_bn=False))
        layer_list.append(nn.Tanh())

        self.net = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.net(x)


class Generator_Unet(nn.Module):
    def __init__(self, A_channel=3, B_channel=3, num_features=64):
        super(Generator_Unet, self).__init__()
        
        self.en1 = Encoding_Block(A_channel, num_features, use_act=False, use_bn=False)
        self.en2 = Encoding_Block(num_features, num_features*2)
        self.en3 = Encoding_Block(num_features*2, num_features*4)
        self.en4 = Encoding_Block(num_features*4, num_features*8)
        self.en5 = Encoding_Block(num_features*8, num_features*8)
        self.en6 = Encoding_Block(num_features*8, num_features*8)
        self.en7 = Encoding_Block(num_features*8, num_features*8)
        self.en8 = Encoding_Block(num_features*8, num_features*8, use_bn=False)

        self.de1 = nn.Sequential(
            Decoding_Block(num_features*8, num_features*8),
            nn.Dropout(0.5)
        )
        self.de1 = nn.Sequential(
            Decoding_Block(num_features*8, num_features*8),
            nn.Dropout(0.5)
        )
        self.de1 = nn.Sequential(
            Decoding_Block(num_features*8, num_features*8),
            nn.Dropout(0.5)
        )
        self.de4 = Decoding_Block(num_features*8, num_features*8)
        self.de5 = Decoding_Block(num_features*8, num_features*4)
        self.de6 = Decoding_Block(num_features*4, num_features*2)
        self.de7 = Decoding_Block(num_features*2, num_features)
        self.de8 = nn.Sequential(
            Decoding_Block(num_features, B_channel, use_bn=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        en1 = self.en1(x)
        en2 = self.en2(en1)
        en3 = self.en3(en2)
        en4 = self.en4(en3)
        en5 = self.en5(en4)
        en6 = self.en6(en5)
        en7 = self.en7(en6)
        en8 = self.en8(en7)

        de1 = torch.cat([self.de1(en8), en7], dim=1)
        de2 = torch.cat([self.de2(de1), en6], dim=1)
        de3 = torch.cat([self.de3(de2), en5], dim=1)
        de4 = torch.cat([self.de4(de3), en4], dim=1)
        de5 = torch.cat([self.de5(de4), en3], dim=1)
        de6 = torch.cat([self.de6(de5), en2], dim=1)
        de7 = torch.cat([self.de7(de6), en1], dim=1)
        de8 = self.de8(de7)
        return de8

class Encoding_Block_Dis(nn.Module):
    def __init__(self, in_channel=3, output_channel=32, ksize=4, strides=2, padding=1, use_act=True, use_bn=True):
        super(Encoding_Block_Dis, self).__init__()
        
        layer_list = []
        layer_list.append(nn.Conv2d(in_channel, output_channel, ksize, strides, padding))
        if use_bn:
            layer_list.append(nn.BatchNorm2d(output_channel))
        if use_act:
            layer_list.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.module = nn.Sequetial(*layer_list)

    def forward(self, x):
        return self.module(x)

class Discriminator(nn.Module):
    def __init__(self, A_channel, B_channel, num_features=64, n_layers=0):
        super(Discriminator, self).__init__()

        layer_list = []

        if n_layers == 0:
            layer_list.append(Encoding_Block_Dis(A_channel+B_channel, num_features, ksize=1, strides=1, use_bn=False))
            layer_list.append(Encoding_Block_Dis(num_features, num_features*2, ksize=1, strides=1, use_bn=False))
            layer_list.append(Encoding_Block_Dis(num_features*2, 1, ksize=1, strides=1, use_act=False, use_bn=False))
            layer_list.append(nn.Sigmoid())
        
        else:
            layer_list.append(Encoding_Block_Dis(A_channel+B_channel, num_features, use_bn=False))
            
            prev_features = num_features
            for i in range(1, n_layers-1):
                mul_fact = min(2**i, 8)
                layer_list.append(Encoding_Block_Dis(prev_features, num_features*mul_fact))
                prev_features = num_features*mul_fact
            
            mul_fact = min(2**(n_layers-1), 8)
            layer_list.append(Encoding_Block(prev_features, num_features*mul_fact, strides=1))
            layer_list.append(Encoding_Block(num_features*mul_fact, 1, strides=1, use_act=False, use_bn=False))
            layer_list.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.net(x)