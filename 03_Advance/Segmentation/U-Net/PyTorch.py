# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
    
class conv_block(nn.Module):
    '''(Conv, ReLU) * 2'''
    def __init__(self, in_ch, out_ch, pool=None):
        super(conv_block, self).__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(out_ch, out_ch, 3, padding=1),
                  nn.ReLU(inplace=True)]
        
        if pool:
            layers.insert(0, nn.MaxPool2d(2, 2))
        
        self.conv = nn.Sequential(*layers)
            

    def forward(self, x):
        x = self.conv(x)s
        return x


class upconv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(upconv_block, self).__init__()

        self.upconv = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x1, x2):
        # x1 : unpooled feature
        # x2 : encoder feature
        x1 = self.upconv(x1)
        x1 = nn.UpsamplingBilinear2d(x2.size()[2:])(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class build_unet(nn.Module):
    def __init__(self):
        super(build_unet, self).__init__()
        self.conv1 = conv_block(1, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.conv5 = conv_block(512, 1024, pool=True)
        
        self.unconv4 = upconv_block(1024, 512)
        self.unconv3 = upconv_block(512, 256)
        self.unconv2 = upconv_block(256, 128)
        self.unconv1 = upconv_block(128, 64)
        
        self.prediction = nn.Conv2d(64, 1, 1)
        
    def forward(self, x):
        en1 = self.conv1(x) #/2
        en2 = self.conv2(en1) #/4
        en3 = self.conv3(en2) #/8
        en4 = self.conv4(en3) #/16
        en5 = self.conv5(en4) 
        
        de4 = self.unconv4(en5, en4) # /8
        de3 = self.unconv3(de4, en3) # /4
        de2 = self.unconv2(de3, en2) # /2
        de1 = self.unconv1(de2, en1) # /1
        
        output = self.prediction(de1)
        return output
        