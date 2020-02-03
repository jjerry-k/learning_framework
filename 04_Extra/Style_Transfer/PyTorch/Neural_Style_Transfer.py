import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--content", help="Path of Content Image", type=str)
parser.add_argument("--style", help="Path of Style Image", type=str)
parser.add_argument("--scale", help="Scaling Factor", type=float, default=1.0)
parser.add_argument("--steps", help="Steps of Training", type=int, default=2000)
args = parser.parse_args()

import torch
from torch import nn
import torch.optim as opti
from torch.autograd import Variable

import PIL
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from keras.utils import Progbar
import torchvision.transforms as transforms
import torchvision.models as models

print("Loading Packages!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_img(img_path, scale=None, resize=None):
    '''
    Image Loader

    Parameter
    =========
    img_path : str, Path of Image
    rescale : float, Scaling Factor
    resize : (int(w), int(h)), Size to resize


    Return
    =========
    img : image
    '''
    img = Image.open(img_path)
    if rescale:
        w, h = img.size
        w = int(w*scale)
        h = int(h*scale)
        img = img.resize((w, h), PIL.Image.BICUBIC)
    if resize:
        img = img.resize(resize, PIL.Image.BICUBIC)
    if img.mode =='RGBA':
        img = img.convert("RGB")
    return img

def preproc4torch(img):
    mean = np.array([[[0.485, 0.456, 0.406]]])
    std = np.array([[[0.229, 0.224, 0.225]]])
    result = np.array(img)/255.
    result = np.flip(result, axis=2)
    result = np.transpose((result-mean)/std, [2,0,1])
    result = torch.Tensor(result).unsqueeze(0)
    return result.to(device)

def deproc4plot(img):
    mean = np.array([[[0.485, 0.456, 0.406]]])
    std = np.array([[[0.229, 0.224, 0.225]]])
    result = img.detach().cpu().squeeze().numpy()
    result = np.transpose(result, [1,2,0])
    result = (result*std + mean)*255.
    result = np.clip(result, 0, 255)
    result = np.flip(result, axis=2)
    return np.uint8(result)

# Make Network for Style Transfer Using Pretrained VGG19
class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.style_idx = ['0', '5', '10', '19', '28']
        self.content_idx = ['20']
        self.extractor = models.vgg19(pretrained=True).features

    def forward(self, x, mode = None):
        """Extract multiple convolutional feature maps."""
        assert mode, "Please input mode of Extractor"
        if mode == 'content':feature_idx = self.content_idx
        else: feature_idx = self.style_idx
        features = []
        for num, layer in self.extractor.named_children():
            x = layer(x)
            if num in feature_idx:
                features.append(x)
        return features

print("Define Done!")

# Load image
content_img = load_img(args.content, rescale=args.scale)
style_img = load_img(args.style, resize=content_img.size)

content_img = preproc4torch(content_img)
style_img = preproc4torch(style_img)
print('Content image shape : ', content_img.shape)
print('Style image shape : ', style_img.shape)

target_img = Variable(content_img.data.clone(), requires_grad=True).to(device)
print('Target imate shape : ', target_img.shape)

print("Loading Image Donw!")

extractor = Extractor().to(device).eval()

optim = torch.optim.Adam([target_img], lr=0.001, betas=[0.5, 0.1])


def Content_Loss(content, target):
    return torch.mean((content[0] - target[0])**2)


def Style_Loss(style, target):
    loss = 0
    for s_f, t_f in zip(style, target):
        b, c, h, w = s_f.size()
        s_f = s_f.view(b, c, h*w)
        t_f = t_f.view(b, c, h*w)

        s_f = torch.bmm(s_f, s_f.transpose(1,2))
        t_f = torch.bmm(t_f, t_f.transpose(1,2))
        loss += torch.mean((s_f - t_f)**2) / (c*h*w)
    return loss

print("Start Styling!")

steps = args.steps
progbar = Progbar(steps)
for step in range(steps):
    content = extractor(content_img, 'content')
    style = extractor(style_img, 'style')
    target_content = extractor(target_img, 'content')
    target_style = extractor(target_img, 'style')

    c_loss = Content_Loss(content, target_content)
    s_loss = Style_Loss(style, target_style)

    loss = c_loss + 100*s_loss

    optim.zero_grad()
    loss.backward()
    optim.step()
    if (step +1)%500 ==0 :
        new_img = deproc4plot(target_img)
        save_img = Image.fromarray(new_img)
        save_img.save('new_style_image_%06d.jpg'%(step+1))
    progbar.update(step+1, [('Content loss', c_loss.cpu().detach().numpy()), ('Style loss', s_loss.cpu().detach().numpy()*100)])



save_img = Image.fromarray(new_img)
save_img.save('new_style_image.jpg')