import torch
import torch.nn as nn

from torchvision import models
from torchvision import transforms

from PIL import Image
import numpy as np

class build_model(nn.Module):
    def __init__(self, base_model="efficientnet_b0"):
        super(build_model, self).__init__()

        assert base_model in dir(models), "Please Check 'base_model' in https://pytorch.org/vision/stable/models.html"
        # get the pretrained VGG19 network
        self.net = eval(f"models.{base_model}")(pretrained=True)
        
        # disect the network to access its last convolutional layer
        self.features = self.net.features
        
        # get the classifier of the vgg19
        self.classifier = self.net.classifier
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features(x)
        # register the hook
        h = x.register_hook(self.activations_hook)
        x = self.net.avgpool(x)
        x = nn.Flatten()(x)
        # apply the remaining pooling
        x = self.classifier(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features(x)
    
net = build_model(base_model="vgg19")    
net.eval()

# use the ImageNet transformation
transform = transforms.Compose([transforms.Resize((224, 224)), 
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

raw_img = Image.open("./cat_dog.jpg")
img = transform(raw_img).unsqueeze(0)

pred = net(img)

idx2cls = {
174: "tabby",
211: "german_shepherd"
}

idx = 174

pred[:, idx].backward()
gradients = net.get_activations_gradient()
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
activations = net.get_activations(img).detach()

for i in range(activations.shape[1]):
    activations[:, i, :, :] *= pooled_gradients[i]
    
heatmap = torch.mean(activations, dim=1).squeeze().numpy()
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

# TDL
# Colormap draw
import cv2
img = cv2.imread('./cat_dog.jpg')
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite(f'./{idx2cls[idx]}.jpg', superimposed_img)