# %%
# Load package
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.applications import VGG19, MobileNet, Xception

def grad_cam(model, layer_name, label_index, img):
    """
    ========= Input =========
    model: Model instance
    activation_layer: Name of layer
    label_index: Index of labels
    img: Image using Grad-CAM

    ========= Output =========
    output_image: Activation map + real image
    cam: Activation map
    """
    H, W = img.shape[1:3]
    grad_model = models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = predictions[:, label_index]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

    weights = tf.reduce_mean(guided_grads, axis=(0, 1))

    cam = np.ones(output.shape[0: 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv.resize(cam.numpy(), (W, H))
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())

    cam = cv.applyColorMap(np.uint8(255*heatmap), cv.COLORMAP_JET)
    
    output_image = cv.addWeighted(cv.cvtColor((img[0]*255).astype('uint8'), cv.COLOR_RGB2BGR), 0.5, cam, 0.5, 0)

    output_image = cv.cvtColor(output_image, cv.COLOR_BGR2RGB)
    
    cam = cv.cvtColor(cam, cv.COLOR_BGR2RGB)
    return output_image, cam

# %%
# Read image
img = cv.imread('../cat_dog.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.resize(img, (224, 224))

# %%
# Load model

model = VGG19(weights='imagenet')

model.summary()

# After prinintg, copy layer's name
# %%
# 174 tabby
# 211 german_shepherd

overlaped, cam = grad_cam(model, 'block5_pool', 174, img[np.newaxis]/255.)
cv.imwrite(f'./tabby.jpg', overlaped)

overlaped, cam = grad_cam(model, 'block5_pool', 211, img[np.newaxis]/255.)
cv.imwrite(f'./german_shepherd.jpg', overlaped)