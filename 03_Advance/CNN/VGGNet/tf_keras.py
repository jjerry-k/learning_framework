#%%
# Import Package
import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers, models, losses, optimizers, datasets, utils

# %%
# Data Prepare
img_size = 96
num_classes = 10

(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()

train_x = np.array([cv.resize(i, (img_size, img_size)) for i in train_x])[..., np.newaxis]/255.
test_x = np.array([cv.resize(i, (img_size, img_size)) for i in test_x])[..., np.newaxis]/255.

train_y = utils.to_categorical(train_y, num_classes)
test_y = utils.to_categorical(test_y, num_classes)

print("Train Data's Shape : ", train_x.shape, train_y.shape)
print("Test Data's Shape : ", test_x.shape, test_y.shape)

# %%
# Build Networks
def build_vgg(input_shape=(None, None, 3), num_layer=16, num_classes=1, name='vgg'):
    num_layer_list = [11, 13, 16, 19]
    
    blocks_dict = {
        11: [1, 1, 2, 2, 2],
        13: [2, 2, 2, 2, 2], 
        16: [2, 2, 3, 3, 3], 
        19: [2, 2, 4, 4, 4]
    }

    num_channel_list = [64, 128, 256, 512, 512]

    assert num_layer in  num_layer_list, "Number of layer must be in %s"%num_layer_list
    
    last_act = 'sigmoid' if num_classes==1 else 'softmax'
    name = name+str(num_layer)

    model = models.Sequential(name=name)
    model.add(layers.Input(shape=input_shape, name=name+"_Input"))
    for idx, num_iter in enumerate(blocks_dict[num_layer]):
        for jdx in range(num_iter):
            model.add(layers.Conv2D(num_channel_list[idx], 3, strides=1, padding='same', activation='relu', name=name+"_Block_%d_Conv%d"%(idx+1, jdx+1)))
        model.add(layers.MaxPool2D(name=name+"_Block%d_Pool"%(idx+1)))
    model.add(layers.GlobalAveragePooling2D(name=name+"_GAP"))
    model.add(layers.Dense(512, activation='relu', name=name+"_Dense_1"))
    model.add(layers.Dense(512, activation='relu', name=name+"_Dense_2"))
    model.add(layers.Dense(num_classes, activation=last_act, name=name+"_Output"))
    return model

num_layer = 11
input_shape = train_x.shape[1:]

vgg = build_vgg(input_shape=input_shape, num_layer=num_layer, num_classes=num_classes)
vgg.summary()

loss = 'binary_crossentropy' if num_classes==1 else 'categorical_crossentropy'
vgg.compile(optimizer=optimizers.Adam(), loss=loss, metrics=['accuracy'])



# %%
# Training Network
epochs=10
batch_size=256

history=vgg.fit(train_x, train_y, epochs = epochs, batch_size=batch_size, validation_data=[test_x, test_y])