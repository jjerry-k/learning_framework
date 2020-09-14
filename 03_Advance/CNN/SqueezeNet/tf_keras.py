# %%
# Import Package
import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers, models, losses, optimizers, datasets, utils
from tensorflow.keras import backend as K

# %%
# Data Prepare

URL = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
path_to_zip  = utils.get_file('flower_photos.tgz', origin=URL, extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'flower_photos')

category_list = [i for i in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, i)) ]
print(category_list)

num_classes = len(category_list)
img_size = 150

def read_img(path, img_size):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (img_size, img_size))
    return img

imgs_tr = []
labs_tr = []

imgs_val = []
labs_val = []

for i, category in enumerate(category_list):
    path = os.path.join(PATH, category)
    imgs_list = os.listdir(path)
    print("Total '%s' images : %d"%(category, len(imgs_list)))
    ratio = int(np.round(0.05 * len(imgs_list)))
    print("%s Images for Training : %d"%(category, len(imgs_list[ratio:])))
    print("%s Images for Validation : %d"%(category, len(imgs_list[:ratio])))
    print("=============================")

    imgs = [read_img(os.path.join(path, img),img_size) for img in imgs_list]
    labs = [i]*len(imgs_list)

    imgs_tr += imgs[ratio:]
    labs_tr += labs[ratio:]
    
    imgs_val += imgs[:ratio]
    labs_val += labs[:ratio]

imgs_tr = np.array(imgs_tr)/255.
labs_tr = utils.to_categorical(np.array(labs_tr), num_classes)

imgs_val = np.array(imgs_val)/255.
labs_val = utils.to_categorical(np.array(labs_val), num_classes)

print(imgs_tr.shape, labs_tr.shape)
print(imgs_val.shape, labs_val.shape)

# %%
# Build Network

def Conv_Block(input, filters, ksize, stride, padding, activation, use_bn=False, name="Conv"):
    out = layers.Conv2D(filters, ksize, stride, padding, name=name+"_Conv")(input)
    if use_bn:
        out = layers.BatchNormalization(name=name+"_BN")(out)
    out = layers.Activation(activation, name=name+"_Act")(out)
    return out

def Fire_Module(input, squ, exp_1x1, exp_3x3, use_bn=False, name="Fire"):
    
    squeeze = Conv_Block(input, squ, 1, 1, 'valid', 'relu', name=name+"_Squeeze")

    expand_1x1 = Conv_Block(squeeze, exp_1x1, 1, 1, 'valid', 'linear', name=name+"_Expand_1x1")
    expand_3x3 = Conv_Block(squeeze, exp_3x3, 3, 1, 'same', 'linear', name=name+"_Expand_3x3")

    out = layers.Concatenate(name=name+"_Expand")([expand_1x1, expand_3x3])
    out = layers.ReLU(name=name+"_Act")(out)

    return out

def build_squeezenet(input_shape=(None, None, 3), num_classes=1, name='SqueezeNet'):
    
    last_act = 'sigmoid' if num_classes==1 else 'softmax'

    input = layers.Input(shape=input_shape, name=name+"_input")

    x = Conv_Block(input, 96, 7, 2, 'same', 'relu', name=name+"_Block_1")
    x = layers.MaxPool2D(3, 2, name=name+"_Pool_1")(x)

    x = Fire_Module(x, 16, 64, 64, name=name+"_Fire_2")
    x = Fire_Module(x, 16, 64, 64, name=name+"_Fire_3")
    x = Fire_Module(x, 32, 128, 128, name=name+"_Fire_4")
    x = layers.MaxPool2D(3, 2, name=name+"_Pool_4")(x)

    x = Fire_Module(x, 32, 128, 128, name=name+"_Fire_5")
    x = Fire_Module(x, 48, 192, 192, name=name+"_Fire_6")
    x = Fire_Module(x, 48, 192, 192, name=name+"_Fire_7")
    x = Fire_Module(x, 64, 256, 256, name=name+"_Fire_8")
    x = layers.MaxPool2D(3, 2, name=name+"_Pool_8")(x)

    x = Fire_Module(x, 64, 256, 256, name=name+"_Fire_9")
    x = layers.Dropout(0.5, name=name+"_Dropout")(x)
    x = Conv_Block(x, num_classes, 1, 1, 'valid', 'relu', name=name+"_Block_10")

    x = layers.GlobalAveragePooling2D(name=name+"_GAP")(x)
    x = layers.Activation(last_act, name=name+"_Output")(x)

    return models.Model(input, x)

input_shape = imgs_tr.shape[1:]
alpha = 1

squeeze = build_squeezenet(input_shape=input_shape, num_classes=num_classes, name="Squeeze")
squeeze.summary()

loss = 'binary_crossentropy' if num_classes==1 else 'categorical_crossentropy'
squeeze.compile(optimizer=optimizers.Adam(0.04), loss=loss, metrics=['accuracy'])

# %%
# Training Network
epochs=100
batch_size=16

history=squeeze.fit(imgs_tr, labs_tr, epochs = epochs, batch_size=batch_size, 
                   validation_data=(imgs_val, labs_val))

plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.title("Loss graph")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Train', 'Validation'], loc='upper right')

plt.subplot(122)
plt.title("Acc graph")
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['Train', 'Validation'], loc='upper right')

plt.show()