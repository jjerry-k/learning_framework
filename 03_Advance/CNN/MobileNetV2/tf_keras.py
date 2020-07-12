# %%
# Import Package
import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers, models, losses, optimizers, datasets, utils

# %%
# Data Prepare

URL = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
path_to_zip  = tf.keras.utils.get_file('flower_photos.tgz', origin=URL, extract=True)

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

def conv_block(x, filters, ksize=3, strides=1, padding="same", name="Block"):
    x = layers.Conv2D(filters, ksize, strides=strides, padding=padding, name=name+"_Conv")(x)
    x = layers.BatchNormalization(name=name+"_BN")(x)
    x = layers.ReLU(max_value=6, name=name+"_Act")(x)
    return x

def depthwise_separable_block(x, filters, ksize=3, strides=1, padding="same", alpha=1, name="Block"):    
    x = layers.DepthwiseConv2D(ksize, strides=strides, padding=padding, name=name+"_Depthwise")(x)
    x = layers.BatchNormalization(name=name+"_BN_1")(x)
    x = layers.ReLU(max_value=6, name=name+"_Act_1")(x)

    x = layers.Conv2D(int(filters*alpha), 1, name=name+"_Pointwise")(x)
    x = layers.BatchNormalization(name=name+"_BN_2")(x)
    # x = layers.ReLU(name=name+"_Act_2")(x)
    return x

def inverted_residual_block(input, expansion, filters, strides=1, alpha=1, name="Inverted_Residual"):
    n_features = int(input.shape[-1])
    
    x = layers.Conv2D(int(n_features*expansion), 1, name=name+"_Expansion")(input)
    x = layers.BatchNormalization(name=name+"_BN_1")(x)
    x = layers.ReLU(max_value=6, name=name+"_Act_1")(x)

    x = layers.DepthwiseConv2D(3, strides, padding="same", name=name+"_Depthwise")(x)
    x = layers.BatchNormalization(name=name+"_BN_2")(x)
    x = layers.ReLU(max_value=6, name=name+"_Act_2")(x)

    x = layers.Conv2D(int(filters*alpha), 1, name=name+"_Pointwise")(x)
    x = layers.BatchNormalization(name=name+"_BN_3")(x)

    if strides==1 and n_features==int(filters*alpha):
        x = layers.Add(name=name+"_Add")([input, x])
    return x


def build_mobilenet(input_shape=(None, None, 3), num_classes=1, alpha=1, name='mobile'):
    
    last_act = 'sigmoid' if num_classes==1 else 'softmax'

    input = layers.Input(shape=input_shape, name=name+"_input")

    x = conv_block(input, 32, 3, 2, "same", name+"_Stem")
    
    x = inverted_residual_block(x, 1, 16, alpha=alpha, name=name+"_Block_1")
    
    x = inverted_residual_block(x, 6, 24, strides=2, alpha=alpha, name=name+"_Block_2")
    x = inverted_residual_block(x, 6, 24, alpha=alpha, name=name+"_Block_3")

    x = inverted_residual_block(x, 6, 32, strides=2, alpha=alpha, name=name+"_Block_4")
    x = inverted_residual_block(x, 6, 32, alpha=alpha, name=name+"_Block_5")
    x = inverted_residual_block(x, 6, 32, alpha=alpha, name=name+"_Block_6")
    
    x = inverted_residual_block(x, 6, 64, strides=2, alpha=alpha, name=name+"_Block_7")
    x = inverted_residual_block(x, 6, 64, alpha=alpha, name=name+"_Block_8")
    x = inverted_residual_block(x, 6, 64, alpha=alpha, name=name+"_Block_9")
    x = inverted_residual_block(x, 6, 64, alpha=alpha, name=name+"_Block_10")

    x = inverted_residual_block(x, 6, 96, alpha=alpha, name=name+"_Block_11")
    x = inverted_residual_block(x, 6, 96, alpha=alpha, name=name+"_Block_12")
    x = inverted_residual_block(x, 6, 96, alpha=alpha, name=name+"_Block_13")

    x = inverted_residual_block(x, 6, 160, strides=2, alpha=alpha, name=name+"_Block_14")
    x = inverted_residual_block(x, 6, 160, alpha=alpha, name=name+"_Block_15")
    x = inverted_residual_block(x, 6, 160, alpha=alpha, name=name+"_Block_16")

    x = inverted_residual_block(x, 6, 320, alpha=alpha, name=name+"_Block_17")

    x = layers.Conv2D(1280, 1, name=name+"_Exit_Conv")(x)
    x = layers.BatchNormalization(name=name+"_Exit_BN")(x)
    x = layers.ReLU(max_value=6, name=name+"_Exit_Act")(x)
    
    x = layers.GlobalAveragePooling2D(name=name+"_GAP")(x)
    x = layers.Dense(num_classes, activation=last_act, name=name+"_Output")(x)

    return models.Model(input, x)

input_shape = imgs_tr.shape[1:]
alpha = 1

mobile = build_mobilenet(input_shape=input_shape, num_classes=num_classes, alpha=1, name="Mobile")
mobile.summary()

loss = 'binary_crossentropy' if num_classes==1 else 'categorical_crossentropy'
mobile.compile(optimizer=optimizers.Adam(), loss=loss, metrics=['accuracy'])

# %%
# Training Network
epochs=100
batch_size=16

history=mobile.fit(imgs_tr, labs_tr, epochs = epochs, batch_size=batch_size, validation_data=[imgs_val, labs_val])

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
