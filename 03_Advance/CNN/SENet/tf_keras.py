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

def Conv_Block(input, filters, ksize, stride, padding, activation, use_bn=False, name="Conv"):
    out = layers.Conv2D(filters, ksize, stride, padding, name=name+"_Conv")(input)
    if use_bn:
        out = layers.BatchNormalization(name=name+"_BN")(out)
    out = layers.Activation(activation, name=name+"_Act")(out)
    return out

def Squeeze_Excitation_Module(input, filters, reduction_ratio, name="SE"):

    sq = layers.GlobalAvgPool2D(name=name+"_Squeeze")(input)
    ex1 = layers.Dense(filters//reduction_ratio, activation='relu', name=name+"_Excitation_1")(sq)
    ex2 = layers.Dense(filters, activation='sigmoid', name=name+"_Excitation_2")(ex1)
    ex = layers.Reshape([1, 1, filters], name=name+"_Reshape")(ex2)

    out = layers.Multiply(name=name+"_Multiply")([input, ex])
    
    return out

def SE_Block(input, filters, strides, reduction_ratio, use_bn=False, use_proj=False, proj_ksize=3, name="Block"):
    out = Conv_Block(input, filters//2, 1, 1, 'same', 'relu', use_bn, name=name+"_Conv_1")
    out = Conv_Block(out, filters, 3, strides, 'same', 'relu', use_bn, name=name+"_Conv_2")
    out = Conv_Block(out, filters, 1, 1, 'same', 'linear', use_bn, name=name+"_Conv_3")

    out = Squeeze_Excitation_Module(out, filters, reduction_ratio, name=name+"_SE")

    proj = Conv_Block(input, filters, proj_ksize, strides, 'same', 'linear', use_bn, name=name+"_Proj") if use_proj else input # k:[1, 3], s:[1, 2]

    out = layers.Add(name=name+"_Add")([out, proj])

    out = layers.ReLU(name=name+"_Act")(out)
    return out
    
def build_senet(input_shape=(None, None, 3), num_classes=1, name='SqueezeNet'):
    # Very Heavy
    last_act = 'sigmoid' if num_classes==1 else 'softmax'

    input = layers.Input(shape=input_shape, name=name+"_input")

    x = Conv_Block(input, 64, 3, 2, 'same', 'relu', use_bn=True, name=name+"_Block_1_1")
    x = Conv_Block(input, 64, 3, 1, 'same', 'relu', use_bn=True, name=name+"_Block_1_2")
    x = Conv_Block(input, 128, 3, 1, 'same', 'relu', use_bn=True, name=name+"_Block_1_3")
    x = layers.MaxPool2D(3, 2, name=name+"_Pool_1")(x)
    
    x = SE_Block(x, 256, 1, 16, True, True, 1, name=name+"_Block_2_1")
    for i in range(2):
        x = SE_Block(x, 256, 1, 16, True, name=name+"_Block_2_%d"%(i+2))

    x = SE_Block(x, 512, 2, 16, True, True, 3, name=name+"_Block_3_1")
    for i in range(7):
        x = SE_Block(x, 512, 1, 16, True, name=name+"_Block_3_%d"%(i+2))

    x = SE_Block(x, 1024, 2, 16, True, True, 3, name=name+"_Block_4_1")
    for i in range(35):
        x = SE_Block(x, 1024, 1, 16, True, name=name+"_Block_4_%d"%(i+2))

    x = SE_Block(x, 2048, 2, 16, True, True, 3, name=name+"_Block_5_1")
    for i in range(2):
        x = SE_Block(x, 2048, 1, 16, True, name=name+"_Block_5_%d"%(i+2))

    x = layers.GlobalAveragePooling2D(name=name+"_GAP")(x)
    x = layers.Dropout(0.2, name=name+"_Dropout")(x)
    x = layers.Dense(num_classes, activation=last_act, name=name+"_Output")(x)

    return models.Model(input, x)

input_shape = imgs_tr.shape[1:]
alpha = 1

senet = build_senet(input_shape=input_shape, num_classes=num_classes, name="SENet")
senet.summary()

loss = 'binary_crossentropy' if num_classes==1 else 'categorical_crossentropy'
senet.compile(optimizer=optimizers.Adam(0.04), loss=loss, metrics=['accuracy'])

# %%
# Training Network
epochs=100
batch_size=16

history=senet.fit(imgs_tr, labs_tr, epochs = epochs, batch_size=batch_size, 
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