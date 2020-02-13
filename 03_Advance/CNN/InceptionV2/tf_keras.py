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

def inception_module_A(x, filters_b1, filters_b2_1, filters_b2_2, filters_b3_1, filters_b3_2, filters_b3_3, filters_b4, name="Inception"):
    
    branch1 = layers.Conv2D(filters_b1, 1, activation="relu", name=name+"_branch_1_1x1")(x)
    
    branch2 = layers.Conv2D(filters_b2_1, 1, activation="relu", name=name+"_branch_2_1x1")(x)
    branch2 = layers.Conv2D(filters_b2_2, 3, padding="same", activation="relu", name=name+"_branch_2_3x3")(branch2)
    
    branch3 = layers.Conv2D(filters_b3_1, 1, activation="relu", name=name+"_branch_3_1x1")(x)
    branch3 = layers.Conv2D(filters_b3_2, 3, padding="same", activation="relu", name=name+"_branch_3_3x3_1")(branch3)
    branch3 = layers.Conv2D(filters_b3_3, 3, padding="same", activation="relu", name=name+"_branch_3_3x3_2")(branch3)

    branch4 = layers.AvgPool2D(pool_size=3, strides=1, padding="same", name=name+"_branch_4_pool")(x)
    branch4 = layers.Conv2D(filters_b4, 1, activation="relu", name=name+"_branch_4_1x1")(branch4)
    
    x = layers.Concatenate(name=name+"_Mixed")([branch1, branch2, branch3, branch4])
    
    return x

def grid_reduction(x, filters_b1_1, filters_b1_2, filters_b2_1, filters_b2_2, filters_b2_3, name="Inception"):
    
    branch1 = layers.Conv2D(filters_b1_1, 1, activation="relu", name=name+"_branch_1_1x1")(x)
    branch1 = layers.Conv2D(filters_b1_2, 3, strides=2, padding="same", activation="relu", name=name+"_branch_1_3x3")(branch1)
    
    branch2 = layers.Conv2D(filters_b2_1, 1, activation="relu", name=name+"_branch_2_1x1")(x)
    branch2 = layers.Conv2D(filters_b2_2, 3, padding="same", activation="relu", name=name+"_branch_2_3x3_1")(branch2)
    branch2 = layers.Conv2D(filters_b2_3, 3, strides=2, padding="same", activation="relu", name=name+"_branch_2_3x3_2")(branch2)
    
    branch3 = layers.MaxPool2D(pool_size=3, strides=2, padding="same", name=name+"_branch_3_Pool")(x)
     
    x = layers.Concatenate(name=name+"_Mixed")([branch1, branch2, branch3])
    
    return x


def build_inceptionv2(input_shape=(None, None, 3), num_classes=1, name="inception"):
    
    last_act = 'sigmoid' if num_classes==1 else 'softmax'

    input = layers.Input(shape=input_shape, name=name+"_input")

    x = layers.Conv2D(64, 7, strides=2, padding="same", activation="relu", name=name+"_Conv_1")(input)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="same", name=name+"_Pool_1")(x)
    
    x = layers.Conv2D(64, 1, strides=1, padding="same", activation="relu", name=name+"_Conv_2")(x)
    x = layers.Conv2D(192, 3, strides=1, padding="same", activation="relu", name=name+"_Conv_3")(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="same", name=name+"_Pool_2")(x)
    
    x = inception_module_A(x, 64, 64, 64, 64, 96, 96, 32, name=name+"_Inception_3b")
    x = inception_module_A(x, 64, 64, 96, 64, 96, 96, 64, name=name+"_Inception_3c")

    x = grid_reduction(x, 128, 160, 64, 96, 96, name=name+"_Reduction_4a")    
    x = inception_module_A(x, 224, 64, 96, 96, 128, 128, 128, name=name+"_Inception_4b")
    x = inception_module_A(x, 224, 64, 96, 96, 128, 128, 128, name=name+"_Inception_4c")
    x = inception_module_A(x, 160, 128, 160, 128, 160, 160, 96, name=name+"_Inception_4d")
    x = inception_module_A(x, 96, 128, 192, 160, 192, 192, 96, name=name+"_Inception_4e")

    x = grid_reduction(x, 128, 192, 192, 256, 256, name=name+"_Reduction_5a")
    x = inception_module_A(x, 352, 192, 160, 224, 224, 224, 128, name=name+"_Inception_5b")
    x = inception_module_A(x, 352, 192, 320, 192, 224, 224, 128, name=name+"_Inception_5c")

    x = layers.GlobalAveragePooling2D(name=name+"_GAP")(x)
    x = layers.Dropout(0.4, name=name+"_Dropout")(x)
    x = layers.Dense(num_classes, activation=last_act, name=name+"_Output")(x)

    return models.Model(input, x)

input_shape = imgs_tr.shape[1:]

inception = build_inceptionv2(input_shape=input_shape, num_classes=num_classes, name="InceptionV2")
inception.summary()

loss = 'binary_crossentropy' if num_classes==1 else 'categorical_crossentropy'
inception.compile(optimizer=optimizers.Adam(), loss=loss, metrics=['accuracy'])

# %%
# Training Network
epochs=100
batch_size=16

history=inception.fit(imgs_tr, labs_tr, epochs = epochs, batch_size=batch_size, validation_data=[imgs_val, labs_val])

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