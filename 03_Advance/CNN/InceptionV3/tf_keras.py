# %%
# Import Package
import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers, models, losses, optimizers, datasets, utils

# # %%
# # Data Prepare

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

def inception_module_A(x, filters_b1, filters_b2_1, filters_b2_2, 
                        filters_b3_1, filters_b3_2, filters_b3_3, filters_b4, name="Inception"):
    
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

def inception_module_B(x, filters_b1, filters_b2_1, filters_b2_2, filters_b2_3, 
                        filters_b3_1, filters_b3_2, filters_b3_3, filters_b3_4, filters_b3_5, 
                        filters_b4, name="Inception"):
    
    branch1 = layers.Conv2D(filters_b1, 1, activation="relu", name=name+"_branch_1_1x1")(x)
    
    branch2 = layers.Conv2D(filters_b2_1, 1, activation="relu", name=name+"_branch_2_1x1")(x)
    branch2 = layers.Conv2D(filters_b2_2, (1,7), padding="same", activation="relu", name=name+"_branch_2_1x7")(branch2)
    branch2 = layers.Conv2D(filters_b2_3, (7,1), padding="same", activation="relu", name=name+"_branch_2_7x1")(branch2)
    
    branch3 = layers.Conv2D(filters_b3_1, 1, activation="relu", name=name+"_branch_3_1x1")(x)
    branch3 = layers.Conv2D(filters_b3_2, (7,1), padding="same", activation="relu", name=name+"_branch_3_7x1_1")(branch3)
    branch3 = layers.Conv2D(filters_b3_3, (1,7), padding="same", activation="relu", name=name+"_branch_3_1x7_1")(branch3)
    branch3 = layers.Conv2D(filters_b3_4, (7,1), padding="same", activation="relu", name=name+"_branch_3_7x1_2")(branch3)
    branch3 = layers.Conv2D(filters_b3_5, (1,7), padding="same", activation="relu", name=name+"_branch_3_1x7_2")(branch3)

    branch4 = layers.AvgPool2D(pool_size=3, strides=1, padding="same", name=name+"_branch_4_pool")(x)
    branch4 = layers.Conv2D(filters_b4, 1, activation="relu", name=name+"_branch_4_1x1")(branch4)
    
    x = layers.Concatenate(name=name+"_Mixed")([branch1, branch2, branch3, branch4])
    
    return x

def inception_module_C(x, filters_b1, filters_b2_1, filters_b2_2, filters_b2_3, 
                        filters_b3_1, filters_b3_2, filters_b3_3, filters_b3_4, 
                        filters_b4, name="Inception"):
    
    branch1 = layers.Conv2D(filters_b1, 1, activation="relu", name=name+"_branch_1_1x1")(x)
    
    branch2 = layers.Conv2D(filters_b2_1, 1, activation="relu", name=name+"_branch_2_1x1")(x)
    branch2_1 = layers.Conv2D(filters_b2_2, (1,3), padding="same", activation="relu", name=name+"_branch_2_1x3")(branch2)
    branch2_2 = layers.Conv2D(filters_b2_3, (3,1), padding="same", activation="relu", name=name+"_branch_2_3x1")(branch2)
    branch2 = layers.Concatenate(name=name+"_branch_2")([branch2_1, branch2_2])

    branch3 = layers.Conv2D(filters_b3_1, 1, activation="relu", name=name+"_branch_3_1x1")(x)
    branch3 = layers.Conv2D(filters_b3_2, 3, padding="same", activation="relu", name=name+"_branch_3_7x1_1")(branch3)
    branch3_1 = layers.Conv2D(filters_b3_3, (1,3), padding="same", activation="relu", name=name+"_branch_3_1x7_1")(branch3)
    branch3_2 = layers.Conv2D(filters_b3_4, (3,1), padding="same", activation="relu", name=name+"_branch_3_7x1_2")(branch3)
    branch3 = layers.Concatenate(name=name+"_branch_3")([branch3_1, branch3_2])

    branch4 = layers.AvgPool2D(pool_size=3, strides=1, padding="same", name=name+"_branch_4_pool")(x)
    branch4 = layers.Conv2D(filters_b4, 1, activation="relu", name=name+"_branch_4_1x1")(branch4)
    
    x = layers.Concatenate(name=name+"_Mixed")([branch1, branch2, branch3, branch4])
    
    return x

def grid_reduction_1(x, filters_b1_1, filters_b2_1, filters_b2_2, filters_b2_3, name="Inception"):
    
    branch1 = layers.Conv2D(filters_b1_1, 3, strides=2, activation="relu", name=name+"_branch_1_3x3")(x)
    
    branch2 = layers.Conv2D(filters_b2_1, 1, activation="relu", name=name+"_branch_2_1x1")(x)
    branch2 = layers.Conv2D(filters_b2_2, 3, padding="same", activation="relu", name=name+"_branch_2_3x3_1")(branch2)
    branch2 = layers.Conv2D(filters_b2_3, 3, strides=2, activation="relu", name=name+"_branch_2_3x3_2")(branch2)
    
    branch3 = layers.MaxPool2D(pool_size=3, strides=2, name=name+"_branch_3_Pool")(x)
     
    x = layers.Concatenate(name=name+"_Mixed")([branch1, branch2, branch3])
    
    return x

def grid_reduction_2(x, filters_b1_1, filters_b1_2, 
                        filters_b2_1, filters_b2_2, filters_b2_3, filters_b2_4, name="Inception"):
    
    branch1 = layers.Conv2D(filters_b1_1, 1, activation="relu", name=name+"_branch_1_1x1")(x)
    branch1 = layers.Conv2D(filters_b1_2, 3, strides=2, activation="relu", name=name+"_branch_1_3x3")(branch1)
    
    branch2 = layers.Conv2D(filters_b2_1, 1, activation="relu", name=name+"_branch_2_1x1")(x)
    branch2 = layers.Conv2D(filters_b2_2, 3, padding="same", activation="relu", name=name+"_branch_2_1x7")(branch2)
    branch2 = layers.Conv2D(filters_b2_3, 3, padding="same", activation="relu", name=name+"_branch_2_7x1")(branch2)
    branch2 = layers.Conv2D(filters_b2_4, 3, strides=2, activation="relu", name=name+"_branch_2_3x3")(branch2)
    
    branch3 = layers.MaxPool2D(pool_size=3, strides=2, name=name+"_branch_3_Pool")(x)
     
    x = layers.Concatenate(name=name+"_Mixed")([branch1, branch2, branch3])
    
    return x

def auxiliary_classifier(x, num_classes=1, last_act="softmax", name="Auxiliary"):
    x = layers.AveragePooling2D(pool_size=5, strides=3, padding="same", name=name+"_Pool")(x)
    x = layers.Conv2D(128, 1, activation="relu", name=name+"_Conv")(x)
    x = layers.Flatten(name=name+"_Flatten")(x)
    x = layers.Dense(num_classes, last_act, name=name+"_Output")(x)
    return x

def build_inceptionv3(input_shape=(None, None, 3), num_classes=1, name="inception"):
    
    last_act = 'sigmoid' if num_classes==1 else 'softmax'

    input = layers.Input(shape=input_shape, name=name+"_input")

    x = layers.Conv2D(32, 3, strides=2, activation="relu", name=name+"_Conv_1")(input)
    x = layers.Conv2D(32, 3, activation="relu", name=name+"_Conv_2")(x)
    x = layers.Conv2D(64, 3, strides=1, padding="same", activation="relu", name=name+"_Conv_3")(x)
    
    x = layers.MaxPool2D(pool_size=3, strides=2, name=name+"_Pool_1")(x)
    x = layers.Conv2D(80, 1, activation="relu", name=name+"_Conv_4")(x)
    x = layers.Conv2D(192, 3, activation="relu", name=name+"_Conv_5")(x)
    
    x = layers.MaxPool2D(pool_size=3, strides=2, name=name+"_Pool_2")(x)
    x = inception_module_A(x, 64, 48, 64, 64, 96, 96, 64, name=name+"_Inception_5b")
    x = inception_module_A(x, 64, 48, 64, 64, 96, 96, 64, name=name+"_Inception_5c")
    x = inception_module_A(x, 64, 48, 64, 64, 96, 96, 64, name=name+"_Inception_5d")

    x = grid_reduction_1(x, 384, 64, 96, 96, name=name+"_Reduction_6a")    
    x = inception_module_B(x, 192, 128, 128, 192, 128, 128, 128, 128, 192, 192, name=name+"_Inception_6b")
    x = inception_module_B(x, 192, 160, 160, 192, 160, 160, 160, 160, 192, 192, name=name+"_Inception_6c")
    x = inception_module_B(x, 192, 160, 160, 192, 160, 160, 160, 160, 192, 192, name=name+"_Inception_6d")
    x = inception_module_B(x, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, name=name+"_Inception_6e")
    aux = auxiliary_classifier(x, num_classes, last_act, name=name+"_Aux")

    x = grid_reduction_2(x, 192, 320, 192, 192, 192, 192, name=name+"_Reduction_7a")
    x = inception_module_C(x, 320, 384, 384, 384, 448, 384, 384, 384, 192, name=name+"_Inception_7b")
    x = inception_module_C(x, 320, 384, 384, 384, 448, 384, 384, 384, 192, name=name+"_Inception_7c")

    x = layers.GlobalAveragePooling2D(name=name+"_GAP")(x)
    x = layers.Dropout(0.4, name=name+"_Dropout")(x)
    x = layers.Dense(num_classes, activation=last_act, name=name+"_Output")(x)

    return models.Model(input, [aux, x])

input_shape = imgs_tr.shape[1:]

inception = build_inceptionv3(input_shape=input_shape, num_classes=num_classes, name="InceptionV3")
inception.summary(line_length=120)

loss = 'binary_crossentropy' if num_classes==1 else 'categorical_crossentropy'
inception.compile(optimizer=optimizers.Adam(), loss=loss, metrics=['accuracy'])

# %%
# Training Network
epochs=100
batch_size=16

history=inception.fit(imgs_tr, [labs_tr, labs_tr], epochs = epochs, batch_size=batch_size, 
                      validation_data=[imgs_val, [labs_val, labs_val]])

plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.title("Loss graph")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Train', 'Validation'], loc='upper right')

plt.subplot(122)
plt.title("Acc graph")
plt.plot(history.history['InceptionV3_Output_acc'])
plt.plot(history.history['val_InceptionV3_Output_acc'])
plt.legend(['Train', 'Validation'], loc='upper right')

plt.show()