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

def middle_flow(input, name="middle_flow"):
    x = layers.ReLU(name=name+"_Act_1")(input)
    x = layers.SeparableConv2D(728, 3, padding='same', name=name+"_Separable_1")(x)
    x = layers.BatchNormalization(name=name+"_BN_1")(x)
    x = layers.ReLU(name=name+"_Act_2")(x)
    x = layers.SeparableConv2D(728, 3, padding='same', name=name+"_Separable_2")(x)
    x = layers.BatchNormalization(name=name+"_BN_2")(x)
    x = layers.ReLU(name=name+"_Act_3")(x)
    x = layers.SeparableConv2D(728, 3, padding='same', name=name+"_Separable_3")(x)
    x = layers.BatchNormalization(name=name+"_BN_3")(x)
    x = layers.Add(name=name+"_Add")([input, x])
    return x

def build_xception(input_shape=(None, None, 3), num_classes=1, name='xception'):
    
    last_act = 'sigmoid' if num_classes==1 else 'softmax'

    input = layers.Input(shape=input_shape, name=name+"_input")

    x = layers.Conv2D(32, 3, strides=2, name=name+"_Stem_Conv_1")(input)
    x = layers.BatchNormalization(name=name+"_Stem_BN_1")(x)
    x = layers.ReLU(name=name+"_Stem_Act_1")(x)

    x = layers.Conv2D(64, 3, name=name+"_Stem_Conv_2")(x)
    x = layers.BatchNormalization(name=name+"_Stem_BN_2")(x)
    x = layers.ReLU(name=name+"_Stem_Act_2")(x)

    identity = layers.Conv2D(128, 1, strides=2, padding='same', name=name+"_Entry_Identity_Conv_1")(x)
    identity = layers.BatchNormalization(name=name+"_Entry_Identity_BN_1")(identity)

    x = layers.SeparableConv2D(128, 3, padding='same', name=name+"_Entry_Separable_1")(x)
    x = layers.BatchNormalization(name=name+"_Entry_BN_1")(x)
    x = layers.ReLU(name=name+"_Entry_Act_1")(x)
    
    x = layers.SeparableConv2D(128, 3, padding='same', name=name+"_Entry_Separable_2")(x)
    x = layers.BatchNormalization(name=name+"_Entry_BN_2")(x)
    
    x = layers.MaxPooling2D(3, strides=2, padding='same', name=name+"_Entry_Pool_1")(x)
    
    x = layers.Add(name=name+"_Entry_Add_1")([identity, x])

    identity = layers.Conv2D(256, 1, strides=2, padding='same', name=name+"_Entry_Identity_Conv_2")(x)
    identity = layers.BatchNormalization(name=name+"_Entry_Identity_BN_2")(identity)

    x = layers.ReLU(name=name+"_Entry_Act_2")(x)
    x = layers.SeparableConv2D(256, 3, padding='same', name=name+"_Entry_Separable_3")(x)
    x = layers.BatchNormalization(name=name+"_Entry_BN_3")(x)

    x = layers.ReLU(name=name+"_Entry_Act_3")(x)
    x = layers.SeparableConv2D(256, 3, padding='same', name=name+"_Entry_Separable_4")(x)
    x = layers.BatchNormalization(name=name+"_Entry_BN_4")(x)

    x = layers.MaxPooling2D(3, strides=2, padding='same', name=name+"_Entry_Pool_2")(x)
    
    x = layers.Add(name=name+"_Entry_Add_2")([identity, x])

    identity = layers.Conv2D(728, 1, strides=2, padding='same', name=name+"_Entry_Identity_Conv_3")(x)
    identity = layers.BatchNormalization(name=name+"_Entry_Identity_BN_3")(identity)

    x = layers.ReLU(name=name+"_Entry_Act_4")(x)
    x = layers.SeparableConv2D(728, 3, padding='same', name=name+"_Entry_Separable_5")(x)
    x = layers.BatchNormalization(name=name+"_Entry_BN_5")(x)

    x = layers.ReLU(name=name+"_Entry_Act_5")(x)
    x = layers.SeparableConv2D(728, 3, padding='same', name=name+"_Entry_Separable_6")(x)
    x = layers.BatchNormalization(name=name+"_Entry_BN_6")(x)

    x = layers.MaxPooling2D(3, strides=2, padding='same', name=name+"_Entry_Pool_3")(x)
    
    x = layers.Add(name=name+"_Entry_Add_3")([identity, x])

    for i in range(8):
        x = middle_flow(x, name=name+"_Middle_%d"%(i+1))
    

    identity = layers.Conv2D(1024, 1, strides=2, padding='same', name=name+"_Exit_Identity_Conv_1")(x)
    identity = layers.BatchNormalization(name=name+"_Exit_Identity_BN_1")(identity)

    x = layers.ReLU(name=name+"_Exit_Act_1")(x)
    x = layers.SeparableConv2D(728, 3, padding='same', name=name+"_Exit_Separable_1")(x)
    x = layers.BatchNormalization(name=name+"_Exit_BN_1")(x)

    x = layers.ReLU(name=name+"_Exit_Act_2")(x)
    x = layers.SeparableConv2D(1024, 3, padding='same', name=name+"_Exit_Separable_2")(x)
    x = layers.BatchNormalization(name=name+"_Exit_BN_2")(x)

    x = layers.MaxPooling2D(3, strides=2, padding='same', name=name+"_Exit_Pool_1")(x)
    
    x = layers.Add(name=name+"_Exit_Add")([identity, x])

    x = layers.SeparableConv2D(1536, 3, padding='same', name=name+"_Exit_Separable_3")(x)
    x = layers.BatchNormalization(name=name+"_Exit_BN_3")(x)
    x = layers.ReLU(name=name+"_Exit_Act_3")(x)

    x = layers.SeparableConv2D(2048, 3, padding='same', name=name+"_Exit_Separable_4")(x)
    x = layers.BatchNormalization(name=name+"_Exit_BN_4")(x)
    x = layers.ReLU(name=name+"_Exit_Act_4")(x)

    x = layers.GlobalAveragePooling2D(name=name+"_GAP")(x)

    x = layers.Dense(num_classes, activation=last_act, name=name+"_Output")(x)

    return models.Model(input, x)

input_shape = imgs_tr.shape[1:]

xception = build_xception(input_shape=input_shape, num_classes=num_classes, name="Xception")
xception.summary()

loss = 'binary_crossentropy' if num_classes==1 else 'categorical_crossentropy'
xception.compile(optimizer=optimizers.Adam(), loss=loss, metrics=['accuracy'])

# %%
# Training Network
epochs=100
batch_size=16

history=xception.fit(imgs_tr, labs_tr, epochs = epochs, batch_size=batch_size, validation_data=[imgs_val, labs_val])

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