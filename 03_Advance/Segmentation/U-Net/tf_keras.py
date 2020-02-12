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

URL = 'https://www.robots.ox.ac.uk/~vgg/data/bicos/data/horses.tar'
path_to_zip  = tf.keras.utils.get_file('horses.tar', origin=URL, extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'horses')

PATH_img = os.path.join(PATH, 'jpg')
PATH_lab = os.path.join(PATH, 'gt')

img_list = sorted(os.listdir(PATH_img))
lab_list = sorted(os.listdir(PATH_lab))


img_size = 224

def read_img(path, img_size, mode='rgb'):
    mode_dict = {"rgb":cv.COLOR_BGR2RGB, 
            "gray":cv.COLOR_BGR2GRAY}
    
    img = cv.imread(path)
    img = cv.cvtColor(img, mode_dict[mode])
    img = cv.resize(img, (img_size, img_size))
    return img

print("Total images : %d"%(len(img_list)))
print("Total labels : %d"%(len(lab_list)))

imgs = np.array([read_img(os.path.join(PATH_img, i), img_size, 'rgb') for i in img_list])/255.
labs = np.greater(np.array([read_img(os.path.join(PATH_lab, i), img_size, 'gray') for i in lab_list])/255., 0.5)[..., np.newaxis]

ratio = int(len(img_list)*0.05)

imgs_tr = imgs[ratio:]
labs_tr = labs[ratio:]

imgs_val = imgs[:ratio]
labs_val = labs[:ratio]

print("Training images : %d"%(len(imgs_tr)))
print("Training labels : %d"%(len(labs_tr)))

print("Validation images : %d"%(len(imgs_val)))
print("Validation labels : %d"%(len(labs_val)))

print(imgs_tr.shape, labs_tr.shape)
print(imgs_val.shape, labs_val.shape)

# %%
# Plot Data

idxs = np.random.choice(len(imgs), 8, replace=False)

plt.figure(figsize=(24, 6))
for i in range(len(idxs)):
    plt.subplot(2, 8, i+1)
    plt.imshow(imgs[idxs[i]])
    plt.axis("off")
    plt.subplot(2, 8, i+1+8)
    plt.imshow(labs[idxs[i], ..., 0], cmap='gray')
    plt.axis("off")
plt.tight_layout()
plt.show()

# %%
# Build Network

def build_unet(input_shape= (None, None, 1), num_classes = 1, name='unet'):
    
    last_act = 'sigmoid' if num_classes==1 else 'softmax'

    input_layer = layers.Input(shape=input_shape, name=name+"_input")

    encoder1 = layers.Conv2D(64, 3, strides=1, padding='same', activation='relu', name=name+"_en1_conv1")(input_layer)
    encoder1 = layers.Conv2D(64, 3, strides=1, padding='same', activation='relu', name=name+"_en1_conv2")(encoder1)

    encoder2 = layers.MaxPooling2D(name=name+"_en2_pool")(encoder1)
    encoder2 = layers.Conv2D(128, 3, strides=1, padding='same', activation='relu', name=name+"_en2_conv1")(encoder2)
    encoder2 = layers.Conv2D(128, 3, strides=1, padding='same', activation='relu', name=name+"_en2_conv2")(encoder2)

    encoder3 = layers.MaxPooling2D(name=name+"_en3_pool")(encoder2)
    encoder3 = layers.Conv2D(256, 3, strides=1, padding='same', activation='relu', name=name+"_en3_conv1")(encoder3)
    encoder3 = layers.Conv2D(256, 3, strides=1, padding='same', activation='relu', name=name+"_en3_conv2")(encoder3)

    encoder4 = layers.MaxPooling2D(name=name+"_en4_pool")(encoder3)
    encoder4 = layers.Conv2D(512, 3, strides=1, padding='same', activation='relu', name=name+"_en4_conv1")(encoder4)
    encoder4 = layers.Conv2D(512, 3, strides=1, padding='same', activation='relu', name=name+"_en4_conv2")(encoder4)

    encoder5 = layers.MaxPooling2D(name=name+"_en5_pool")(encoder4)
    encoder5 = layers.Conv2D(1024, 3, strides=1, padding='same', activation='relu', name=name+"_en5_conv1")(encoder5)
    encoder5 = layers.Conv2D(1024, 3, strides=1, padding='same', activation='relu', name=name+"_en5_conv2")(encoder5)

    decoder4 = layers.Conv2DTranspose(512, 2, strides=2, padding='same', activation='relu', name=name+"_de4_upconv")(encoder5)
    decoder4 = layers.Concatenate(axis=-1, name=name+"_de4_concat")([encoder4, decoder4])
    decoder4 = layers.Conv2D(512, 3, strides=1, padding='same', activation='relu', name=name+"_de4_conv1")(decoder4)
    decoder4 = layers.Conv2D(512, 3, strides=1, padding='same', activation='relu', name=name+"_de4_conv2")(decoder4)

    decoder3 = layers.Conv2DTranspose(256, 2, strides=2, padding='same', activation='relu', name=name+"_de3_upconv")(decoder4)
    decoder3 = layers.Concatenate(axis=-1, name=name+"_de3_concat")([encoder3, decoder3])
    decoder3 = layers.Conv2D(256, 3, strides=1, padding='same', activation='relu', name=name+"_de3_conv1")(decoder3)
    decoder3 = layers.Conv2D(256, 3, strides=1, padding='same', activation='relu', name=name+"_de3_conv2")(decoder3)

    decoder2 = layers.Conv2DTranspose(128, 2, strides=2, padding='same', activation='relu', name=name+"_de2_upconv")(decoder3)
    decoder2 = layers.Concatenate(axis=-1, name=name+"_de2_concat")([encoder2, decoder2])
    decoder2 = layers.Conv2D(128, 3, strides=1, padding='same', activation='relu', name=name+"_de2_conv1")(decoder2)
    decoder2 = layers.Conv2D(128, 3, strides=1, padding='same', activation='relu', name=name+"_de2_conv2")(decoder2)

    decoder1 = layers.Conv2DTranspose(64, 2, strides=2, padding='same', activation='relu', name=name+"_de1_upconv")(decoder2)
    decoder1 = layers.Concatenate(axis=-1, name=name+"_de1_concat")([encoder1, decoder1])
    decoder1 = layers.Conv2D(64, 3, strides=1, padding='same', activation='relu', name=name+"_de1_conv1")(decoder1)
    decoder1 = layers.Conv2D(64, 3, strides=1, padding='same', activation='relu', name=name+"_de1_conv2")(decoder1)

    output = layers.Conv2D(num_classes, 1, strides=1, padding='same', activation=last_act, name=name+"_prediction")(decoder1)

    return models.Model(inputs=input_layer, outputs=output, name=name)

def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2))
    return tf.reduce_mean(1 - numerator / denominator, axis=-1)

input_shape = imgs_tr.shape[1:]
num_classes = 1

unet = build_unet(input_shape=input_shape, num_classes=num_classes)
unet.summary()

loss = 'binary_crossentropy' if num_classes==1 else 'categorical_crossentropy'
# loss = dice_loss
unet.compile(optimizer=optimizers.Adam(), loss=loss)

# %%
# Training Network

epochs=10
batch_size=16

history=unet.fit(imgs_tr, labs_tr, epochs = epochs, batch_size=batch_size, validation_data=[imgs_val, labs_val])

plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.title("Loss graph")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Train', 'Validation'], loc='upper right')

# %%
# Test Network

prediction = unet.predict(imgs_val)

idxs = np.random.choice(len(imgs_val), 8, replace=False)

plt.figure(figsize=(24, 6))
for i in range(len(idxs)):
    plt.subplot(2, 8, i+1)
    plt.imshow(prediction[idxs[i], ..., 0], cmap='gray')
    plt.axis("off")
    plt.subplot(2, 8, i+1+8)
    plt.imshow(labs_val[idxs[i], ..., 0], cmap='gray')
    plt.axis("off")
plt.tight_layout()
plt.show()