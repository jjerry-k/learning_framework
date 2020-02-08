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
epochs=100
batch_size=16

history=vgg.fit(imgs_tr, labs_tr, epochs = epochs, batch_size=batch_size, validation_data=[imgs_val, labs_val])

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