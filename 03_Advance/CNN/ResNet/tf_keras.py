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
def conv_block(x, num_filters, ksize, strides=(1, 1), padding='same', activation='relu', name='conv_block'):
    output = layers.Conv2D(num_filters, ksize, strides=strides, padding=padding, activation="linear", name=name+"_conv")(x)
    output = layers.BatchNormalization(name=name+"_bn")(output)
    output = layers.Activation(activation, name=name+"_Act")(output)
    return output

def residual_block(x, num_filters, strides=(1, 1), activation='relu', use_branch=True, name='res_block'):
    
    if use_branch: 
        branch1 = conv_block(x, num_filters, 1, strides=strides, padding='valid', activation='linear', name=name+"_Branch1")
    else : 
        branch1 = x
        
    branch2 = conv_block(x, num_filters//4, 1, strides=strides, padding='valid', activation=activation, name=name+"_Branch2a")
    branch2 = conv_block(branch2, num_filters//4, 3, activation=activation, name=name+"_Branch2b")
    branch2 = conv_block(branch2, num_filters, 1, activation='linear', name=name+"_Branch2c")

    output = layers.Add(name=name+"_Add")([branch1, branch2])
    output = layers.Activation(activation, name=name+"_Act")(output)
    return output

def build_resnet(input_shape=(None, None, 3, ), num_classes=10, num_layer = 50, name="Net"): 

    
    blocks_dict = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3], 
        152: [3, 8, 36, 3]
    }

    num_channel_list = [256, 512, 1024, 2048]
    block_name = ['a', 'b', 'c', 'd']
    assert num_layer in  blocks_dict.keys(), "Number of layer must be in %s"%blocks_dict.keys()
    
    name = name+str(num_layer)

    last_act = 'sigmoid' if num_classes==1 else 'softmax'

    _input = layers.Input(shape=input_shape, name=name+"_input")

    x = layers.ZeroPadding2D((3, 3), name=name+"_pad")(_input)
    x = conv_block(x, 64, 7, (2, 2), 'valid', 'relu', name=name+"_stem")
    x = layers.MaxPool2D(name=name+'_pool')(x)
    
    for idx, num_iter in enumerate(blocks_dict[num_layer]):
        for j in range(num_iter):
            if j==0:
                x = residual_block(x, num_channel_list[idx], activation='relu',  strides=(2, 2), name=name+"_res_"+block_name[idx]+str(j))
            else:
                x = residual_block(x, num_channel_list[idx], activation='relu', use_branch=False, name=name+"_res_"+block_name[idx]+str(j))

    x = layers.GlobalAveragePooling2D(name=name+"_GAP")(x)
    x = layers.Dense(num_classes, activation=last_act, name=name+"_Output")(x)
    return models.Model(_input, x, name=name)

num_layer = 50
input_shape = imgs_tr.shape[1:]

resnet = build_resnet(input_shape=input_shape, num_classes=num_classes, num_layer=num_layer, name="ResNet")
resnet.summary()


loss = 'binary_crossentropy' if num_classes==1 else 'categorical_crossentropy'
resnet.compile(optimizer=optimizers.Adam(), loss=loss, metrics=['accuracy'])

# %%
# Training Network
epochs=100
batch_size=16

history=resnet.fit(imgs_tr, labs_tr, epochs = epochs, batch_size=batch_size, validation_data=[imgs_val, labs_val])

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