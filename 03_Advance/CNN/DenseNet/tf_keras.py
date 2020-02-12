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

#%%
def Dense_Layer(input, growth_rate, name="Dense_Layer"):
    x = layers.BatchNormalization(name=name+"_BN_1")(input)
    x = layers.ReLU(name=name+"_Act_1")(x)
    x = layers.Conv2D(growth_rate*4, 1, name=name+"_Conv_1")(x)
    x = layers.BatchNormalization(name=name+"_BN_2")(x)
    x = layers.ReLU(name=name+"_Act_2")(x)
    x = layers.Conv2D(growth_rate, 3, padding='same', name=name+"_Conv_2")(x)
    x = layers.Concatenate(name=name+"_Concat")([input, x])
    return x

def Dense_Block(input, num_layer, name="Dense_Block"):
    x = Dense_Layer(input, 32, name=name+"_1")
    for i in range(2, num_layer+1):
        x = Dense_Layer(x, 32, name=name+"_%d"%i)
    return x

def Transition_Layer(input, reduction, name="Transition_Layer"):
    n_features = int(input.shape[-1])
    x = layers.BatchNormalization(name=name+"_BN_1")(input)
    x = layers.ReLU(name=name+"_Act_1")(x)
    x = layers.Conv2D(int(n_features*reduction), 1, name=name+"_Conv_1")(x)
    x = layers.AveragePooling2D(name=name+"_Pool")(x)
    return x


def build_densenet(input_shape=(None, None, 3), num_classes = 100, num_blocks=121, name = "DenseNet"):
    
    blocks_dict = {
        121: [6, 12, 24, 16],
        169: [6, 12, 32, 32], 
        201: [6, 12, 48, 32], 
        264: [6, 12, 64, 48]
    }
    
    assert num_blocks in  blocks_dict.keys(), "Number of layer must be in %s"%blocks_dict.keys()
    
    input = layers.Input(shape=input_shape, name=name+"_Input")
    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name=name+"_Stem_Pad_1")(input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name=name+"_Stem_Conv")(x)
    x = layers.BatchNormalization(name=name+'_Stem_BN')(x)
    x = layers.ReLU(name=name+"_Stem_Act")(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name+"_Stem_Pad_2")(x)
    x = layers.MaxPooling2D(3, strides=2, name=name+"_Stem_Pool")(x)

    x = Dense_Block(x, blocks_dict[num_blocks][0], name=name+"_Dense_Block_1")
    x = Transition_Layer(x, 0.5, name='Transition_1')
    x = Dense_Block(x, blocks_dict[num_blocks][1], name=name+"_Dense_Block_2")
    x = Transition_Layer(x, 0.5, name='Transition_2')
    x = Dense_Block(x, blocks_dict[num_blocks][2], name=name+"_Dense_Block_3")
    x = Transition_Layer(x, 0.5, name='Transition_3')
    x = Dense_Block(x, blocks_dict[num_blocks][3], name=name+"_Dense_Block_4")

    x = layers.BatchNormalization(name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)
    
    x = layers.GlobalAveragePooling2D(name=name+"_GAP")(x)
    x = layers.Dense(num_classes, activation='softmax', name=name+"_Output")(x)
    
    return models.Model(input, x, name=name)

num_blocks = 121
input_shape = imgs_tr.shape[1:]

dense = build_densenet(input_shape=input_shape, num_classes=num_classes, num_blocks=num_blocks, name = "DenseNet")

loss = 'binary_crossentropy' if num_classes==1 else 'categorical_crossentropy'
dense.compile(optimizer=optimizers.Adam(), loss=loss, metrics=['accuracy'])



# %%
# Training Network
epochs=100
batch_size=16

history=dense.fit(imgs_tr, labs_tr, epochs = epochs, batch_size=batch_size, validation_data=[imgs_val, labs_val])

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