# %%
import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, datasets, utils

# %%
URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
path_to_zip  = tf.keras.utils.get_file('flower_photos.tgz', origin=URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'flower_photos')

category_list = [i for i in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, i)) ]
print(category_list)

num_classes = len(category_list)
img_size = 150
EPOCHS = 500
BATCH_SIZE = 16
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

train_ds = tf.data.Dataset.from_tensor_slices((imgs_tr, labs_tr)).shuffle(10000).batch(BATCH_SIZE)

test_ds = tf.data.Dataset.from_tensor_slices((imgs_val, labs_val)).shuffle(10000).batch(BATCH_SIZE)

print("Data Prepared!")

# %%
# Build Networks
class Conv_Block(models.Model):
    def __init__(self, num_filters, ksize, strides=(1, 1), padding='same', activation='relu', name='conv_block'):
        super(Conv_Block, self).__init__()
        self.conv = layers.Conv2D(num_filters, ksize, strides=strides, padding=padding, activation="linear", name=name+"_conv")
        self.bn = layers.BatchNormalization(name=name+"_bn")
        self.act = layers.Activation(activation, name=name+"_Act")
    
    def call(self, x, training=None):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.act(x)
        return x

class Residual_Block(models.Model):
    def __init__(self, num_filters, strides=(1, 1), activation='relu', use_branch=True, name='res_block'):
        super(Residual_Block, self).__init__()
        self.use_branch = use_branch
        if use_branch: 
            self.branch1 = Conv_Block(num_filters, 1, strides=strides, padding='valid', activation='linear', name=name+"_Branch1")

        self.branch2 = models.Sequential()
        self.branch2.add(Conv_Block(num_filters//4, 1, strides=strides, padding='valid', activation=activation, name=name+"_Branch2a"))
        self.branch2.add(Conv_Block(num_filters//4, 3, activation=activation, name=name+"_Branch2b"))
        self.branch2.add(Conv_Block(num_filters, 1, activation='linear', name=name+"_Branch2c"))
        self.add = layers.Add(name=name+"_Add")
        self.act = layers.Activation(activation, name=name+"_Act")
    
    def call(self, x, training=None):
        if self.use_branch:
            branch1 = self.branch1(x, training=training)
        else:
            branch1 = x

        branch2 = self.branch2(x, training=training)
        output = self.add([branch1, branch2])
        output = self.act(output)
        return output


class Build_ResNet(models.Model):
    def __init__(self, num_classes=10, num_layer = 50, name="Net" ):
        super(Build_ResNet, self).__init__()

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

        self.Stem = models.Sequential()
        self.Stem.add(layers.ZeroPadding2D((3, 3), name=name+"_pad"))
        self.Stem.add(Conv_Block(64, 7, (2, 2), 'valid', 'relu', name=name+"_stem"))
        self.Stem.add(layers.MaxPool2D(name=name+'_pool'))
        
        self.Main = models.Sequential()
        for idx, num_iter in enumerate(blocks_dict[num_layer]):
            for j in range(num_iter):
                if j==0:
                    self.Main.add(Residual_Block(num_channel_list[idx], activation='relu',  strides=(2, 2), name=name+"_res_"+block_name[idx]+str(j)))
                else:
                    self.Main.add(Residual_Block(num_channel_list[idx], activation='relu', use_branch=False, name=name+"_res_"+block_name[idx]+str(j)))

        self.Tail = models.Sequential()
        self.Tail.add(layers.GlobalAveragePooling2D(name=name+"_GAP"))
        self.Tail.add(layers.Dense(num_classes, activation=last_act, name=name+"_Output"))

    def call(self, x, training=None):
        x = self.Main(x, training=training)
        x = self.Stem(x, training=training)
        x = self.Tail(x, training=training)
        return x

num_layer = 50

model = Build_ResNet(num_classes=num_classes, num_layer=num_layer, name='resnet')

loss_object = losses.BinaryCrossentropy() if num_classes==1 else losses.CategoricalCrossentropy()

optimizer = optimizers.Adam()

print("Start Training")
for epoch in range(EPOCHS):
    for batch_x, batch_y in train_ds:
        with tf.GradientTape() as tape:
            predictions = model(batch_x, training=True)
            loss = loss_object(batch_y, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
    print("{:5}|{:10.6f}".format(epoch+1, loss))