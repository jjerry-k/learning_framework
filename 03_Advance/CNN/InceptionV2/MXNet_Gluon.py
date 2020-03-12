# %%
import os, tarfile
import cv2 as cv
import numpy as np
import mxnet as mx
from tqdm import tqdm
from mxnet import io, nd, gluon, init, autograd
from mxnet.gluon.data.vision import datasets
from mxnet.gluon import nn, data, utils
from matplotlib import pyplot as plt
from multiprocessing import cpu_count
CPU_COUNT = cpu_count()
print("Package Loaded!")

# %%
# Data Prepare

'''
Download Flower classification dataset
'''

SAVE_PATH = "../../../data"
URL = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
file_name = URL.split("/")[-1]
os.makedirs(SAVE_PATH, exist_ok=True)
data = utils.download(URL, SAVE_PATH)
PATH = os.path.join(SAVE_PATH, "flower_photos")
with tarfile.open(os.path.join(SAVE_PATH, file_name)) as tf:
    tf.extractall(SAVE_PATH)
    
category_list = [i for i in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, i)) ]
print(category_list, '\n')

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
    print("================================\n")

    imgs = [read_img(os.path.join(path, img),img_size) for img in imgs_list]
    labs = [i]*len(imgs_list)

    imgs_tr += imgs[ratio:]
    labs_tr += labs[ratio:]
    
    imgs_val += imgs[:ratio]
    labs_val += labs[:ratio]

imgs_tr = np.array(imgs_tr)/255.
labs_tr = np.array(labs_tr)

imgs_val = np.array(imgs_val)/255.
labs_val = np.array(labs_val)

print(imgs_tr.shape, labs_tr.shape)
print(imgs_val.shape, labs_val.shape)

# %%
# Build network

class ReLU6(nn.Block):
    def __init__(self):
        super(ReLU6, self).__init__()

    def forward(self, x):
        return nd.clip(x, 0, 6)

class Conv_Block(nn.Block):
    def __init__(self, output_feature, ksize=3, strides=1, padding=1):
        super(Conv_Block, self).__init__()

        self.block = nn.Sequential()
        self.block.add(nn.Conv2D(output_feature, ksize, strides, padding))
        self.block.add(nn.BatchNorm())
        self.block.add(ReLU6())
        

    def forward(self, x):
        return self.block(x)

class Depthwise_Separable_Block(nn.Block):
    def __init__(self, input_feature, output_feature, ksize=3, strides=1, padding=1, alpha=1):
        super(Depthwise_Separable_Block, self).__init__()
        
        self.block = nn.Sequential()
        self.block.add(nn.Conv2D(input_feature, ksize, strides, padding, groups=input_feature))
        self.block.add(nn.BatchNorm())
        self.block.add(ReLU6())
        self.block.add(nn.Conv2D(output_feature, 1))
        self.block.add(nn.BatchNorm())
        self.block.add(ReLU6())

    def forward(self, x):
        return self.block(x)

class Inverted_Residual_Block(nn.Block):
    def __init__(self, input_feature, expansion, output_feature, strides=1, alpha=1):
        super(Inverted_Residual_Block, self).__init__()
        
        self.stride = strides

        self.intermediate_featrue = int(input_feature*expansion)
        
        self.output_feature = output_feature

        self.alpha = alpha

        self.block = nn.Sequential()
        self.block.add(Conv_Block(self.intermediate_featrue, 1, 1, 0))
        self.block.add(Depthwise_Separable_Block(self.intermediate_featrue, self.output_feature, 3, strides, 1, self.alpha))
    
    def forward(self, x):
        output = self.block(x)
        if self.stride==1 and self.intermediate_featrue == int(self.output_feature*self.alpha):
            return x + output
        return output

class Build_MobileNetV2(nn.Block):
    def __init__(self, num_classes=1000, alpha=1):
        super(Build_MobileNetV2, self).__init__()

        self.Stem = nn.Sequential()
        self.Stem.add(Conv_Block(32, 3, 2, 1))

        self.Main_Block = nn.Sequential()
        self.Main_Block.add(Inverted_Residual_Block(32, 1, 16, 1, 1))

        self.Main_Block.add(Inverted_Residual_Block(16, 6, 24, 2, 1))
        self.Main_Block.add(Inverted_Residual_Block(24, 6, 24, 1, 1))

        self.Main_Block.add(Inverted_Residual_Block(24, 6, 32, 2, 1))
        self.Main_Block.add(Inverted_Residual_Block(32, 6, 32, 1, 1))
        self.Main_Block.add(Inverted_Residual_Block(32, 6, 32, 1, 1))

        self.Main_Block.add(Inverted_Residual_Block(32, 6, 64, 2, 1))
        self.Main_Block.add(Inverted_Residual_Block(64, 6, 64, 1, 1))
        self.Main_Block.add(Inverted_Residual_Block(64, 6, 64, 1, 1))
        self.Main_Block.add(Inverted_Residual_Block(64, 6, 64, 1, 1))

        self.Main_Block.add(Inverted_Residual_Block(64, 6, 96, 1, 1))
        self.Main_Block.add(Inverted_Residual_Block(96, 6, 96, 1, 1))
        self.Main_Block.add(Inverted_Residual_Block(96, 6, 96, 1, 1))
        
        self.Main_Block.add(Inverted_Residual_Block(96, 6, 160, 2, 1))
        self.Main_Block.add(Inverted_Residual_Block(160, 6, 160, 1, 1))
        self.Main_Block.add(Inverted_Residual_Block(160, 6, 160, 1, 1))

        self.Main_Block.add(Inverted_Residual_Block(160, 6, 320, 1, 1))

        self.Exit = nn.Sequential()
        self.Exit.add(Conv_Block(1280, 1, 1, 0))
        
        self.Classifier = nn.Sequential()
        self.Classifier.add(nn.GlobalAvgPool2D())
        self.Classifier.add(nn.Flatten())
        self.Classifier.add(nn.Dense(num_classes))

    def forward(self, x):
        x = self.Stem(x)
        x = self.Main_Block(x)
        x = self.Exit(x)
        x = self.Classifier(x)
        return x
    
mobilenetv2 = Build_MobileNetV2(num_classes=5)

gpus = mx.test_utils.list_gpus()
ctx = [mx.gpu(i) for i in gpus] if gpus else [mx.cpu()]

mobilenetv2.initialize(ctx=ctx[0])

cross_entropy = gluon.loss.SoftmaxCELoss()
trainer = gluon.Trainer(mobilenetv2.collect_params(), 'adam', {'learning_rate': 0.001})
print("Setting Done!")

# %%

epochs=100
batch_size=32

class DataIterLoader():
    def __init__(self, X, Y, batch_size=1, shuffle=True, ctx=mx.cpu()):
        self.data_iter = io.NDArrayIter(data=gluon.utils.split_and_load(np.transpose(X, [0, 3, 1, 2]), ctx_list=ctx, batch_axis=0), 
                                        label=gluon.utils.split_and_load(Y, ctx_list=ctx, batch_axis=0), 
                                        batch_size=batch_size, shuffle=shuffle)
        self.len = len(X)

    def __iter__(self):
        self.data_iter.reset()
        return self

    def __next__(self):
        batch = self.data_iter.__next__()
        assert len(batch.data) == len(batch.label) == 1
        data = batch.data[0]
        label = batch.label[0]
        return data, label


train_loader = DataIterLoader(imgs_tr, labs_tr, batch_size, ctx=ctx)
validation_loader = DataIterLoader(imgs_val, labs_val, batch_size, ctx=ctx)

print("\nStart Training!")

for epoch in range(epochs):
    train_loss, train_acc, valid_loss, valid_acc = 0., 0., 0., 0.
    #tic = time.time()
    # forward + backward
    for step, (batch_img, batch_lab) in enumerate(train_loader):
        
        with autograd.record():
            output = mobilenetv2(batch_img)
            loss = cross_entropy(output, batch_lab)

        loss.backward()
        # update parameters
        trainer.step(batch_size)
        correct = np.argmax(output.asnumpy(), axis = 1)
        acc = np.mean(correct == batch_lab.asnumpy())
        train_loss += loss.mean().asnumpy()[0]
        train_acc += acc

    for idx, (val_img, val_lab) in enumerate(validation_loader):
        output = mobilenetv2(val_img)
        val_loss = cross_entropy(output, val_lab)
        correct = np.argmax(output.asnumpy(), axis = 1)
        acc = np.mean(correct == val_lab.asnumpy())
        valid_loss += loss.mean().asnumpy()[0]
        valid_acc += acc
        
    print(f"Epoch : {epoch+1}, loss : {train_loss/(step+1):.4f}, acc : {train_acc/(step+1):.4f}, \
            val_loss : {valid_loss/(idx+1):.4f},  val_acc : {valid_acc/(idx+1):.4f}")