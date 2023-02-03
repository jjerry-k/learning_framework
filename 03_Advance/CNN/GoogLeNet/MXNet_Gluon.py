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
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(tf, SAVE_PATH)
    
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

class Inception_Module(gluon.Block):
    def __init__(self, filters_b1, filters_b2_1, filters_b2_2, 
                filters_b3_1, filters_b3_2, filters_b4):
        super(Inception_Module, self).__init__()

        self.branch1 = nn.Sequential()
        self.branch1.add(nn.Conv2D(filters_b1, 1, activation='relu'))

        self.branch2 = nn.Sequential()
        self.branch2.add(nn.Conv2D(filters_b2_1, 1, activation='relu'))
        self.branch2.add(nn.Conv2D(filters_b2_2, 3, 1, 1, activation='relu'))

        self.branch3 = nn.Sequential()
        self.branch3.add(nn.Conv2D(filters_b3_1, 1, activation='relu'))
        self.branch3.add(nn.Conv2D(filters_b3_2, 5, 1, 2, activation='relu'))

        self.branch4 = nn.Sequential()
        self.branch4.add(nn.MaxPool2D(3, 1, 1))
        self.branch4.add(nn.Conv2D(filters_b4, 1, activation='relu'))

    def forward(self, x):
        return nd.concat(self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x), dim=1)

class Auxiliary_Classifier(gluon.Block):
    def __init__(self, num_classes):
        super(Auxiliary_Classifier, self).__init__()
        self.block = nn.Sequential()
        self.block.add(nn.AvgPool2D(5, 3, 1))
        self.block.add(nn.Conv2D(128, 1, activation='relu'))
        self.block.add(nn.GlobalAvgPool2D())
        self.block.add(nn.Flatten())
        self.block.add(nn.Dense(num_classes))
    
    def forward(self, x):
        return self.block(x)

class Build_GoogLeNet(gluon.Block):
    def __init__(self, num_classes=1000):
        super(Build_GoogLeNet, self).__init__()

        self.stem_1 = nn.Sequential()
        self.stem_1.add(nn.Conv2D(64, (7, 7), (2, 2), (3, 3), activation='relu'))
        self.stem_1.add(nn.MaxPool2D((3, 3), (2, 2)))
        
        self.stem_2 = nn.Sequential()
        self.stem_2.add(nn.Conv2D(64, (1, 1), activation='relu'))
        self.stem_2.add(nn.Conv2D(192, (3, 3), (1, 1), (1, 1), activation='relu'))
        
        self.pool = nn.MaxPool2D((3, 3), (2, 2))
        
        self.inception1 = Inception_Module(64, 96, 128, 16, 32, 32)
        self.inception2 = Inception_Module(128, 128, 192, 32, 96, 64)
        self.inception3 = Inception_Module(192, 96, 208, 16, 48, 64)
        self.aux1 = Auxiliary_Classifier(num_classes)
        self.inception4 = Inception_Module(160, 112, 224, 24, 64, 64)
        self.inception5 = Inception_Module(128, 128, 256, 24, 64, 64)
        self.inception6 = Inception_Module(112, 144, 288, 32, 64, 64)
        self.aux2 = Auxiliary_Classifier(num_classes)
        self.inception7 = Inception_Module(256, 160, 320, 32, 128, 128)
        self.inception8 = Inception_Module(256, 160, 320, 32, 128, 128)
        self.inception9 = Inception_Module(384, 192, 384, 48, 128, 128)
        
        self.classifier = nn.Sequential()
        self.classifier.add(nn.GlobalAvgPool2D())
        self.classifier.add(nn.Flatten())
        self.classifier.add(nn.Dropout(0.4))
        self.classifier.add(nn.Dense(num_classes))
        
    def forward(self, x):
        x = self.stem_1(x)
        x = nd.LRN(x, nsize=63)
        x = self.stem_2(x)
        x = nd.LRN(x, nsize=191)
        x = self.pool(x)

        x = self.inception1(x)
        x = self.inception2(x)
        x = self.pool(x)
        
        x = self.inception3(x)
        aux1 = self.aux1(x)

        x = self.inception4(x)
        x = self.inception5(x)
        x = self.inception6(x)
        aux2 = self.aux2(x)
        
        x = self.inception7(x)
        x = self.pool(x)

        x = self.inception8(x)
        x = self.inception9(x)

        x = self.classifier(x)
        return x, aux1, aux2
    
googlenet = Build_GoogLeNet(num_classes=5)

gpus = mx.test_utils.list_gpus()
ctx = [mx.gpu(i) for i in gpus] if gpus else [mx.cpu()]

googlenet.initialize(ctx=ctx[0])

cross_entropy = gluon.loss.SoftmaxCELoss()
trainer = gluon.Trainer(googlenet.collect_params(), 'adam', {'learning_rate': 0.001})
print("Setting Done!")

# %%

epochs=100
batch_size=64

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
            output = googlenet(batch_img)

            loss_final = cross_entropy(output[0], batch_lab)
            loss_aux_1 = cross_entropy(output[1], batch_lab)
            loss_aux_2 = cross_entropy(output[2], batch_lab)

            loss = loss_final + 0.4*loss_aux_1 + 0.4*loss_aux_2

        loss.backward()
        # update parameters
        trainer.step(batch_size)
        correct = np.argmax(output[0].asnumpy(), axis = 1)
        acc = np.mean(correct == batch_lab.asnumpy())
        train_loss += loss.mean().asnumpy()[0]
        train_acc += acc

    for idx, (val_img, val_lab) in enumerate(validation_loader):
        output = googlenet(val_img)
        val_loss_final = cross_entropy(output[0], val_lab)
        val_loss_aux_1 = cross_entropy(output[1], val_lab)
        val_loss_aux_2 = cross_entropy(output[2], val_lab)
        val_loss = val_loss_final + 0.4*val_loss_aux_1 + 0.4*val_loss_aux_2
        correct = np.argmax(output[0].asnumpy(), axis = 1)
        acc = np.mean(correct == val_lab.asnumpy())
        valid_loss += loss.mean().asnumpy()[0]
        valid_acc += acc
        
    print(f"Epoch : {epoch+1}, loss : {train_loss/(step+1):.4f}, acc : {train_acc/(step+1):.4f}, \
            val_loss : {valid_loss/(idx+1):.4f},  val_acc : {valid_acc/(idx+1):.4f}")