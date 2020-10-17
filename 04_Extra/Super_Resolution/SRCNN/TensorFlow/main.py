import os
import numpy as np
import tensorflow as tf

from tensorflow import image as tfi
from tensorflow.keras import models, layers, losses, metrics, optimizers, callbacks
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.python.ops.array_ops import newaxis
from model import *

# Data Loader
ROOT = "../../datasets"
data = "BSR/BSDS500/data/images/train"
data_path = os.path.join(ROOT, data)
img_list = sorted(os.listdir(data_path))
val_ratio = 0.2
n_imgs = len(img_list)
input_size = 33
output_size = 21
batch_size = 8

def image_reader(root, img_list, input_size, output_size, scale=3):
    padding = (input_size - output_size)//2
    while 1:
        for img in img_list:
            # =================
            # Read & Preprocess
            # =================
            img = np.array(load_img(os.path.join(root, img), grayscale=True))[..., np.newaxis]
            h, w, _ = img.shape
            h, w = h//3 * h, w//3 * w
            img = img[:h, :w]/255.
            lab = img.copy()

            # Downsampling
            img = tf.image.resize(img, (h//scale, w//scale), method='bicubic')
            # Upsampling
            img = tf.image.resize(img, (h, w), method='bicubic')

            for row in range(0, h-input_size+1, 14):
                for col in range(0, w-input_size+1, 14):
                    x = img[row:row+input_size, col:col+input_size]
                    y = lab[row+padding:row+padding+output_size, col+padding:col+padding+output_size]
                    yield x, y

train_gen = image_reader(data_path, img_list[int(val_ratio*n_imgs):], input_size, output_size)
train_ds = tf.data.Dataset.from_generator(lambda: train_gen, 
                                            output_types = (tf.float64, tf.float64)
                                            ).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).repeat()

val_gen = image_reader(data_path, img_list[:int(val_ratio*n_imgs)], input_size, output_size)
val_ds = tf.data.Dataset.from_generator(lambda: val_gen, 
                                            output_types = (tf.float64, tf.float64)
                                            ).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).repeat()

# Defile psnr, ssim for metrics
class PSNR():
    def __init__(self, max_val):
        self.max_val = max_val
    
    def run(self, y_true, y_pred):
        return tf.reduce_mean(tfi.psnr(y_true, y_pred, max_val=self.max_val))

class SSIM():
    def __init__(self, max_val):
        self.max_val = max_val
    
    def run(self, y_true, y_pred):
        return tf.reduce_mean(tfi.ssim(y_true, y_pred, max_val=self.max_val))

# Build model
model = SRCNN()

model.compile(loss = losses.MeanSquaredError(), 
                optimizers = optimizers.SGD(learning_rate=0.0001, momentum=0.99),
                metrics=[PSNR(1).run, SSIM(1).run])
# %%
