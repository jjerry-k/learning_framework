# %%
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import io as tfi
from tensorflow import image as tfimg
from tensorflow.keras import models, layers, losses, metrics, optimizers, callbacks
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing import image_dataset_from_directory

from model import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# For Efficiency
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# %%
# Data Loader
ROOT = "../../datasets"
data = "BSR/BSDS500/data/images/"
root_dir = os.path.join(ROOT, "BSR/BSDS500/data/images")
train_path = os.path.join(ROOT, data, "train")
val_path = os.path.join(ROOT, data, "val")
input_size = 132
scale = 3
batch_size = 32

train_ds = image_dataset_from_directory(
    root_dir,
    batch_size=batch_size,
    image_size=(input_size*scale, input_size*scale),
    validation_split=0.2,
    subset="training",
    seed=42, 
    label_mode=None,
)

val_ds = image_dataset_from_directory(
    root_dir,
    batch_size=batch_size,
    image_size=(input_size*scale, input_size*scale),
    validation_split=0.2,
    subset="validation",
    seed=42,
    label_mode=None,
)

def process_input(image, input_size, scale):
    image = image / 255.0
    image = tf.image.rgb_to_yuv(image)
    last_dimension_axis = len(image.shape) - 1
    y, u, v = tf.split(image, 3, axis=last_dimension_axis)
    label = y
    image = tf.image.resize(y, [input_size//scale, input_size//scale], method="bicubic")
    image = tf.image.resize(image, [input_size, input_size], method="bicubic")

    return image, label[:,6:-6, 6:-6, :]

# train_ds = tf.data.Dataset.list_files(os.path.join(train_path, '*.jpg'))
# val_ds = tf.data.Dataset.list_files(os.path.join(val_path, '*.jpg'))

# def parse_image(filename, target_size, scale=3):
#     # Load image & Preprocessing
#     image = tfi.read_file(filename)
#     image = tfimg.decode_jpeg(image)
#     image = tfimg.convert_image_dtype(image, tf.float32)/255.
#     image = tfimg.rgb_to_yuv(image)[..., 0]
#     image = tf.expand_dims(tfimg.random_crop(image, [target_size, target_size]), axis=-1)

#     # Set label image
#     label = image[6:-6, 6:-6]

#     # Set 

#     image = tfimg.resize(image, [target_size//scale, target_size//scale], 'bicubic')
#     image = tfimg.resize(image, [target_size, target_size], 'bicubic')

#     return image, label

# train_ds = train_ds.map(lambda x: parse_image(x, input_size, scale)).batch(batch_size)
# train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

# val_ds = val_ds.map(lambda x: parse_image(x, input_size, scale)).batch(batch_size)
# val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)

# %%
# Defile psnr, ssim for metrics
class Metric():
    def __init__(self, max_val):
        self.max_val = max_val
    
    def psnr(self, y_true, y_pred):
        return tf.reduce_mean(tfimg.psnr(y_true, y_pred, max_val=self.max_val))

    def ssim(self, y_true, y_pred):
        return tf.reduce_mean(tfimg.ssim(y_true, y_pred, max_val=self.max_val))

# Define plot callback
class PlotCallback(callbacks.Callback):
    def __init__(self):
        super(PlotCallback, self).__init__()
        for i in val_ds.take(1):
            self.test_img, self.test_lab = i[0], i[1]

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % 10 == 0:
            pred = self.model(self.test_img)
            plt.subplot(131)
            plt.imshow(self.test_img[0, ..., 0], cmap='gray')
            plt.subplot(132)
            plt.imshow(pred[0, ..., 0], cmap='gray')
            plt.subplot(133)
            plt.imshow(self.test_lab[0, ..., 0], cmap='gray')
            plt.show()

# Build model
model = SRCNN()

model.compile(loss = losses.MeanSquaredError(), 
                optimizer = optimizers.Adam(learning_rate=0.0001),
                metrics=[Metric(1).psnr, Metric(1).ssim])
# %%
model.fit(train_ds, epochs=50, validation_data=val_ds, callbacks = [PlotCallback()])

# %%

# %%
