import os
import time
import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm

img_size = 128
batch_size = 32

data_dir = "../data/flower_photos"
IMG_FORMAT = ["jpg", "jpeg", "tif", "tiff", "bmp", "png"]

class DataGenerator():
    def __init__(self, data_dir, img_size):
        self.filelist = []
        self.classes = sorted(os.listdir(data_dir))
        for root, sub_dir, files in os.walk(data_dir):
            if not len(files): continue
            files = [os.path.join(root, file) for file in files if file.split(".")[-1].lower() in IMG_FORMAT]
            self.filelist += files
        self.filelist.sort()    
    
    def __len__(self):
        return len(self.filelist)

    def __call__(self):
        for file in self.filelist:
            image = Image.open(file)
            image = image.resize((img_size, img_size))
            image = np.array(image)
            label = file.split('/')[-2]
            label = self.classes.index(label)
            yield image, label

train_dataset = DataGenerator(os.path.join(data_dir, "train"), img_size)
val_dataset = DataGenerator(os.path.join(data_dir, "validation"), img_size)

train_ds = tf.data.Dataset.from_generator(
    train_dataset, (tf.float32, tf.int16))
train_ds = train_ds.batch(batch_size).prefetch(2)
train_ds.__len__ = int(np.ceil(len(train_dataset)/batch_size))

val_ds = tf.data.Dataset.from_generator(
    val_dataset, (tf.float32, tf.int16))
val_ds = val_ds.batch(batch_size).prefetch(2)
val_ds.__len__ = int(np.ceil(len(val_dataset)/batch_size))

with tqdm(total=train_ds.__len__) as t:
    t.set_description(f'Train Loader')
    for i, (batch_img, batch_lab) in enumerate(train_ds):
        time.sleep(0.1)
        # Add feedforward & Optimization code
        t.set_postfix({"Train data shape": f"{batch_img.shape} {batch_lab.shape}"})
        t.update()

with tqdm(total=val_ds.__len__) as t:
    t.set_description(f'Validation Loader')
    for i, (batch_img, batch_lab) in enumerate(val_ds):
        time.sleep(0.1)
        # Add evaluation code
        t.set_postfix({"Validation data shape": f"{batch_img.shape} {batch_lab.shape}"})
        t.update()