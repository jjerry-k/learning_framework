import os
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm

img_size = 128
batch_size = 32

data_dir = "../data/flower_photos"

print("Training Dataset")
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(data_dir, "train"),
    seed=123,
    image_size=(img_size, img_size),
    batch_size=batch_size)

print("Validation Dataset")
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(data_dir, "validation"),
    seed=123,
    image_size=(img_size, img_size),
    batch_size=batch_size)

with tqdm(total=len(train_ds)) as t:
    t.set_description(f'Train Loader')
    for i, (batch_img, batch_lab) in enumerate(train_ds):
        time.sleep(0.1)
        # Add feedforward & Optimization code
        t.set_postfix({"Train data shape": f"{batch_img.shape} {batch_lab.shape}"})
        t.update()

with tqdm(total=len(val_ds)) as t:
    t.set_description(f'Validation Loader')
    for i, (batch_img, batch_lab) in enumerate(val_ds):
        time.sleep(0.1)
        # Add evaluation code
        t.set_postfix({"Validation data shape": f"{batch_img.shape} {batch_lab.shape}"})
        t.update()