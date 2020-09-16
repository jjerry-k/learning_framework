import os
import csv
import tqdm
import datetime
import random
import argparse

import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models, layers, losses, optimizers
from tensorflow.keras.utils import Progbar

from models import *

tf.random.set_seed(42)

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

def load_img(path, size):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (size, size))
    return img

# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():
def main(args):
    
    print(f"Load {args.DATASET} dataset.....")

    datasets_root = "../../datasets" # Please edit your root path of datasets
    
    train_domain_A_path = os.path.join(datasets_root, args.DATASET, "train", "domain_A")
    train_domain_B_path = os.path.join(datasets_root, args.DATASET, "train", "domain_B")

    val_domain_A_path = os.path.join(datasets_root, args.DATASET, "val", "domain_A")
    val_domain_B_path = os.path.join(datasets_root, args.DATASET, "val", "domain_B")

    train_A = np.array([load_img(os.path.join(train_domain_A_path, img), args.IMG_SIZE) for img in sorted(os.listdir(train_domain_A_path))])/127.5 -1
    train_B = np.array([load_img(os.path.join(train_domain_B_path, img), args.IMG_SIZE) for img in sorted(os.listdir(train_domain_B_path))])/127.5 -1
    
    val_A = np.array([load_img(os.path.join(val_domain_A_path, img), args.IMG_SIZE) for img in sorted(os.listdir(val_domain_A_path))])/127.5 -1
    val_B = np.array([load_img(os.path.join(val_domain_B_path, img), args.IMG_SIZE) for img in sorted(os.listdir(val_domain_B_path))])/127.5 -1

    print("\nTraining data shape")
    print(f"Domain A: {train_A.shape}")
    print(f"Domain B: {train_B.shape}")

    print("\nValidation data shape")
    print(f"Domain A: {val_A.shape}")
    print(f"Domain B: {val_B.shape}")


    # print("================ Building Network ================")
    # residual_channels = args.num_channel
    # G = generator_unet(input_size=args.IMG_SIZE, residual_channel=residual_channels, 
    #             layer_activation='leaky_relu', 
    #             name='3D_SR_Gen')
    # D = discriminator(input_size=args.IMG_SIZE, input_channel=)

    # D.compile(optimizer=optimizers.Adam(lr=0.0001, epsilon=1e-8), loss=losses.binary_crossentropy)

    # D.trainable=False

    # A = models.Model(inputs=G.input, outputs = [G.output, D(G.input)], name='GAN')
    # A.compile(optimizer=optimizers.Adam(lr=0.0001, epsilon=1e-8), loss=[loss_dict[args.loss.upper()], losses.binary_crossentropy], 
    #         loss_weights=[10, 1], metrics={'3D_SR_Gen_output_act':[gradient_3d_loss, psnr]})
    # print("==================================================\n")


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATASET", default="cityscapes", type=str, help="Dataset")
    parser.add_argument("--IMG_SIZE", default=64, type=int, help="Imgae size")
    

    args = parser.parse_args()

    dict_args = vars(args)

    for i in dict_args.keys():
        assert dict_args[i]!=None, '"%s" key is None Value!'%i
    print("\n================ Training Options ================")
    print(f"Dataset : {args.DATASET}")
    print(f"Imgae size : {args.IMG_SIZE}")
    print("====================================================\n")


    main(args)