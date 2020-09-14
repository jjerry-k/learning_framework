# %%
import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers, models, optimizers, losses, utils
from tensorflow.python.keras.engine import input_layer

# %%
def generator(input_channel = 3, output_channel = 3, name="Generator"):
    def encoding_block(x, filters=32, ksize=(4, 4), strides=(2, 2), padding="same", use_act=True, use_bn=True, name="Encoding"):
        if use_act:
            x = layers.LeakyReLU(0.2, name=name+"_Act")(x)
        x = layers.Conv2D(filters, ksize, strides, padding, name=name+"_Conv")(x)
        if use_bn:
            x = layers.BatchNormalization(name=name+"_BN")(x)
        return x
    
    def decoding_block(x, filters=32, ksize=(4, 4), strides=(2, 2), padding="same", use_bn=True, name="Decoding"):
        x = layers.ReLU(name=name+"_Act")(x)
        x = layers.Conv2DTranspose(filters, ksize, strides, padding, name=name+"_ConvTranspose")(x)
        if use_bn:
            x = layers.BatchNormalization(name=name+"_BN")(x)
        return x

    input_layer = layers.Input(shape=(256, 256, input_channel), name=name+"_Input")

    en1 = encoding_block(input_layer, 64, use_act=False, use_bn=False, name='En1')
    en2 = encoding_block(en1, 128, name='En2')
    en3 = encoding_block(en2, 256, name='En3')
    en4 = encoding_block(en3, 512, name='En4')
    en5 = encoding_block(en4, 512, name='En5')
    en6 = encoding_block(en5, 512, name='En6')
    en7 = encoding_block(en6, 512, name='En7')
    en8 = encoding_block(en7, 512, use_bn=False, name='En8')

    de1 = decoding_block(en8, 512, name='De1')
    de1 = layers.Dropout(0.5, name="De1_Dropout")(de1)
    de2 = decoding_block(de1, 512, name='De2')
    de2 = layers.Dropout(0.5, name="De2_Dropout")(de2)
    de3 = decoding_block(de2, 512, name='De3')
    de3 = layers.Dropout(0.5, name="De3_Dropout")(de3)
    de4 = decoding_block(de3, 512, name='De4')
    de5 = decoding_block(de4, 256, name='De5')
    de6 = decoding_block(de5, 128, name='De6')
    de7 = decoding_block(de6, 64, name='De7')
    de8 = decoding_block(de7, output_channel, use_bn=False, name='De8')
    
    output = layers.Activation('tanh', name=name+"_Output")(de8)

    return models.Model(inputs=input_layer, outputs=output, name=name)
