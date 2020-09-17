# %%
import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# %%
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

def generator_encoder_decoder(input_size = 256, A_channel = 3, B_channel = 3, name="Generator"):

    input_layer = layers.Input(shape=(input_size, input_size, A_channel), name=name+"_Input")

    en1 = encoding_block(input_layer, 64, use_act=False, use_bn=False, name=name+"_En1")
    en2 = encoding_block(en1, 128, name=name+"_En2")
    en3 = encoding_block(en2, 256, name=name+"_En3")
    en4 = encoding_block(en3, 512, name=name+"_En4")
    en5 = encoding_block(en4, 512, name=name+"_En5")
    en6 = encoding_block(en5, 512, name=name+"_En6")
    en7 = encoding_block(en6, 512, name=name+"_En7")
    en8 = encoding_block(en7, 512, use_bn=False, name=name+"_En8")

    de1 = decoding_block(en8, 512, name=name+"_De1")
    de1 = layers.Dropout(0.5, name=name+"_De1_Dropout")(de1)
    de2 = decoding_block(de1, 512, name=name+"_De2")
    de2 = layers.Dropout(0.5, name=name+"_De2_Dropout")(de2)
    de3 = decoding_block(de2, 512, name=name+"_De3")
    de3 = layers.Dropout(0.5, name=name+"_De3_Dropout")(de3)
    de4 = decoding_block(de3, 512, name=name+"_De4")
    de5 = decoding_block(de4, 256, name=name+"_De5")
    de6 = decoding_block(de5, 128, name=name+"_De6")
    de7 = decoding_block(de6, 64, name=name+"_De7")
    de8 = decoding_block(de7, B_channel, use_bn=False, name=name+"_De8")
    
    output = layers.Activation("tanh", name=name+name+"__Output")(de8)

    return models.Model(inputs=input_layer, outputs=output, name=name)

def generator_unet(input_size = 256, A_channel = 3, B_channel = 3, name="Generator"):
    
    input_layer = layers.Input(shape=(input_size, input_size, A_channel), name=name+"_Input")

    en1 = encoding_block(input_layer, 64, use_act=False, use_bn=False, name=name+"_En1")
    en2 = encoding_block(en1, 128, name=name+"_En2")
    en3 = encoding_block(en2, 256, name=name+"_En3")
    en4 = encoding_block(en3, 512, name=name+"_En4")
    en5 = encoding_block(en4, 512, name=name+"_En5")
    en6 = encoding_block(en5, 512, name=name+"_En6")
    en7 = encoding_block(en6, 512, name=name+"_En7")
    en8 = encoding_block(en7, 512, use_bn=False, name=name+"_En8")

    de1 = decoding_block(en8, 512, name=name+"_De1")
    de1 = layers.Dropout(0.5, name=name+"_De1_Dropout")(de1)
    de1 = layers.Concatenate(name=name+"_De1_Concat")([de1, en7])
    de2 = decoding_block(de1, 512, name=name+"_De2")
    de2 = layers.Dropout(0.5, name=name+"_De2_Dropout")(de2)
    de2 = layers.Concatenate(name=name+"_De2_Concat")([de2, en6])
    de3 = decoding_block(de2, 512, name=name+"_De3")
    de3 = layers.Dropout(0.5, name=name+"_De3_Dropout")(de3)
    de3 = layers.Concatenate(name=name+"_De3_Concat")([de3, en5])
    de4 = decoding_block(de3, 512, name=name+"_De4")
    de4 = layers.Concatenate(name=name+"_De4_Concat")([de4, en4])
    de5 = decoding_block(de4, 256, name=name+"_De5")
    de5 = layers.Concatenate(name=name+"_De5_Concat")([de5, en3])
    de6 = decoding_block(de5, 128, name=name+"_De6")
    de6 = layers.Concatenate(name=name+"_De6_Concat")([de6, en2])
    de7 = decoding_block(de6, 64, name=name+"_De7")
    de7 = layers.Concatenate(name=name+"_De7_Concat")([de7, en1])
    de8 = decoding_block(de7, B_channel, use_bn=False, name=name+"_De8")
    
    output = layers.Activation("tanh", name=name+name+"__Output")(de8)

    return models.Model(inputs=input_layer, outputs=output, name=name)

def discriminator(input_size = 256, A_channel = 3, B_channel = 3,  n_layers=0, name="Discriminator"):

    def encoding_block(x, filters=32, ksize=(4, 4), strides=(2, 2), padding="valid", use_act=True, use_bn=True, name="Encoding"):
        x = layers.Conv2D(filters, ksize, strides, padding, name=name+"_Conv")(x)
        if use_bn:
            x = layers.BatchNormalization(name=name+"_BN")(x)
        if use_act:
            x = layers.LeakyReLU(0.2, name=name+"_Act")(x)
        return x
        
    input_layer_A = layers.Input(shape=(input_size, input_size, A_channel), name=name+"_Input_A")
    input_layer_B = layers.Input(shape=(input_size, input_size, B_channel), name=name+"_Input_B")

    input_layer = layers.Concatenate(name=name+"_Input_Combin")([input_layer_A, input_layer_B])

    if n_layers==0:
        x = encoding_block(input_layer, 64, ksize=(1, 1), strides=(1, 1), use_bn=False, name=name+"_En1")
        x = encoding_block(x, 128, ksize=(1, 1), strides=(1, 1), name=name+"En2")
        x = encoding_block(x, 1, ksize=(1, 1), strides=(1, 1), use_act=False, use_bn=False, name=name+"En3")
        output = layers.Avtivation("sigmoid", name=name+"_Output")(x) 
        return models.Model(inputs = input_layer, outputs = output, name=name)
    else:
        
        x = encoding_block(input_layer, 64, padding="same", use_bn=False, name=name+"_En1")

        for i in range(1, n_layers-1):
            mul_fact = min(2**i, 8)
            x = encoding_block(x, mul_fact*64, padding="same", name=name+f"_En{i+1}")

        mul_fact = min(2**(n_layers-1), 8)
        x = encoding_block(x, mul_fact*64, ksize=(4, 4), strides=(1, 1), padding="same", name=name+f"_En{n_layers+1}")
        x = encoding_block(x, 1, ksize=(4, 4), strides=(1, 1), padding="same", use_act=False, use_bn=False, name=name+f"_En{n_layers+2}")
        x = layers.Activation("sigmoid", name=name+"_Output")(x)
        return models.Model(inputs = [input_layer_A, input_layer_B], outputs = x, name=name)
    
    # D1: C64-C128
    # D16: C64-C128
    # D70: C64-C128-C256-C512
    # D286: C64-C128-C256-C512-C512-C512
# %%
