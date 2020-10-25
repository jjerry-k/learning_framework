# %%
import math
import tensorflow as tf
from tensorflow.keras import layers, models

# To-do
# Implement MeanShift
# Implement Upsampler

def BasicBlock(x, filters, kernel_size, padding="same", use_bias=False, use_bn=True, act='relu', name="ConvBlock"):
    out = layers.Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, use_bias=use_bias, name=name+f"_Conv")(x)
    if use_bn:
        out = layers.BatchNormalization(name=name+f"_BN")(out)
    if act is not None:
        out = layers.Activation(act, name=name+f"_Act")(out)
    return out

def ResBlock(x, filters, kernel_size, padding="same", use_bias=True, use_bn=False, act = 'relu', res_scale=1, name="ResBlock"):
    out = x
    for i in range(2):
        out = layers.Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, use_bias=use_bias, name=name+f"_Conv{i+1}")(out)
        if use_bn:
            out = layers.BatchNormalization(name=name+f"_BN{i+1}")(out)
        if i==0:
            layers.Activation(act, name=name+f"_Act{i+1}")(out)
    
    out = layers.Add(name=name+"_Skip")([2*out, x])
    return out

# %%
x = layers.Input((128, 128, 16))

test = Residual_Block(x, 16, 3)

# %%
