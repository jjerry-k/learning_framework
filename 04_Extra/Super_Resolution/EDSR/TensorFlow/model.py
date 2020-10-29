import math
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.python.keras.layers.advanced_activations import PReLU

# To-do
# EDSR Model

def MeanShift(x, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1, name="MeanShift"):
    mean = sign * rgb_range * tf.reshape(rgb_mean, [1 ,1, -1]) / tf.reshape(rgb_std, [1, 1, -1])
    out = layers.Add(name=name)([x, mean])
    return out

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

def Upsampler(x, scale, filters, use_bias=True, use_bn=False, act="relu", name="Upsampler"):
    if (scale & (scale-1)) == 0:
        out = x
        for i in range(int(math.log(scale, 2))):
            out = layers.Conv2D(4*filters, 3, padding="same", use_bias=use_bias, name=name+f"_Conv{i+1}")(out)
            out = tf.nn.depth_to_space(out, scale, name=name+"_SubPixel")

            if use_bn:
                out = layers.BatchNormalization(name=name+"_BN")(out)
            if act == "relu":
                out = layers.Activation("relu", name=name+"_Act")(out)
            elif act == "prelu":
                out = layers.PReLU(name=name+"_Act")(out)

    elif scale == 3:
        out = layers.Conv2D(9*filters, 3, padding="same", use_bias=use_bias, name=name+f"_Conv{i+1}")(out)
        out = tf.nn.depth_to_space(out, scale, name=name+"_SubPixel")

        if use_bn:
                out = layers.BatchNormalization(name=name+"_BN")(out)
        if act == "relu":
            out = layers.Activation("relu", name=name+"_Act")(out)
        elif act == "prelu":
            out = layers.PReLU(name=name+"_Act")(out)
    else:
        raise NotImplementedError

    return out


def EDSR(img_channel=3, rgb_range=255, filters=64, n_resblocks=4, res_scale=1, act='relu', scale=4, name="EDSR"):
    input_layer = layers.Input(shape=(None, None, img_channel), name=name+"_Input")
    out = MeanShift(input_layer, rgb_range, name=name+"_MeanShift_Top")
    out = layers.Conv2D(filters, 3, padding="same", name=name+"_Conv1")(out)
    x = out
    for i in range(n_resblocks):
        out = ResBlock(out, filters, kernel_size, act=act, res_scale=res_scale, name=name+f"_ResBlock_{i+1}")
    out = layers.Conv2D(filters, 3, padding="same", name=name+"_Conv2")(out)
    out = layers.Add(name=name+"_Add")([out, x])
    out = Upsampler(out, scale, filters, act=False, name=name+"_Upsampler")
    out = layers.Conv2D(img_channel, 3, padding="same", name=name+"_Conv3")(out)
    out = MeanShift(out, rgb_range, sign=1, name=name+"_MeanShift_Bot")
    return models.Model(inputs=input_layer, outputs=out, name=name)
