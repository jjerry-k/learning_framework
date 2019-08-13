import tensorflow as tf
from tensorflow.keras import layers

def bam(_input, dilation, reduction):
    
    channel_attention = layers.GlobalAveragePooling2D()(_input)
    channel_attention = layers.Dense(tf.shape(_input)[-1]//reduction)(channel_attention)
    channel_attention = layers.BatchNormalization()(channel_attention)
    channel_attention = layers.ReLU()(channel_attention)
    channel_attention = layers.Dense(tf.shape(_input)[-1])(channel_attention)
    channel_attention = layers.BatchNormalization()(channel_attention)
    channel_attention = layers.ReLU()(channel_attention)

    spatial_attention = layers.Conv2D(tf.shape(_input)[-1]//reduction)
    spatial_attention = layers.BatchNormalization()(spatial_attention)
    spatial_attention = layers.ReLU()(spatial_attention)
    spatial_attention = layers.Conv2D(tf.shape(_input)[-1]//reduction, 3, dilation_rate=dilation, padding='same')(spatial_attention)
    spatial_attention = layers.BatchNormalization()(spatial_attention)
    spatial_attention = layers.ReLU()(spatial_attention)
    spatial_attention = layers.Conv2D(tf.shape(_input)[-1]//reduction, 3, dilation_rate=dilation, padding='same')(spatial_attention)
    spatial_attention = layers.BatchNormalization()(spatial_attention)
    spatial_attention = layers.ReLU()(spatial_attention)
    spatial_attention = layers.Conv2D(1, 1)(spatial_attention)

    attention = layers.Add()(channel_attention, spatial_attention)
    attention = layers.Activation('sigmoid')(attention)

    return _input + _input*attention


test = layers.Input(shape=[256, 256, 64])
test = bam(test, 4, 16)

print(test)