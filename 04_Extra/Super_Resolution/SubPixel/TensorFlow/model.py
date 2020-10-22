import tensorflow as tf
from tensorflow.keras import layers, models

def SubPixel(img_channel=1, upscale_factor=4, name="SubPixel"):
    
    inputs = layers.Input(shape=(None, None, img_channel))
    x = layers.Conv2D(64, 5, padding='same', activation='relu', kernel_initializer="Orthogonal", name=name+"_Conv1")(inputs)
    x = layers.Conv2D(64, 3, padding='same', activation='relu', kernel_initializer="Orthogonal", name=name+"_Conv2")(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu', kernel_initializer="Orthogonal", name=name+"_Conv3")(x)
    x = layers.Conv2D(img_channel * (upscale_factor ** 2), 3, padding='same', activation='relu', kernel_initializer="Orthogonal", name=name+"_Conv4")(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor, name=name+"_Output")
    return models.Model(inputs=inputs, outputs=outputs, name=name)