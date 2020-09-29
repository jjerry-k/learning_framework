import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_addons as tfa

def norm_layer(mode="BN", name="Norm"):
    if mode == "BN":
        layer = layers.BatchNormalization(name=name+"_BN")
    elif mode == "IN":
        layer = tfa.layers.InstanceNormalization(name=name+"_IN")
    elif mode == "LN":
        layer = layers.LayerNormalization(name=name+"_LN")
    else :
        layer = lambda: lambda x: x
    return layer

def residual_block(x, filters=32, padding_type='reflect', norm_type="BN", use_dropout=True, use_bias=True, name="RB"):
    output = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode=padding_type, name=name+"_Pad_1")
    output = layers.Conv2D(filters, 3, padding='valid', use_bias=False, name=name+"_Conv_1")(output)
    output = norm_layer(mode = norm_type, name=name+"_Norm_1")(output)
    output = layers.ReLU(name=name+"_Act_1")(output)

    output = tf.pad(output, [[0, 0], [1, 1], [1, 1], [0, 0]], mode=padding_type, name=name+"_Pad_2")
    output = layers.Conv2D(filters, 3, padding='valid', use_bias=False, name=name+"_Conv_2")(output)
    output = norm_layer(mode = norm_type, name=name+"_Norm_2")(output)

    return layers.Add(name=name+"_Add")([x, output])

def ResnetGenerator(input_size=256, input_nc=3, output_nc=3, ngf=64, norm_type="BN", use_dropout=False, n_blocks=6, padding_type='reflect', name="Generator"):
    # To do 
    # [ ] make unetgenerator
    input_layer = layers.Input(shape=(input_size, input_size, input_nc), name=name+"_Input")

    out = tf.pad(input_layer, [[0, 0], [3, 3], [3, 3], [0, 0]], mode=padding_type, name=name+"_Pad_1")
    out = layers.Conv2D(ngf, 7, padding='valid', use_bias=False, name=name+"_Conv_1")(out)
    out = norm_layer(mode = norm_type, name=name+"_Norm_1")(out)
    out = layers.ReLU(name=name+"_Act_1")(out)

    n_downsampling = 2

    for i in range(n_downsampling):
        ngf *= 2
        out = layers.Conv2D(ngf, 3, strides=2, padding='same', use_bias=False, name=name+f"_Down_Conv_{i+2}")(out)
        out = norm_layer(mode = norm_type, name=name+f"_Down_Norm_{i+2}")(out)
        out = layers.ReLU(name=name+f"_Down_Act_{i+2}")(out)

    for i in range(n_blocks):
        out = residual_block(out, filters=ngf, name=name+f"_RB_{i+1}")

    for i in range(n_downsampling):
        ngf //= 2
        out = layers.Conv2DTranspose(ngf, 3, strides=2, padding='same', use_bias=False, name=name+f"_Up_Conv_{i+1}")(out)
        out = norm_layer(mode = norm_type, name=name+f"_Up_Norm_{i+1}")(out)
        out = layers.ReLU(name=name+f"_Up_Act_{i+1}")(out)

    out = tf.pad(out, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT', name=name+"_Out_Pad")
    out = layers.Conv2D(output_nc, 7, padding='valid', name=name+"_Out_Conv")(out)
    out = layers.Activation('tanh', name=name+"_Output")(out)

    return models.Model(inputs=input_layer, outputs=out, name=name)

def NLayerDiscriminator(input_size = 256, input_channel = 3, ndf=64, n_layers=3, norm_type="BN", name="Discriminator"):
    
    kw = 4
    padw = 1
    input_layer = layers.Input(shape=(input_size, input_size, input_channel), name=name+"_Input")
    
    out = layers.Conv2D(ndf, kernel_size=kw, strides=2, padding='same', name=name+"_Conv_1")(input_layer)
    out = layers.LeakyReLU(0.2, name=name+"_Act_1")(out)
    
    nf_mult = 1
    for i in range(1, n_layers):  # gradually increase the number of filters
        nf_mult = min(2 ** i, 8)
        out = layers.Conv2D(ndf * nf_mult, kernel_size=kw, strides=2, padding='same', name=name+f"_Conv_{i+1}")(out)
        out = norm_layer(mode=norm_type, name=name+f"_Norm_{i+1}")(out)
        out = layers.LeakyReLU(0.2, name=name+f"_Act_{i+1}")(out)
        
    nf_mult = min(2 ** n_layers, 8)
    out = layers.Conv2D(ndf * nf_mult, kernel_size=kw, strides=1, padding='same', name=name+f"_Conv_{n_layers+1}")(out)
    out = norm_layer(mode=norm_type, name=name+f"_Norm_{n_layers+1}")(out)
    out = layers.LeakyReLU(0.2, name=name+f"_Act_{n_layers+1}")(out)

    out = layers.Conv2D(1, kernel_size=kw, strides=1, padding='same', name=name+"_Output")(out)
    return models.Model(inputs=input_layer, outputs=out, name=name)

def PixelDiscriminator(input_size = 256, input_channel = 3, ndf=64, norm_type="BN", name="Discriminator"):
    
    input_layer = layers.Input(shape=(input_size, input_size, input_channel), name=name+"_Input")

    out = layers.Conv2D(ndf, kernel_size=1, strides=1, padding='same', name=name+"_Conv_1")(input_layer)
    out = layers.LeakyReLU(0.2, name=name+"_Act_1")(out)
    out = layers.Conv2D(ndf*2, kernel_size=1, strides=1, padding='same', name=name+"_Conv_2")(out)
    out = norm_layer(mode=norm_type, name=name+"_Norm_2")(out)
    out = layers.LeakyReLU(0.2, name=name+"_Act_2")(out)
    out = layers.Conv2D(1, kernel_size=1, strides=1, padding="same", name=name+"_Output")(out)
    
    return models.Model(inputs=input_layer, outputs=out)