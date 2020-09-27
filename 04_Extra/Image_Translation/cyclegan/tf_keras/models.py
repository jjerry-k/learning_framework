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

def residual_block(x, filters=32, padding_type="REFLECT", use_dropout=True, use_bias=True):
    output = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode=padding_type)
    output = layers.Conv2D(filters, 3, padding='valid', use_bias=False)(output)
    output = norm_layer()(output)
    output = tf.nn.relu(output)

    output = tf.pad(output, [[0, 0], [1, 1], [1, 1], [0, 0]], mode=padding_type)
    output = layers.Conv2D(filters, 3, padding='valid', use_bias=False)(output)
    output = norm_layer()(output)

    return layers.add([x, output])

def ResnetGenerator(input_nc, output_nc, ngf=64, norm_layer="BN", use_dropout=False, n_blocks=6, padding_type='reflect'):
    # To do 
    # [ ] make resnetgenerator, unetgenerator
    return 

# 128x128 --> 6 blocks
# 256x256 --> 9 blocks
# three convolutions, sev- eral residual blocks [18], 
# two fractionally-strided convo- lutions with stride 1/2 , 
# and one convolution that maps fea- tures to RGB.

def discriminator(input_size = 256, A_channel = 3, B_channel = 3,  n_layers=0, name="Discriminator"):
    
    # D1: C64-C128
    # D16: C64-C128
    # D70: C64-C128-C256-C512
    # D286: C64-C128-C256-C512-C512-C512

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
