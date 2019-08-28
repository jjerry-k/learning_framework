from tensorflow.keras import layers, models, losses, optimizers

def build_unet(input_shape= (None, None, 1), output_channel = 1, last_activation='linear', name='unet'):
    
    input_layer = layers.Input(shape=input_shape, name=name+"_input")

    encoder1 = layers.Conv2D(64, 3, strides=1, padding='same', activation='relu', name=name+"_en1_conv1")(input_layer)
    encoder1 = layers.Conv2D(64, 3, strides=1, padding='same', activation='relu', name=name+"_en1_conv2")(encoder1)

    encoder2 = layers.MaxPooling2D(name=name+"_en2_pool")(encoder1)
    encoder2 = layers.Conv2D(128, 3, strides=1, padding='same', activation='relu', name=name+"_en2_conv1")(encoder2)
    encoder2 = layers.Conv2D(128, 3, strides=1, padding='same', activation='relu', name=name+"_en2_conv2")(encoder2)

    encoder3 = layers.MaxPooling2D(name=name+"_en3_pool")(encoder2)
    encoder3 = layers.Conv2D(256, 3, strides=1, padding='same', activation='relu', name=name+"_en3_conv1")(encoder3)
    encoder3 = layers.Conv2D(256, 3, strides=1, padding='same', activation='relu', name=name+"_en3_conv2")(encoder3)

    encoder4 = layers.MaxPooling2D(name=name+"_en4_pool")(encoder3)
    encoder4 = layers.Conv2D(512, 3, strides=1, padding='same', activation='relu', name=name+"_en4_conv1")(encoder4)
    encoder4 = layers.Conv2D(512, 3, strides=1, padding='same', activation='relu', name=name+"_en4_conv2")(encoder4)

    encoder5 = layers.MaxPooling2D(name=name+"_en5_pool")(encoder4)
    encoder5 = layers.Conv2D(1024, 3, strides=1, padding='same', activation='relu', name=name+"_en5_conv1")(encoder5)
    encoder5 = layers.Conv2D(1024, 3, strides=1, padding='same', activation='relu', name=name+"_en5_conv2")(encoder5)

    decoder4 = layers.Conv2DTranspose(512, 2, strides=2, padding='same', activation='relu', name=name+"_de4_upconv")(encoder5)
    decoder4 = layers.Concatenate(axis=-1, name=name+"_de4_concat")([encoder4, decoder4])
    decoder4 = layers.Conv2D(512, 3, strides=1, padding='same', activation='relu', name=name+"_de4_conv1")(decoder4)
    decoder4 = layers.Conv2D(512, 3, strides=1, padding='same', activation='relu', name=name+"_de4_conv2")(decoder4)

    decoder3 = layers.Conv2DTranspose(256, 2, strides=2, padding='same', activation='relu', name=name+"_de3_upconv")(decoder4)
    decoder3 = layers.Concatenate(axis=-1, name=name+"_de3_concat")([encoder3, decoder3])
    decoder3 = layers.Conv2D(256, 3, strides=1, padding='same', activation='relu', name=name+"_de3_conv1")(decoder3)
    decoder3 = layers.Conv2D(256, 3, strides=1, padding='same', activation='relu', name=name+"_de3_conv2")(decoder3)

    decoder2 = layers.Conv2DTranspose(128, 2, strides=2, padding='same', activation='relu', name=name+"_de2_upconv")(decoder3)
    decoder2 = layers.Concatenate(axis=-1, name=name+"_de2_concat")([encoder2, decoder2])
    decoder2 = layers.Conv2D(128, 3, strides=1, padding='same', activation='relu', name=name+"_de4_conv1")(decoder4)
    decoder2 = layers.Conv2D(128, 3, strides=1, padding='same', activation='relu', name=name+"_de4_conv2")(decoder4)

    decoder1 = layers.Conv2DTranspose(64, 2, strides=2, padding='same', activation='relu', name=name+"_de1_upconv")(decoder2)
    decoder1 = layers.Concatenate(axis=-1, name=name+"_de1_concat")([encoder1, decoder1])
    decoder1 = layers.Conv2D(64, 3, strides=1, padding='same', activation='relu', name=name+"_de1_conv1")(decoder4)
    decoder1 = layers.Conv2D(64, 3, strides=1, padding='same', activation='relu', name=name+"_de1_conv2")(decoder4)

    output = layers.Conv2D(output_channel, 1, strides=1, padding='same', activation=last_activation, name=name+"_prediction")(decoder1)

    return models.Model(inputs=input_layer, outputs=output, name=name)

unet = build_unet()
unet.summary()