from tensorflow.keras import layers, models

def VDSR(img_channel=1, name="VDSR"):
    
    Input_layer = layers.Input(shape=(None, None, img_channel), name=name+"_Input")

    out = layers.Conv2D(64, 3, padding='same', activation='relu', name=name+"_Conv_1")(Input_layer)

    for i in range(1, 19):
        out = layers.Conv2D(64, 3, padding='same', activation='relu', name=name+f"_Conv_{i+1}")(out)
    
    out = layers.Conv2D(img_channel, 3, padding='same', activation='relu', name=name+"_Conv_20")(out)

    output = layers.Add(name=name+"_Output")([Input_layer, out])

    return models.Model(inputs=Input_layer, outputs=output, name=name)

