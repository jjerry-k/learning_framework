from tensorflow.keras import models, layers

def SRCNN(img_channel = 1, name="SRCNN"):
    
    Input = layers.Input(shape=(None, None, img_channel))

    x = layers.Conv2D(64, 9, activation='relu', name=name+"_Conv_1")(Input)
    x = layers.Conv2D(32, 1, activation='relu', name=name+"_Conv_2")(x)
    x = layers.Conv2D(img_channel, 5, name=name+"_Output")(x)

    return models.Model(Input, x, name=name)