# %%
# Import Package
import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers, models, losses, optimizers, datasets, utils

# %%
# Data Prepare

n_c = 10

(train_x, train_y), (_, _) = datasets.mnist.load_data()
train_x, train_y = train_x/255., utils.to_categorical(train_y, n_c)

print("Train Data's Shape : ", train_x.shape, train_y.shape)

# %%
# Build Network

def Build_Generator(input_shape=(100, ), input_condition=(3,), output_size=(28, 28), name="Generator"):
    
    latent = layers.Input(shape=input_shape, name=name+"_Latent")
    condition = layers.Input(shape=input_condition, name=name+"_Condition")

    x1 = layers.Dense(100, name=name+"_Latent_embedding")(latent)
    x1 = layers.BatchNormalization(name=name+"_Latent_BN")(x1)
    x1 = layers.LeakyReLU(0.03, name=name+"_Latent_Act")(x1)

    x2 = layers.Dense(10, name=name+"_Condition_embedding")(condition)
    x2 = layers.BatchNormalization(name=name+"_Condition_BN")(x2)
    x2 = layers.LeakyReLU(0.03, name=name+"_Condition_Act")(x2)
    
    x = layers.Concatenate(name=name+"_Concat")([x1, x2])

    x = layers.Dense(1200, name=name+"_Dense_1")(x)
    x = layers.BatchNormalization(name=name+"_BN_1")(x)
    x = layers.LeakyReLU(0.03, name=name+"_Act_1")(x)

    x = layers.Dense(1200, name=name+"_Dense_2")(x)
    x = layers.BatchNormalization(name=name+"_BN_2")(x)
    x = layers.LeakyReLU(0.03, name=name+"_Act_2")(x)

    x = layers.Dense(np.prod(output_size), activation='sigmoid', name=name+"Dense")(x)
    x = layers.Reshape(output_size, name=name+"_Output")(x)
    
    return models.Model(inputs=[latent, condition], outputs=x, name=name)

def Build_Discriminator(input_shape=(28, 28), input_condition=(3,), name="Discriminator"):
    
    img = layers.Input(shape=input_shape, name=name+"_Image")
    condition = layers.Input(shape=input_condition, name=name+"_Condition")

    x1 = layers.Flatten(name=name+"_Image_Flatten")(img)
    x1 = layers.Dense(256, name=name+"_Image_embedding")(x1)
    x1 = layers.LeakyReLU(0.03, name=name+"_Image_Act")(x1)

    x2 = layers.Dense(10, name=name+"_Condition_embedding")(condition)
    x2 = layers.LeakyReLU(0.03, name=name+"_Condition_Act")(x2)

    x = layers.Concatenate(name=name+"_Concat")([x1, x2])
    
    x = layers.Dense(240, name=name+"_Dense_1")(x)
    x = layers.LeakyReLU(0.03, name=name+"_Act_1")(x)

    x = layers.Dense(240, name=name+"_Dense_2")(x)
    x = layers.LeakyReLU(0.03, name=name+"_Act_2")(x)

    x = layers.Dense(1, activation='sigmoid', name=name+"_Output")(x)

    return models.Model(inputs=[img, condition], outputs=x, name=name)


n_latent = 100

input_shape = (n_latent, )
input_condition = (n_c, )
img_size = train_x.shape[1:]

D = Build_Discriminator(input_shape=img_size, input_condition=input_condition, name="Discriminator")
D.compile(optimizer=optimizers.RMSprop(), loss=losses.binary_crossentropy, metrics=['acc'])
D.trainable = False

G = Build_Generator(input_shape=input_shape, input_condition=input_condition, output_size=img_size, name="Generator")

A_latent = layers.Input(shape=input_shape, name='GAN_Latent')
A_condition = layers.Input(shape=input_condition, name='GAN_Condition')
A_img = G([A_latent, A_condition])
A_output = D([A_img, A_condition])
A = models.Model(inputs=[A_latent, A_condition], outputs=A_output, name="GAN")
A.compile(optimizer=optimizers.RMSprop(), loss=losses.binary_crossentropy)

D.summary()

G.summary()

A.summary()

# %%
# Training Network
epochs=100
batch_size=128

fake_label = np.zeros((batch_size, 1))
real_label = np.ones((batch_size, 1))

for epoch in range(epochs):
    
    G_loss_epoch = 0
    
    D_loss_epoch = 0

    D_acc_epoch = 0

    for i, idx in enumerate(range(0, len(train_x), batch_size)):
        
        shuffle_idx = np.random.choice(len(train_x), batch_size, replace=False)
        
        condition = train_y[shuffle_idx]
        latent = np.random.normal(-1, 1, (batch_size, n_latent))
        
        fake_x = G.predict([latent, condition])

        real_x = train_x[shuffle_idx]
        
        

        D_loss, D_acc = D.train_on_batch([np.concatenate((fake_x, real_x), axis=0), 
                                            np.concatenate((condition, condition), axis=0)],
                                    np.concatenate((fake_label, real_label), axis=0))
        D_loss_epoch += D_loss
        D_acc_epoch += D_acc

        latent = np.random.normal(-1, 1, (batch_size, n_latent))
        
        G_loss = A.train_on_batch([latent, condition], real_label)

        G_loss_epoch += G_loss
    
    print(f"{epoch+1}/{epochs}, G loss : {G_loss_epoch/(i+1)}, D loss : {D_loss_epoch/(i+1)}, D acc : {D_acc_epoch/(i+1)}")

    latent = np.random.normal(-1, 1, (20, n_latent))
    condition = utils.to_categorical(np.arange(10), n_c)
    condition = np.repeat(condition, 2, axis=0)
    fake_x = G.predict([latent, condition])

    plt.figure(figsize=(10, 3))
    for i in range(10):
        plt.subplot(2, 10, i+1)
        plt.imshow(fake_x[2*i], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title(f"{np.argmax(condition[2*i])}")
        plt.subplot(2, 10, i+1+10)
        plt.imshow(fake_x[2*i+1], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title(f"{np.argmax(condition[2*i+1])}")
    plt.tight_layout()
    plt.show()