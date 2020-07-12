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

(train_x, _), (test_x, _) = datasets.mnist.load_data()
train_x, test_x = train_x/255., test_x/255.

print("Train Data's Shape : ", train_x.shape)
print("Test Data's Shape : ", test_x.shape)

# %%
# Build Network

def Build_Generator(input_shape=(100, ), output_size=(28, 28), name="Generator"):
    
    sub_size = (output_size[0]//4, output_size[1]//4)
    model = models.Sequential(name=name)
    model.add(layers.Dense(np.prod(sub_size)*512, input_shape=input_shape, name=name+"_Dense"))
    model.add(layers.BatchNormalization(name=name+"_BN_1"))
    model.add(layers.ReLU(name=name+"_Act_1"))
    model.add(layers.Reshape((sub_size[0], sub_size[1], 512), name=name+"_Reshape"))
    model.add(layers.Conv2DTranspose(256, 4, 2, padding='same', name=name+"_Upconv_1"))
    model.add(layers.BatchNormalization(name=name+"_BN_2"))
    model.add(layers.ReLU(name=name+"_Act_2"))
    model.add(layers.Conv2DTranspose(64, 4, 2, padding='same', name=name+"_Upconv_2"))
    model.add(layers.BatchNormalization(name=name+"_BN_3"))
    model.add(layers.ReLU(name=name+"_Act_3"))
    model.add(layers.Conv2D(1, 3, 1, padding='same', activation='tanh', name=name+"_Upconv_3"))
    model.add(layers.Reshape(output_size, name=name+"_Output"))
    return model

def Build_Discriminator(input_shape=(28,28), name="Discriminator"):
    
    model = models.Sequential(name=name)
    model.add(layers.Reshape((input_shape[0], input_shape[1], 1), input_shape=input_shape, name=name+"_Reshape"))
    model.add(layers.Conv2D(128, 4, 2, padding='same', name=name+"_Conv_1"))
    model.add(layers.BatchNormalization(name=name+"_BN_1"))
    model.add(layers.LeakyReLU(0.03, name=name+"_Act_1"))
    model.add(layers.Conv2D(64, 4, 2, padding='same', name=name+"_Conv_2"))
    model.add(layers.BatchNormalization(name=name+"_BN_2"))
    model.add(layers.LeakyReLU(0.03, name=name+"_Act_2"))
    model.add(layers.Flatten(name=name+"_Flatten"))
    model.add(layers.Dense(1, activation='sigmoid', name=name+"_Output"))

    return model


n_latent = 100
input_shape = (n_latent, )
img_size = train_x.shape[1:]

D = Build_Discriminator(input_shape=img_size, name="Discriminator")
D.compile(optimizer=optimizers.RMSprop(), loss=losses.binary_crossentropy, metrics=['acc'])
D.trainable = False

G = Build_Generator(input_shape=input_shape, output_size=img_size, name="Generator")

A = models.Model(inputs=G.input, outputs=D(G.output), name="GAN")
A.compile(optimizer=optimizers.RMSprop(), loss=losses.binary_crossentropy)

D.summary()

G.summary()

A.summary()

# %%
# Training Network
epochs=100
batch_size=100

fake_label = np.zeros((batch_size, 1))
real_label = np.ones((batch_size, 1))

for epoch in range(epochs):
    
    G_loss_epoch = 0
    
    D_loss_epoch = 0

    D_acc_epoch = 0

    shuffle_idx = np.random.choice(len(train_x), len(train_x), replace=False)

    for i, idx in enumerate(range(0, len(shuffle_idx), batch_size)):

        latent = np.random.randn(batch_size, n_latent)
        fake_x = G.predict(latent)

        real_x = train_x[idx:idx+batch_size]

        D_loss, D_acc = D.train_on_batch(np.concatenate((fake_x, real_x), axis=0), 
                                    np.concatenate((fake_label, real_label), axis=0))
        D_loss_epoch += D_loss
        D_acc_epoch += D_acc

        latent = np.random.randn(batch_size, n_latent)
        
        G_loss = A.train_on_batch(latent, real_label)

        G_loss_epoch += G_loss
    
    print(f"{epoch+1}/{epochs}, G loss : {G_loss_epoch/(i+1)}, D loss : {D_loss_epoch/(i+1)}, D acc : {D_acc_epoch/(i+1)}")

    latent = np.random.normal(-1, 1, (20, n_latent))
    fake_x = G.predict(latent)

    plt.figure(figsize=(10, 3))
    for i in range(10):
        plt.subplot(2, 10, i+1)
        plt.imshow(fake_x[2*i], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.subplot(2, 10, i+1+10)
        plt.imshow(fake_x[2*i+1], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
    plt.tight_layout()
    plt.show()