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
    
    model = models.Sequential(name=name)
    model.add(layers.Dense(1200, input_shape=input_shape, name=name+"_Dense_1"))
    model.add(layers.BatchNormalization(name=name+"_BN_1"))
    model.add(layers.LeakyReLU(0.03, name=name+"_Act_1"))
    model.add(layers.Dense(1200, name=name+"_Dense_2"))
    model.add(layers.BatchNormalization(name=name+"_BN_2"))
    model.add(layers.LeakyReLU(0.03, name=name+"_Act_2"))
    model.add(layers.Dense(784, activation='sigmoid', name=name+"Dense"))
    model.add(layers.Reshape(output_size, name=name+"_Output"))
    return model

def Build_Discriminator(input_shape=(28,28), name="Discriminator"):
    
    model = models.Sequential(name=name)
    model.add(layers.Flatten(input_shape=input_shape, name=name+"_Flatten"))
    model.add(layers.Dense(240, name=name+"_Dense_1"))
    model.add(layers.LeakyReLU(0.03, name=name+"_Act_1"))
    model.add(layers.Dense(240, name=name+"_Dense_2"))
    model.add(layers.LeakyReLU(0.03, name=name+"_Act_2"))
    model.add(layers.Dense(1, activation='sigmoid', name=name+"_Output"))

    return model


n_latent = 100
input_shape = (n_latent, )
img_size = train_x.shape[1:]

D = Build_Discriminator(input_shape=img_size, name="Discriminator")
D.compile(optimizer=optimizers.Adam(), loss=losses.binary_crossentropy, metrics=['acc'])
D.trainable = False

G = Build_Generator(input_shape=input_shape, output_size=img_size, name="Generator")

A = models.Model(inputs=G.input, outputs=D(G.output), name="GAN")
A.compile(optimizer=optimizers.Adam(), loss=losses.binary_crossentropy)

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
    
    print(f"{epoch+1}/{epochs}, G loss : {G_loss_epoch/i}, D loss : {D_loss_epoch/i}, D acc : {D_acc_epoch/i}")

    latent = np.random.randn(32, n_latent)
    fake_x = G.predict(latent)

    plt.figure(figsize=(8, 4))
    for i in range(32):
        plt.subplot(4, 8, i+1)
        plt.imshow(fake_x[i], cmap='gray')
    plt.show()