import os
import argparse

import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, losses, optimizers
from tensorflow.keras.utils import Progbar

from models import *

tf.random.set_seed(42)

# For Efficiency
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

def load_img(path, size):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (size, size))
    return img

# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():
def main(args):
    
    print(f"Load {args.DATASET} dataset.....")

    datasets_root = "../../datasets" # Please edit your root path of datasets
    
    train_domain_A_path = os.path.join(datasets_root, args.DATASET, "train", "domain_A")
    train_domain_B_path = os.path.join(datasets_root, args.DATASET, "train", "domain_B")

    val_domain_A_path = os.path.join(datasets_root, args.DATASET, "val", "domain_A")
    val_domain_B_path = os.path.join(datasets_root, args.DATASET, "val", "domain_B")

    train_A = np.array([load_img(os.path.join(train_domain_A_path, img), args.IMG_SIZE) for img in sorted(os.listdir(train_domain_A_path))])/127.5 -1
    train_B = np.array([load_img(os.path.join(train_domain_B_path, img), args.IMG_SIZE) for img in sorted(os.listdir(train_domain_B_path))])/127.5 -1
    
    val_A = np.array([load_img(os.path.join(val_domain_A_path, img), args.IMG_SIZE) for img in sorted(os.listdir(val_domain_A_path))])/127.5 -1
    val_B = np.array([load_img(os.path.join(val_domain_B_path, img), args.IMG_SIZE) for img in sorted(os.listdir(val_domain_B_path))])/127.5 -1

    print("\nTraining data shape")
    print(f"Domain A: {train_A.shape}")
    print(f"Domain B: {train_B.shape}")

    print("\nValidation data shape")
    print(f"Domain A: {val_A.shape}")
    print(f"Domain B: {val_B.shape}")


    print("================ Building Network ================")
    A_channel = train_A.shape[-1]
    B_channel = train_B.shape[-1]
    n_layers = 3

    G = generator_unet(input_size=args.IMG_SIZE, A_channel=A_channel, B_channel=B_channel, name="Generator")
    G.summary()

    D = discriminator(input_size=args.IMG_SIZE, A_channel=A_channel, B_channel=B_channel, n_layers=n_layers, name="Discriminator")
    D.summary()
    
    D.compile(optimizer=optimizers.Adam(lr=0.0001, epsilon=1e-8), loss=losses.BinaryCrossentropy())

    D.trainable=False

    A_img = layers.Input(shape=(args.IMG_SIZE, args.IMG_SIZE, A_channel), name="GAN_Input_A")
    B_img = layers.Input(shape=(args.IMG_SIZE, args.IMG_SIZE, B_channel), name="GAN_Input_B")
    
    fake_B = G(A_img)

    D_output = D([A_img, fake_B])

    A = models.Model(inputs=[A_img, B_img], outputs = [D_output, fake_B], name='GAN')
    A.summary()

    A.compile(optimizer=optimizers.Adam(lr=0.0001, epsilon=1e-8), loss=[losses.BinaryCrossentropy(), losses.MeanAbsoluteError()], 
            loss_weights=[1, 100])
    
    print("==================================================\n")


    print("================ Training Network ================")

    d_output_size = args.IMG_SIZE // (2**(n_layers-1))
    epochs = args.EPOCHS
    batch_size = args.BATCH_SIZE
    train_length = len(train_A)
    val_length = len(val_A)
    num_iter = int(np.ceil(train_length/batch_size))
    num_val_iter = int(np.ceil(val_length/batch_size))

    CKPT_PATH = './ckpt'
    os.makedirs(CKPT_PATH, exist_ok=True)
    model_json = A.to_json()
    with open(os.path.join(CKPT_PATH, "GAN.json"), "w") as json_file:
        json_file.write(model_json)
    print("\nModel Saved!\n")

    SAMPLE_PATH = './result'
    os.makedirs(SAMPLE_PATH, exist_ok=True) 

    for epoch in range(epochs):
        
        g_total = 0
        g_ad = 0
        g_mae = 0
        d_ad = 0

        shuffle_idx = np.random.choice(train_length, train_length, replace=False)
        
        epoch_progbar = Progbar(num_iter, width=15)

        for i, step in enumerate(range(0, train_length, batch_size)):

            step_idx = shuffle_idx[step:step+batch_size]
            real_label = np.ones((len(step_idx), d_output_size, d_output_size, 1))
            fake_label = np.zeros((len(step_idx), d_output_size, d_output_size, 1))

            # Generate fake images
            fake_imgs = G.predict(train_A[step_idx])

            # Train Discriminator
            dis_label = np.concatenate([fake_label, real_label])
            Set_A = np.concatenate([fake_imgs, train_A[step_idx]], axis=0)
            Set_B = np.concatenate([train_B[step_idx], train_B[step_idx]], axis=0)
            # [Ad]
            D_Loss = D.train_on_batch([Set_A, Set_B], dis_label)
            
            # Train Generator
            # [Ad + 100*mae, Ad, mae]
            G_Loss = A.train_on_batch([train_A[step_idx], train_B[step_idx]], 
                                        [real_label, train_B[step_idx]])

            g_total += G_Loss[0]
            g_ad += G_Loss[1]
            g_mae += G_Loss[2]
            d_ad += D_Loss
            if i < num_iter:
                epoch_progbar.update(i+1, [("G_Total", G_Loss[0])
                                            ("G_Ad", G_Loss[1]), 
                                            ("G_MAE", G_Loss[2]), 
                                            ("D_Ad", D_Loss)
                                            ])

        val_g_total = 0
        val_g_ad = 0
        val_g_mae = 0

        for j, val_idx in enumerate(range(0, val_length, batch_size)):
            val_label = np.ones([len(val_A[val_idx:val_idx+batch_size]), d_output_size, d_output_size, 1])
            V_loss = A.test_on_batch(val_A[val_idx:val_idx+batch_size], 
                                            [val_B[val_idx:val_idx+batch_size], val_label])
            
            val_g_total += V_loss[0]
            val_g_ad += V_loss[1]
            val_g_mae += V_loss[2]

        epoch_progbar.update(i+1, [("Val_G_Total", val_g_total/num_val_iter), ("Val_G_Ad", val_g_ad/num_val_iter), ("Val_G_MAE", val_g_mae/num_val_iter)])

        A.save_weights(os.path.join(CKPT_PATH, f"{epoch:04d}_params.h5"))

        train_float2int = np.concatenate((train_B[step_idx][0], fake_imgs[0]), axis=1)
        train_float2int = (train_float2int + 1) * 127.5
        train_float2int = cv.cvtColor(train_float2int.astype(np.uint8), cv.COLOR_RGB2BGR)
        Train_Result_PATH = os.path.join(SAMPLE_PATH, f"{epoch+1:04d}_train_result.jpg")
        cv.imwrite(Train_Result_PATH, train_float2int)

        val_result = G.predict(val_A[:1])
        val_float2int = np.concatenate((val_B[0], val_result[0]), axis=1)
        val_float2int = (val_float2int + 1) * 127.5
        val_float2int = cv.cvtColor(val_float2int.astype(np.uint8), cv.COLOR_RGB2BGR)
        Val_Result_PATH = os.path.join(SAMPLE_PATH, f"{epoch+1:04d}_val_result.jpg")
        cv.imwrite(Val_Result_PATH, val_float2int)

    print("Training Done ! ")

if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATASET", default="facades", type=str, help="Dataset")
    parser.add_argument("--IMG_SIZE", default=256, type=int, help="Imgae size")
    parser.add_argument("--EPOCHS", default=100, type=int, help="Number of Epoch")
    parser.add_argument("--BATCH_SIZE", default=32, type=int, help="Number of Batch")

    

    args = parser.parse_args()

    dict_args = vars(args)

    for i in dict_args.keys():
        assert dict_args[i]!=None, '"%s" key is None Value!'%i
    print("\n================ Training Options ================")
    print(f"Dataset : {args.DATASET}")
    print(f"Imgae size : {args.IMG_SIZE}")
    print(f"Epochs : {args.EPOCHS}")
    print(f"Batch size : {args.BATCH_SIZE}")
    print("====================================================\n")


    main(args)