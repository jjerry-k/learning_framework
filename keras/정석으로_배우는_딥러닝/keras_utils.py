import numpy as np
from keras import datasets  # mnist
from keras.utils import np_utils  # to_categorical
from matplotlib import pyplot as plt

def Data_func():
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

    Y_train = np_utils.to_categorical(y_train)
    Y_test = np_utils.to_categorical(y_test)

    L, W, H = X_train.shape
    X_train = X_train.reshape(-1, W * H)
    X_test = X_test.reshape(-1, W * H)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return (X_train, Y_train), (X_test, Y_test)

def plot_history(history):
    
    if not isinstance(history, dict):
        history = history.history
    
    plt.figure(figsize=(10, 4))

    plt.subplot(121)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title("Loss Graph")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(['Training', 'Validation'], loc=0)

    plt.subplot(122)
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title("Accuracy Graph")
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.legend(['Training', 'Validation'], loc=0)
    plt.show()


