import keras
import numpy as np


def load_and_preprocess_data():
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # 正規化

    np.savez('out/mnist_data.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

if __name__ == '__main__':
    load_and_preprocess_data()