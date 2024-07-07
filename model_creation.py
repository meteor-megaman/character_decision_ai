import numpy as np
from keras import layers, models
import os

def create_model():
    print(os.getcwd())
    x_train, y_train, x_test, y_test = load_data()
    model = define_model()
    compile_model(model)
    train_model(model, x_train, y_train)
    evaluate_model(model, x_test, y_test)
    save_model(model)

def load_data():    
    data = np.load(os.path.join('out', 'mnist_data.npz'))
    return data['x_train'], data['y_train'], data['x_test'], data['y_test']


def define_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])

    return model



def compile_model(model):
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

def train_model(model, x_train, y_train):
    model.fit(x_train, y_train, epochs=5)

def evaluate_model(model, x_test, y_test):
    model.evaluate(x_test, y_test)

def save_model(model):
    model.save(os.path.join('out', 'mnist_model.h5'))

if __name__ == '__main__':
    create_model()