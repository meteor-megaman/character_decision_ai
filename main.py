from keras import models
import matplotlib.pyplot as plt
import numpy as np


def main():
    model = load_model()
    test_data = load_test_data()
    make_predictions(model, test_data)

    
def load_test_data():
    data = np.load('out/mnist_data.npz')
    return {
        'x_test': data['x_test'],
        'y_test': data['y_test']
    }

def make_predictions(model, test_data):
    correct_num = 0
    invalid_num = 0
    all_num = len(test_data['x_test'])
    for x_test, y_test in zip(test_data['x_test'], test_data['y_test']):
        x_test_reshaped = x_test.reshape(1, 28, 28)
        predictions = model.predict(x_test_reshaped)
        # predictionの中で最も確率の高いもののindexを取得
        predict_num = np.argmax(predictions[0])
        result = predict_num == y_test
        if result:
            correct_num += 1
        else:
            invalid_num += 1
            plt.imshow(x_test, cmap='gray')
            plt.savefig(f'out/invalid/invalid_{y_test}.png')
        print('-----------------')
    print(f'correct: {correct_num}, invalid: {invalid_num}, all: {all_num}')


def load_model():
    model = models.load_model('out/mnist_model.h5')
    return model

if __name__ == '__main__':
    main()