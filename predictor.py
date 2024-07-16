from keras import models
import matplotlib.pyplot as plt
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import os

def main():
    image_array = import_image('out/x_data/image_0.png')
    print(image_array.shape)
    encoded_image = encode_image_to_base64(image_array)
    predict(encoded_image)

def predict(encoded_data):
    model = load_model()
    make_predictions(model, encoded_data)

def import_image(file_path):
    with Image.open(file_path) as image:
        image_array = np.array(image)
    return image_array

def encode_image_to_base64(image_array):
    image = Image.fromarray(image_array)
    buffered = BytesIO()
    image.save(buffered, format='PNG')
    image_bytes = buffered.getvalue()
    encoded_data = base64.b64encode(image_bytes)
    return encoded_data
    
def load_test_data():
    data = np.load('out/mnist_data.npz')
    return {
        'x_test': data['x_test'],
        'y_test': data['y_test']
    }

def make_predictions(model, base64_image):
    def _decode_base64_to_string(base64_image):
        image_data  = base64.b64decode(base64_image)
        image = Image.open(BytesIO(image_data))
        return np.array(image)
    
    x_data = _decode_base64_to_string(base64_image)
    x_data_reshaped = x_data.reshape(1, 28, 28)
    prediction = model.predict(x_data_reshaped)
    predict_num = np.argmax(prediction[0])
    print('==============================')
    print(f'predict_num: {predict_num}')
    print('==============================')


def load_model():
    model = models.load_model('out/mnist_model.h5')
    return model

def save_images():
    test_data = load_test_data()
    x_data = test_data['x_test']
    for i in range(10):
        image = Image.fromarray((x_data[i] * 255).astype('uint8'), 'L')
        image.save(os.path.join('out/x_data', f'image_{i}.png'))

if __name__ == '__main__':
    main()