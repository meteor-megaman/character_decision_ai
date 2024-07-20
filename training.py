import base64
from io import BytesIO
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
from keras import layers, models
import os

def train(model_type: str) -> str:
    """
    渡されたモデルタイプに従い、モデル学習を行い、学習結果のグラフを返却する
    """
    # モデル学習
    accuracy, loss = create_model(model_type)
    
    # 学習結果のグラフ作成(バイナリデータ)
    img_str = create_train_result_img(accuracy, loss)
    return img_str

def create_model(model_type: str) -> tuple:
    x_train, y_train = load_data()
    model = define_model(model_type)
    compile_model(model)
    accuracy, loss = train_model(model, x_train, y_train)
    
    save_model(model)

    return accuracy, loss

def load_data():    
    data = np.load(os.path.join('out', 'mnist_data.npz'))
    return data['x_train'], data['y_train']


def define_model(model_type: str):
    if model_type == 'simple_perceptron':
        model = models.Sequential([
            layers.Flatten(input_shape=(28, 28)),
            layers.Dense(10, activation='softmax')
        ])
    elif model_type == 'multi_layer_perceptron':
        model = models.Sequential([
            layers.Flatten(input_shape=(28, 28)),
            layers.Dense(512, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
    else:
        raise ValueError('Invalid model type')

    return 


def compile_model(model):
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

def train_model(model, x_train, y_train):
    history = model.fit(x_train, y_train, epochs=5)
    return history.history['accuracy'], history.history['loss']

def save_model(model):
    model.save(os.path.join('out', 'mnist_model.h5'))

def create_train_result_img(accuracy: list, loss: list):
    epochs = range(1, len(accuracy) + 1)
    
    fig = plt.figure(figsize=(12, 6))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy, 'bo-', label='Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'ro-', label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    plt.tight_layout()

    # 返却用データに変換
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    np_img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

    img = Image.fromarray(np_img)
    buffered = BytesIO()
    img.save(buffered, format='PNG')
    image_bytes = buffered.getvalue()
    encoded_img = base64.b64encode(image_bytes)

    return encoded_img