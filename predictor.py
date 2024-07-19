from keras import models
import numpy as np
import base64
from PIL import Image
from io import BytesIO


def predict(model_type: str, image_b64: str) -> int:
    if model_type not in ['mnist_model', 'model_B']:
        raise ValueError('model_type should be either mnist_model or model_B')
    
    model_path = f'out/{model_type}.h5'

    model = load_model(model_path)
    image_np = decode_base64_to_np(image_b64)
    image_reshaped = image_np.reshape(1, 28, 28)
    # 1つの画像のみ推論するため先頭要素のみ取得
    prediction = model.predict(image_reshaped)[0]
    # predictionは0~9となる確率が順に入っているため、最も数値が大きい要素のindexを取得
    predict_num = np.argmax(prediction)
    return predict_num

def load_model(model_path: str):
    model = models.load_model(model_path)
    return model

def decode_base64_to_np(base64_image: str) -> np.ndarray:
    image_data  = base64.b64decode(base64_image)
    image = Image.open(BytesIO(image_data))
    return np.array(image)