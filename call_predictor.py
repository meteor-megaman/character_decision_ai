from predictor import predict
from PIL import Image
from io import BytesIO
import base64
import numpy as np


def main():
    model_type = 'mnist_model'
    image_path = 'out/x_data/image_0.png'

    image_base64 = transform_np_to_b64(image_path)
    

    result = predict(model_type, image_base64)
    print(result)

def transform_np_to_b64(image_path: str) -> str:
    with Image.open(image_path) as image:
        image_array = np.array(image)
    image = Image.fromarray(image_array)
    buffered = BytesIO()
    image.save(buffered, format='PNG')
    image_bytes = buffered.getvalue()
    encoded_data = base64.b64encode(image_bytes)
    return encoded_data

if __name__ == '__main__':
    main()