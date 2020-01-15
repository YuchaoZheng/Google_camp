import requests
import argparse
from PIL import Image
import io

# Initialize the PyTorch REST API endpoint URL.
PyTorch_REST_API_URL = 'http://35.221.192.94:443/predict'


def predict_result(image_path):
    # Initialize image path
    image = open(image_path, 'rb').read()
    print(len(image))
    payload = {'image': image}

    print("PPPPPP")

    # Submit the request.
    picture_bytes = requests.post(PyTorch_REST_API_URL, files=payload)

    # Ensure the request was successful.
    if picture_bytes != None:
        print("success")
        print(picture_bytes)

        image = Image.open(io.BytesIO(picture_bytes.content))
        print(image.size)
        image.save("result.png")

    # Otherwise, the request failed.
    else:
        print('Request failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification demo')
    parser.add_argument('--file', type=str, help='test image file')

    args = parser.parse_args()
    predict_result(args.file)

