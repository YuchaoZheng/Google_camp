import io
import json
from src import *

import flask
import torch
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms as T
from torchvision.models import resnet50

# Initialize our Flask application and the PyTorch model.
app = flask.Flask(__name__)
model = None
use_gpu = True

model_file = "/home/yuchaozheng_zz/Google_camp/segmentation/best.pth"
model = UNet()
model.load_state_dict(torch.load(model_file))
model.eval()
if use_gpu:
    model.cuda()


def prepare_image(img_path, target_size):
    img = cv2.imread(img_path)
    img = torch.from_numpy(img)
    img = img.permute(2, 0, 1).float()
    img /= 255.
    img = torch.unsqueeze(img, 0)

    img = img.cuda()
    return img


@app.route("/predict", methods=["POST"])
def predict():
    # Initialize the data dictionary that will be returned from the view.

    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == 'POST':
        if flask.request.files.get("image"):
            # Read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            img_path = "./receive.jpg"
            image.save(img_path)
            # Preprocess the image and prepare it for classification.
            img = prepare_image(img_path, target_size=(800, 600))

            with torch.no_grad():
                pred = model(img)
                pred = torch.argmax(pred.cpu(), dim=1)

                img = cv2.imread(img_path)
               	pred = torch.unsqueeze(pred, 1)
                pred = torch.squeeze(pred, 0)

                # c, h, w -> h, w, c
                pred = pred.permute(1, 2, 0).numpy()
                alpha_preds = pred * 255
                predicted_masks = np.concatenate((img, alpha_preds), axis=-1)
                cv2.imwrite('./result.png', predicted_masks)  
	        	
                result = open('./result.png', 'rb').read()

                return result
    # Return the data dictionary as a JSON response.
    return None 


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=443)
