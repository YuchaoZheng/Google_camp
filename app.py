import io
import json
from src import *
from beautify import Makeup
import time
import base64
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

model_file = "/home/yuchaozheng_zz/Google_camp/best.pth"
model = UNet()
model.load_state_dict(torch.load(model_file))
model.eval()
if use_gpu:
    model.cuda()


@app.route('/')
def home():
    return flask.render_template("index.html")


def get_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (600, 800), interpolation=cv2.INTER_NEAREST)
    return img


def prepare_image(img_path, target_size):
    img = get_image(img_path)
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
    st_time = time.clock()
    print("11111")
    if flask.request.method == 'POST':
        print("PPPPPPP")
        image = base64.urlsafe_b64decode(request.form['img'])
        image = Image.open(io.BytesIO(image))

        img_path = "./receive.jpg"
        image.save(img_path)

        data = json.load(flask.request.files["data"])
        whighten = request.form['whighten']
        lip_brighten = request.form['lip_brighten']
        thin = request.form['thin']
        smooth = request.form['smooth']
        sharpen = request.form['sharpen']

        raw_img = get_image(img_path)
        makeup_obj = Makeup(raw_img)
        results = makeup_obj.beautify(smooth_val=smooth, whiten_val=whighten,
                                      lip_brighten_val=lip_brighten, sharpen_val=sharpen, thin_val=thin)

        print(time.clock() - st_time)
        img_path = "./beauty_result.jpg"
        cv2.imwrite(img_path, results)
        # Read the image in PIL format

        # Preprocess the image and prepare it for classification.
        img = prepare_image(img_path, target_size=(800, 600))

        with torch.no_grad():
            pred = model(img)
            pred = torch.argmax(pred.cpu(), dim=1)

            img = get_image(img_path)
            pred = torch.unsqueeze(pred, 1)
            pred = torch.squeeze(pred, 0)

            # c, h, w -> h, w, c
            pred = pred.permute(1, 2, 0).numpy()
            alpha_preds = pred * 255
            # kernel = np.ones((10, 10), np.uint8)
            # alpha_preds = cv2.dilate(alpha_preds, kernel, iterations=1)
            # alpha_preds = cv2.erode(alpha_preds, kernel, iterations=1)

            img = np.concatenate((img, alpha_preds), axis=-1)

            cv2.imwrite('./result.png', img)

            img = Image.open(r'./result.png').convert("RGBA")
            x, y = img.size
            card = Image.new("RGBA", img.size, (67, 142, 219))
            card.paste(img, (0, 0, x, y), img)
            card.save("result.png", format="png")

            result = open('./result.png', 'rb').read()
            print(time.clock() - st_time)

            return result
    # Return the data dictionary as a JSON response.
    return None


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=443)
