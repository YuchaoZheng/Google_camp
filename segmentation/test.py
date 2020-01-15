from torch.utils.data import DataLoader, RandomSampler
from sklearn.model_selection import train_test_split
import pandas as pd
from src import *
import torch
import torch.nn as nn
import os
import numpy as np
import cv2
import torch
from itertools import cycle
import time
from tqdm import tqdm
import random

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

def iou(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)

    pred_inds = (pred == 1)
    target_inds = (target == 1)
    intersection = (pred_inds[target_inds]).long().sum().item()  # Cast to long to prevent overflows
    union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
    if union == 0:
        ious = float('nan')
    else:
        ious = float(intersection) / union

    return ious

seed_everything(1025)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_file = "/home/yuchaozheng_zz/Google_camp/segmentation/best.pth"

model = UNet()
model = model.to(device)

model.load_state_dict(torch.load(model_file))
model.eval()

def get_img(img_path):
    img = cv2.imread(img_path)
    return img

img_path = "/home/yuchaozheng_zz/test.jpg"

img = get_img(img_path)

img = torch.from_numpy(img)
img = img.permute(2, 0, 1).float()
img /= 255.

img = torch.unsqueeze(img, 0)

with torch.no_grad():
    img = img.to(device)

    pred = model(img)

    pred = torch.argmax(pred.cpu(), dim=1)

    img = get_img(img_path)
    pred = torch.unsqueeze(pred, 1)
    pred = torch.squeeze(pred, 0)
    # c, h, w -> h, w, c
    pred = pred.permute(1, 2, 0).numpy()
    alpha_preds = pred * 255

    predicted_masks = np.concatenate((img, alpha_preds), axis=-1)
    cv2.imwrite('/home/yuchaozheng_zz/result.png', predicted_masks)
