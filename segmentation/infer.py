from torch.utils.data import DataLoader, RandomSampler
from sklearn.model_selection import train_test_split
import pandas as pd
from src import *
import torch
import torch.nn as nn
import os
import cv2
import torch
import numpy as np
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

BATCH_SIZE = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df_test = pd.read_csv("/home/yuchaozheng_zz/Google_camp/segmentation/test.csv")

model_file = "/home/yuchaozheng_zz/Google_camp/segmentation/best.pth"

model = UNet()
model = model.to(device)

model.load_state_dict(torch.load(model_file))
model.eval()

def get_mask(label_path):
    mask = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
    mask = mask[:, :, 3]
    mask[mask[:, :] > 0] = 1

    h, w = mask.shape
    masks = np.zeros((h, w, 2))
    masks[mask == 1, 1] = 1
    masks[mask == 0, 0] = 1
    return masks

def get_img(img_path):
    img = cv2.imread(img_path)
    return img

for _, row in df_test.iterrows():
    img_path = os.path.join(row['img_base_dir'], row['image_name'])
    mask_path = os.path.join(row['mask_base_dir'], row['mask_name'])

    img = get_img(img_path)
    mask = get_mask(mask_path)

    img = torch.from_numpy(img)
    mask = torch.from_numpy(mask)
    img = img.permute(2, 0, 1).float()
    mask = mask.permute(2, 0, 1).float()
    img /= 255.

    img = torch.unsqueeze(img, 0)
    mask = torch.unsqueeze(mask, 0)

    with torch.no_grad():
        img = img.to(device)

        pred = model(img)

        pred = torch.argmax(pred.cpu(), dim=1)
        mask = torch.argmax(mask, dim=1)
        
        m_iou = iou(pred, mask)
        if m_iou >= 0.995:
            output_file = "/home/yuchaozheng_zz/result/result_{}".format(row['mask_name']) 
            print(img_path, output_file, m_iou)
            img = get_img(img_path)
            # b, h ,w -> b, c, h, w
            pred = torch.unsqueeze(pred, 1)
            pred = torch.squeeze(pred, 0)
            # c, h, w -> h, w, c
            pred = pred.permute(1, 2, 0).numpy()
            alpha_preds = pred * 255

            predicted_masks = np.concatenate((img, alpha_preds), axis=-1)

            cv2.imwrite(output_file, predicted_masks)

