from torch.utils.data import Dataset
import numpy as np
import os
import cv2
import random
import torch
import glob
from imgaug import augmenters as iaa

def img_aug(img, mask):
    mask = np.where(mask > 0, 0, 255).astype(np.uint8)
    # 水平翻转
    if random.random() < 0.5:
        flipper = iaa.Fliplr(0.5).to_deterministic()
        mask = flipper.augment_image(mask)
        img = flipper.augment_image(img)

    # 上下翻转
    if random.random() < 0.5:
        vflipper = iaa.Flipud(0.5).to_deterministic()
        img = vflipper.augment_image(img)
        mask = vflipper.augment_image(mask)

    # 旋转
    if random.random() < 0.5:
        rot_time = random.choice([1, 2, 3])
        for _ in range(rot_time):
            img = np.rot90(img)
            mask = np.rot90(mask)

    mask = np.where(mask > 0, 0, 255).astype(np.uint8)
    return img, mask


class MattingHumanDataset(Dataset):
    def __init__(self, img_path, train=True, img_size=768, class_num=4):
        super(MattingHumanDataset, self).__init__()
        self.class_num = class_num + 1
        self.img_size = img_size
        self.train = train
        self.test = False

        self.img_path = img_path
        self.label_list = sorted(glob.glob(os.path.join(img_path, '*label*.png')))

    def _get_mask(self, label_path):
        img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        mask = np.zeros((h, w, self.class_num))
        for row in range(h):  # 遍历每一行
            for col in range(w):  # 遍历每一列
                mask[row, col, int(img[row][col])] = 1
        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        return mask

    def _get_img(self, img_path):
        img = cv2.imread(img_path)
        return img

    def perpare_train_val(self, idx):
        label_path = self.label_list[idx]
        img_path = label_path.replace('_label', '')
        # print("img_path {} label_path {}".format(img_path, label_path))
        img = self._get_img(img_path)
        mask = self._get_mask(label_path)
        if self.train:
            img, mask = img_aug(img, mask)

        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
        img = img.permute(2, 0, 1).float()
        mask = mask.permute(2, 0, 1).float()
        return img, mask

    def perpare_test(self, idx):
        img = self._get_img(idx)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1).float()
        return img

    def __getitem__(self, idx):
        img, mask = self.perpare_train_val(idx)
        return img / 255., mask / 255.

    def __len__(self):
        return len(self.label_list)
