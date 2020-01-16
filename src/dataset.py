import numpy as np
from torch.utils.data import Dataset
import os
import cv2
import torch

# def img_aug(img, mask):
#     mask = np.where(mask > 0, 0, 255).astype(np.uint8)
#     # 水平翻转
#     if random.random() < 0.5:
#         flipper = iaa.Fliplr(0.5).to_deterministic()
#         mask = flipper.augment_image(mask)
#         img = flipper.augment_image(img)
#
#     # 上下翻转
#     if random.random() < 0.5:
#         vflipper = iaa.Flipud(0.5).to_deterministic()
#         img = vflipper.augment_image(img)
#         mask = vflipper.augment_image(mask)
#
#     # 旋转
#     if random.random() < 0.5:
#         rot_time = random.choice([1, 2, 3])
#         for _ in range(rot_time):
#             img = np.rot90(img)
#             mask = np.rot90(mask)
#
#     mask = np.where(mask > 0, 0, 255).astype(np.uint8)
#     return img, mask


class MattingHumanDataset(Dataset):
    def __init__(self, data_frame, img_size_h=512, img_size_w=512):
        super(MattingHumanDataset, self).__init__()
        self.img_size_h = img_size_h
        self.img_size_w = img_size_w
        self.img_path = []
        self.mask_path = []
        for _, row in data_frame.iterrows():
            self.img_path.append(os.path.join(row['img_base_dir'], row['image_name']))
            self.mask_path.append(os.path.join(row['mask_base_dir'], row['mask_name']))


    def _get_mask(self, label_path):
        mask = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        mask = mask[:, :, 3]
        mask = cv2.resize(mask, (self.img_size_w, self.img_size_h))
        
        mask[mask[:, :] > 0] = 1

        h, w = mask.shape
        masks = np.zeros((h, w, 2))
        masks[mask == 1, 1] = 1
        masks[mask == 0, 0] = 1
        return masks

    def _get_img(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.img_size_w, self.img_size_h))
        return img

    def __getitem__(self, idx):
        img = self._get_img(self.img_path[idx])
        mask = self._get_mask(self.mask_path[idx])
    
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
        img = img.permute(2, 0, 1).float()
        mask = mask.permute(2, 0, 1).float()
        return img / 255., mask

    def __len__(self):
        return len(self.img_path)
