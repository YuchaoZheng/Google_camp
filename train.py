from torch.utils.data import DataLoader, RandomSampler
from sklearn.model_selection import train_test_split
import pandas as pd
from src import *
import torch
import torch.nn as nn
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

seed_everything(1025)

BATCH_SIZE = 8 
EPOCH = 100
LR = 1e-4
EVAL_STEP = 1500
IMG_SIZE_W = 512
IMG_SIZE_H = 512
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device ", device)
test_proportion = 0.1
eval_proportion = 0.1

df_data = pd.read_csv("/home/yuchaozheng_zz/Google_camp/df_data.csv")

# df_data = df_data.sample(10000, random_state=101)
# print(df_data.shape)


NUM_TEST_IMAGES = int(df_data.shape[0] * test_proportion)
df_test = df_data.sample(NUM_TEST_IMAGES, random_state=101)
df_test = df_test.reset_index(drop=True)

test_images_list = list(df_test['image_name'])

df_test.to_csv("test.csv")

df_data = df_data[~df_data['image_name'].isin(test_images_list)]
df_train, df_val = train_test_split(df_data, test_size=eval_proportion, random_state=101)

test_len = df_test.shape[0]
eval_len = df_val.shape[0]

print(df_train.shape)
print(df_val.shape)
print(df_test.shape)

trainsets = MattingHumanDataset(df_train, img_size_h=IMG_SIZE_H, img_size_w=IMG_SIZE_W)
trainloader = torch.utils.data.DataLoader(trainsets,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True,
                                         drop_last=True)
epoch_step = len(trainloader)

evalsets = MattingHumanDataset(df_val, img_size_h=IMG_SIZE_H, img_size_w=IMG_SIZE_W)
evalloader = torch.utils.data.DataLoader(evalsets,
                                         batch_size=BATCH_SIZE,
                                         shuffle=False,
                                         drop_last=False)

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


def valid(model, evalloader, criterion, data_len):
    preds = torch.zeros((data_len, IMG_SIZE_H, IMG_SIZE_W))
    targets = torch.zeros((data_len,IMG_SIZE_H, IMG_SIZE_W))
    sum_val_loss = 0

    with torch.no_grad():
        for batch_idx, (img, mask) in enumerate(evalloader):
            img = img.to(device)
            mask = mask.to(device)
            pred = model(img)

            loss = criterion(pred, mask)
            sum_val_loss += loss.item()

            pred = torch.argmax(pred.cpu(), dim=1)
            mask = torch.argmax(mask.cpu(), dim=1)
            preds[batch_idx*BATCH_SIZE:min(data_len, (batch_idx+1) * BATCH_SIZE), :, :] = pred
            targets[batch_idx*BATCH_SIZE:min(data_len, (batch_idx+1) * BATCH_SIZE), :, :] = mask

    return sum_val_loss / int(len(evalloader)), iou(preds, targets)


criterion = nn.BCEWithLogitsLoss()
model = Unet("resnet34", encoder_weights="imagenet", classes=2, activation=None)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print('iter   epoch  | valid_loss  valid_iou |  train_loss |  time')
print('---------------------------------------------------------------------------------')

start_time = time.time()
sum_loss = 0
best_eval_m_iou = 0
step = 0

for epoch in range(EPOCH):
    for (img, mask) in trainloader:
        optimizer.zero_grad()
        img = img.to(device)
        mask = mask.to(device)
        pred = model(img)
        loss = criterion(pred, mask)

        sum_loss += loss.item()
        loss.backward()
        optimizer.step()

        step += 1

        if step % EVAL_STEP == 0:
            model.eval()
            eval_loss, eval_m_iou = valid(model, evalloader, criterion, eval_len)

            if best_eval_m_iou < eval_m_iou:
                torch.save(model.state_dict(), "./new_resnet_unet_best.pth")
                best_eval_m_iou = eval_m_iou

            model.train()
            elapsed_time = time.time() - start_time

            print(
                '{:.1f}k  {:.1f}  |  {:.4f}  {:.4f} | {:.4f}  | {:.1f}second'.format(
                    step / 1000, step / epoch_step, eval_loss, eval_m_iou, sum_loss / (step + 1),
                    elapsed_time))
