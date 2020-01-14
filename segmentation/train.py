import pandas as pd
from src import *
import torch
import torch.nn as nn
from itertools import cycle
import time

BATCH_SIZE = 8
STEPS = 100000000
LR = 1e-5
EVAL_STEP = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df_data = pandas.read_csv("/home/yuchaozheng_zz/Google_camp/segmentation/df_data.csv")

df_data = df_data.sample(100000, random_state=101)
print(df_data.shape)


NUM_TEST_IMAGES = int(df_data.shape * 0.1)
df_test = df_data.sample(NUM_TEST_IMAGES, random_state=101)
df_test = df_test.reset_index(drop=True)

test_images_list = list(df_test['image_name'])
df_data = df_data[~df_data['image_name'].isin(test_images_list)]
df_train, df_val = train_test_split(df_data, test_size=0.15, random_state=101)

test_len = df_test.shape[0]
eval_len = df_eval.shape[0]

print(df_train.shape)
print(df_val.shape)
print(df_test.shape)

trainsets = MattingHumanDataset(df_train)

trainloader = torch.utils.data.DataLoader(trainsets,
                                          num_workers=8,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          drop_last=True)

evalsets = MattingHumanDataset(df_val)
evalloader = torch.utils.data.DataLoader(evalsets,
                                         num_workers=8,
                                         batch_size=BATCH_SIZE,
                                         shuffle=False,
                                         drop_last=False)

tests = MattingHumanDataset(df_test)
testloader = torch.utils.data.DataLoader(tests,
                                         num_workers=8,
                                         batch_size=BATCH_SIZE,
                                         shuffle=False,
                                         drop_last=False)


def iou(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)

    pred_inds = pred == 1
    target_inds = target == 1
    intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]  # Cast to long to prevent overflows
    union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
    if union == 0:
        ious = float('nan')
    else:
        ious = float(intersection)

    return ious


def valid(model, evalloader, len):
    preds = torch.zeros((len, 800, 600))
    targets = torch.zeros((len, 800, 699))

    with torch.no_grad():
        for batch_idx, (img, mask) in enumerate(evalloader):
            img = img.to(device)
            pred = model(img)

            pred = torch.argmax(pred.cpu(), dim=1)

            preds[batch_idx:min(len, batch_idx+BATCH_SIZE), :, :, :] = pred
            targets[batch_idx:min(len, batch_idx + BATCH_SIZE), :, :, :] = mask

    return iou(preds, targets)


criterion = nn.BCEWithLogitsLoss()
model = UNet()
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

epoch_step = len(trainloader)
trainloader = cycle(trainloader)

print('iter   epoch    | valid_loss  valid_iou | test_loss  test_iou|  train_loss |  time')
print(
    '------------------------------------------------------------------------------------------------------')

start_time = time.time()
sum_loss = 0

for step in range(STEPS):
    data = next(train_loader)
    optimizer.zero_grad()
    img, mask = data
    img = img.to(device)
    mask = mask.to(device)

    pred = model(img)
    loss = criterion(pred, mask)

    sum_loss += loss.item()
    loss.backward()
    optimizer.step()

    if (step + 1) % EVAL_STEP == 0:
        model.eval()

        eval_loss, eval_m_iou = valid(model, evalloader, eval_len)
        test_loss, test_m_iou = valid(model, testloader, test_len)

        model.train()
        elapsed_time = time.time() - start_time

        print(
            '{:.1f}k  {:.1f}  |  {:.4f}  {:.4f} |  {:.4f}  {:.4f} | {:.4f}  | {:.1f}second'.format(
                step / 1000, step / epoch_step, val_loss, eval_m_iou, test_loss, test_m_iou, sum_loss / (step + 1),
                elapsed_time))
