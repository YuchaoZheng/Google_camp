from torch.utils.data import DataLoader, RandomSampler
from sklearn.model_selection import train_test_split
import pandas as pd
from src import *
import torch
import torch.nn as nn
from itertools import cycle
import time

BATCH_SIZE = 4 
STEPS = 100000000
LR = 1e-5
EVAL_STEP = 1000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_proportion = 0.1
eval_proportion = 0.15

df_data = pd.read_csv("/home/yuchaozheng_zz/Google_camp/segmentation/df_data.csv")

# df_data = df_data.sample(10000, random_state=101)
# print(df_data.shape)


NUM_TEST_IMAGES = int(df_data.shape[0] * test_proportion)
df_test = df_data.sample(NUM_TEST_IMAGES, random_state=101)
df_test = df_test.reset_index(drop=True)

test_images_list = list(df_test['image_name'])
df_data = df_data[~df_data['image_name'].isin(test_images_list)]
df_train, df_val = train_test_split(df_data, test_size=eval_proportion, random_state=101)

test_len = df_test.shape[0]
eval_len = df_val.shape[0]

print(df_train.shape)
print(df_val.shape)
print(df_test.shape)

trainsets = MattingHumanDataset(df_train)
train_sampler = RandomSampler(trainsets)

train_loader = DataLoader(trainsets, sampler=train_sampler, batch_size=BATCH_SIZE)
epoch_step = len(train_loader)
train_loader = cycle(train_loader)


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
    preds = torch.zeros((data_len, 800, 600))
    targets = torch.zeros((data_len, 800, 600))
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
model = UNet()
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print('iter   epoch  | valid_loss  valid_iou | test_loss  test_iou|  train_loss |  time')
print('---------------------------------------------------------------------------------')

start_time = time.time()
sum_loss = 0
best_eval_m_iou = 0

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

        eval_loss, eval_m_iou = valid(model, evalloader, criterion, eval_len)
        test_loss, test_m_iou = valid(model, testloader, criterion, test_len)
        
        if best_eval_m_iou < eval_m_iou:
            torch.save(model.state_dict(), "./best.pth")
            best_eval_m_ioy = eval_m_iou

        model.train()
        elapsed_time = time.time() - start_time

        print(
            '{:.1f}k  {:.1f}  |  {:.4f}  {:.4f} |  {:.4f}  {:.4f} | {:.4f}  | {:.1f}second'.format(
                step / 1000, step / epoch_step, eval_loss, eval_m_iou, test_loss, test_m_iou, sum_loss / (step + 1),
                elapsed_time))
