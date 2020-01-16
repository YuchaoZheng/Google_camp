import pandas as pd
import os 

def make_csv(path, train = False):
    folder_list = os.listdir(path)
    img_name = []
    base_dir = []

    for f in folder_list:
        path1 = os.path.join(path, f)

        # 1803151818
        folder_list1 = os.listdir(path1)

        for f1 in folder_list1:
            # clip_00000000
            path2 = os.path.join(path1, f1)

            image_list = os.listdir(path2)

            for img in image_list:
                img_name.append(img)
                base_dir.append(path2)


    if train:
        result = {"image_name": img_name,
             "img_base_dir": base_dir}
    else:
        result = {"mask_name": img_name,
                  "mask_base_dir": base_dir}

    df = pd.DataFrame(result)

    return df

train_dir = '/home/yuchaozheng_zz/ML-camp/MattingHuman/aisegmentcom-matting-human-datasets/clip_img'

df_img = make_csv(train_dir, True)

test_dir = '/home/yuchaozheng_zz/ML-camp/MattingHuman/aisegmentcom-matting-human-datasets/matting'
df_mask = make_csv(test_dir, False)

print('Num Images: ', df_img.shape[0])
print('Num Masks: ', df_mask.shape[0])


def get_name(x):
    name = x.split('.')[0]
    return name


df_img['merge_col'] = df_img['image_name'].apply(get_name)
df_mask['merge_col'] = df_mask['mask_name'].apply(get_name)
print(len(df_img['merge_col'].unique()))
df_data = pd.merge(df_img, df_mask, on='merge_col')

# save as a compressed csv file
df_data.to_csv('df_data.csv', index=False)
print(df_data.shape)
print(df_data.columns)
print(df_data.head())
