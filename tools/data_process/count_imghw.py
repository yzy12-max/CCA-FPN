#
import os
import pandas as pd
from PIL import Image
import tqdm
from collections import Counter

# source = '/data/wjh/datasets/forUser_A/voice-train/image'  #
source = '/data1/yzycode/datasets/VOCdevkit/VOC2007/JPEGImages'
imgfile = os.listdir(source)
shot_cut = []
long_cut = []
area_list = []
long_short = []
print('filenum:', len([lists for lists in os.listdir(source)])) #打印总数量

for img in tqdm.tqdm(imgfile):
    img_path = os.path.join(source, img)
    img = Image.open(img_path)
    imgSize = img.size  # 图片的长和宽
    #print(imgSize)
    maxSize = max(imgSize)  # 图片的长边
    long_cut.append(maxSize)
    minSize = min(imgSize)  # 图片的短边
    shot_cut.append(minSize)
    area = maxSize*minSize
    area_list.append(area)
    long_short.append((maxSize,minSize))
print(min(area_list),max(area_list))
#shot_result = Counter(shot_cut)
#long_result = Counter(long_cut)
shot_result = pd.value_counts(shot_cut, normalize=True)
long_result = pd.value_counts(long_cut, normalize=True)
long_short_result = pd.value_counts(long_short, normalize=True)
area_result = pd.value_counts(area_list, normalize=True)
# print(shot_result)
# print(long_result)

# shot_result.to_csv('optics_short.csv', header=0)
# long_result.to_csv('optics_long.csv', header=0)
long_short_result.to_csv('acoustics_long_short.csv', header=0)
# area_result.to_csv('optics_imgarea.csv', header=0)