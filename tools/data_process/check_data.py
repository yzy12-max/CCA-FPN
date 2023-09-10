import numpy as np
from tqdm import tqdm
import os


path = '/data1/yzycode/datasets/VOCdevkit/VOC2007/Annotations'
path1 = '/data1/yzycode/datasets/VOCdevkit/VOC2007/box'
files = os.listdir(path) #顺序提取在一个列表里
files1 = os.listdir(path1) #顺序提取在一个列表里
for i in range(0,5543):
    if files[i]!=files1[i]:
        print(files1[i])
# for f in tqdm(files):
#     strlist = f.split('.')
#     num = int(strlist[0])
#     if num not in range(1,5544):
#         print(num)
pass