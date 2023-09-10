import os
import xml.etree.cElementTree as et
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import json

# JSON路径
path = "/home/yzy/datasets/coco2/annotations/instances_train2017.json"
f = open(path,"r")
results = json.load(f)
annotations = results["annotations"]
areas_list = []
for anno in annotations:
    bbox = anno["bbox"]
    h = bbox[-2]
    w = bbox[-1]
    area = h*w
    areas_list.append(area)
a,b,c,d=0,0,0,0
for i in tqdm(range(len(areas_list))):
    if 0<areas_list[i]<1024:     # 32*32
        a += 1
    elif 1024<areas_list[i]<9216:    # 96*96
        b += 1
    elif 9216<areas_list[i]:
        c += 1
    elif areas_list[i]<0:
        d += 1
nums = [a,b,c]
area_results = pd.value_counts(nums, normalize=True)
area_results.to_csv('coco_nums.csv', header=0)

pass



