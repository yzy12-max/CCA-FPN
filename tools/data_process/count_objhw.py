# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 21:48:48 2021

@author: YaoYee
"""

import os
import xml.etree.cElementTree as et
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# path = '/data/wjh/datasets/forUser_A/train/box'  #
path = '/data1/yzycode/datasets/VOCdevkit/VOC2007/Annotations' #
files = os.listdir(path) #顺序提取在一个列表里

area_list = []
ratio_list = []


def file_extension(path):
    return os.path.splitext(path)[1]


for xmlFile in tqdm(files, desc='Processing'):
    if not os.path.isdir(xmlFile): #判断文件是否为目录
        if file_extension(xmlFile) == '.xml':
            tree = et.parse(os.path.join(path, xmlFile))
            root = tree.getroot()

            # filename = root.find('filename').text
            # filename = root.find('name').text
            # print("--Filename is", xmlFile)

            for Object in root.findall('object'):
                bndbox = Object.find('bndbox')
                xmin = bndbox.find('xmin').text
                ymin = bndbox.find('ymin').text
                xmax = bndbox.find('xmax').text
                ymax = bndbox.find('ymax').text

                h = abs(int(ymax) - int(ymin))
                w = abs(int(xmax) - int(xmin))
                area = h * w
                area_list.append(area)
                # print("Area is", area)

                ratio = h / w   # h/w
                ratio_list.append(ratio)
                # print("Ratio is", round(ratio,2))
print(min(area_list),max(area_list))
# area_results = pd.value_counts(area_list, normalize=True)
# area_results.to_csv('area_results.csv', header=0)
# area_results = pd.value_counts(ratio_list, normalize=True)
# area_results.to_csv('ratio_list.csv', header=0)
square_array = np.array(area_list)
square_max = np.max(square_array)
square_min = np.min(square_array)
square_mean = np.mean(square_array)
square_var = np.var(square_array)
a,b,c,d=0,0,0,0
for i in range(len(area_list)):
    if 0<area_list[i]<1024:     # 32*32
        a += 1
    elif 1024<area_list[i]<9216:    # 96*96
        b += 1
    elif 9216<area_list[i]:
        c += 1
    elif area_list[i]<0:
        d += 1
# print(a,b,c,d)
nums = [a,b,c]
area_results = pd.value_counts(nums, normalize=True)
area_results.to_csv('acou_nums.csv', header=0)

plt.figure(1)
#plt.bar(x,square_array)
#bins=np.arange(0,6,1)#设置连续的边界值，即直方图的分布区间[0,10],[10,20]...
plt.hist(square_array, 20)
plt.xlabel('Area in pixel`')
plt.ylabel('Frequency of area')
plt.title('Area\n' \
          + 'max=' + str(square_max) + ', min=' + str(square_min) + '\n' \
          + 'mean=' + str(int(square_mean)) + ', var=' + str(int(square_var))
          )    # 生成的图片标题中加入了最值，均值和方差
plt.xlim(0,20000)  # 设置x轴分布范围
# plt.xticks([0,1,2,3])
ratio_array = np.array(ratio_list)  #
ratio_max = np.max(ratio_array)
ratio_min = np.min(ratio_array)
ratio_mean = np.mean(ratio_array)
ratio_var = np.var(ratio_array)
plt.figure(2)
plt.hist(ratio_array,10,(0,8))
plt.xlabel('Ratio of length / width')
plt.ylabel('Frequency of ratio')
plt.title('Ratio\n' \
          + 'max=' + str(round(ratio_max, 2)) + ', min=' + str(round(ratio_min, 2)) + '\n' \
          + 'mean=' + str(round(ratio_mean, 2)) + ', var=' + str(round(ratio_var, 2))
          )
plt.show()  #图不太好看，可以自己写代码统计
pass

# plt.figure(figsize = (6, 4)) #新建画布
#     n, bins, patches = plt.hist(diff, bins = [0, 0.01,  0.03, 0.05, 0.1], color='brown', alpha = 0.8, label = "直方图" ) #绘制直方图
#     for i in range(len(n)):
#         plt.text(bins[i]*1.0, n[i]*1.01, n[i]) #打标签，在合适的位置标注每个直方图上面样本数
#     plt.grid(alpha = 0.5) #添加网格线
#     plt.xlabel("误差率")
#     plt.ylabel("案例数")
#     plt.title("柱状分布统计图")
#     plt.show()

#原文链接：https://blog.csdn.net/zengbowengood/article/details/108780582