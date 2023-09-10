import numpy as np
from tqdm import tqdm
import os
from glob import glob
import mmcv
from PIL import Image
import cv2
from mmdet.core.visualization import imshow_det_bboxes
import xml.etree.ElementTree as ET
#

class_names={'holothurian':0,'echinus':1,'scallop':2,'starfish':3}
underwater_classes = ['holothurian', 'echinus', 'scallop', 'starfish']
def bbox_iou(box1, box2):
    """
    Calculate the IOU between box1 and box2.

    :param boxes: 2-d array, shape(n, 4)
    :param anchors: 2-d array, shape(k, 4)
    :return: 2-d array, shape(n, k)
    """
    # Calculate the intersection,
    # the new dimension are added to construct shape (n, 1) and shape (1, k),
    # so we can get (n, k) shape result by numpy broadcast
    box1 = box1[:, np.newaxis]  # [n, 1, 4]
    box2 = box2[np.newaxis]     # [1, k, 4]

    xx1 = np.maximum(box1[:, :, 0], box2[:, :, 0])
    yy1 = np.maximum(box1[:, :, 1], box2[:, :, 1])
    xx2 = np.minimum(box1[:, :, 2], box2[:, :, 2])
    yy2 = np.minimum(box1[:, :, 3], box2[:, :, 3])
    w = np.maximum(0, xx2-xx1+1)
    h = np.maximum(0, yy2-yy1+1)
    inter = w * h
    area1 = (box1[:, :, 2] - box1[:, :, 0] + 1) * (box1[:, :, 3] - box1[:, :, 1] + 1)
    area2 = (box2[:, :, 2] - box2[:, :, 0] + 1) * (box2[:, :, 3] - box2[:, :, 1] + 1)
    ious = inter / (area1 + area2 - inter)

    return ious

# 转成YOLOV5的数据格式
def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[2])/2.0 - 1
    y = (box[1] + box[3])/2.0 - 1
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def data_clean(res_path,label_path,out_path,xml_path):
    # 转成列表
    # res_lists = os.listdir(res_path)
    # label_lists = os.listdir(label_path)    # 检查错误的时候解除注释,标签文件所有列表
    # 检查出错误之后,使用的列表,检查错误时注释掉
    # label_lists=['000149.txt','000249.txt','000147.txt','001539.txt','001627.txt','001677.txt','001753.txt','001754.txt','001767.txt','001769.txt',
    # '001770.txt','001773.txt','001774.txt','001775.txt','001839.txt','001970.txt','002042.txt','002053.txt','002094.txt','002145.txt','002331.txt','002377.txt',
    # '002436.txt','002558.txt','002732.txt','002746.txt','002758.txt','002771.txt','002839.txt','003014.txt','003117.txt','003169.txt','003170.txt','003259.txt',
    # '003415.txt','003529.txt','003539.txt','003550.txt','003672.txt','003695.txt','003758.txt','003795.txt','003813.txt','003819.txt','003820.txt','003833.txt','004153.txt','004158.txt',
    # '004418.txt','004722.txt','004723.txt','004854.txt','004879.txt','004880.txt','004907.txt','004936.txt','004949.txt','005071.txt','005075.txt','005115.txt','005210.txt','005536.txt']
    label_lists = [
    'c001423.txt', 'c001929.txt', 'c002330.txt', 'u000124.txt', 'u000216.txt', 'u000683.txt', 'u001152.txt', 'u001413.txt', 'u001489.txt', 'u001605.txt',
    'u001606.txt','u001626.txt', 'u001683.txt', 'u001870.txt', 'u001880.txt', 'u001916.txt', 'u002289.txt', 'u002378.txt', 'u002434.txt', 'u002524.txt', 'u002668.txt',
    'u002805.txt', 'u002806.txt', 'u002886.txt', 'u002893.txt', 'u003014.txt', 'u003108.txt', 'u003117.txt', 'u003126.txt', 'u003230.txt', 'u003251.txt',
    'u003334.txt','u003350.txt', 'u003355.txt', 'u003356.txt', 'u003367.txt', 'u003867.txt', 'u004115.txt', 'u004116.txt', 'u004226.txt', 'u004246.txt',
    'u004247.txt', 'u004458.txt', 'u004536.txt', 'u005021.txt', 'u005333.txt', 'u005492.txt', 'u005566.txt', 'u005582.txt', 'u005609.txt', 'u005681.txt',
    'u006019.txt', 'u006360.txt', 'u006391.txt', 'u006505.txt',
    ]
    # 将标签
    for label_list in tqdm(label_lists):
        # 获取GT坐标和类别,用列表表示
        flag = False
        gt =[]
        pre = []
        # 错误标签保存
        wrong_label = []
        with open(os.path.join(label_path,label_list),'r') as f:
            for line in f.readlines():
                li = line.strip('\n')
                # 字符串根据空格进行分割
                li = li.split(' ')
                gt.append([int(li[0]),int(li[1]),int(li[2]),int(li[3]),int(li[4])])

        # 如果对应图片预测到了,获取预测坐标和类别,用列表表示
        if os.path.exists(os.path.join(res_path,label_list)):
            with open(os.path.join(res_path,label_list),'r') as f1:
                for line1 in f1.readlines():
                    li1 = line1.strip('\n')
                    # 字符串根据空格进行分割
                    li1 = li1.split(',')
                    #
                    pre.append([class_names[li1[0]],int(li1[1]),int(li1[2]),int(li1[3]),int(li1[4]),float(li1[5])])
        # 求两者的IOU
        # 转为np.array,方便维度索引
        gt = np.array(gt)
        pre = np.array(pre)
        # 如果预测不到目标,就不筛选
        if len(pre)>0:
            pre = pre[pre[:, 5] > 0.1]
            ious = bbox_iou(pre[:,1:5], gt[:,1:])  # [n, k],计算IOU
            max_idx = np.argmax(ious, axis=0)  # [k,],返回最大索引,axis=0表示最大值在第0维的索引值,即第几行
            max_value = np.amax(ious, axis=0)  # [k,],返回最大IOU值
            pre = pre[max_idx]  # gt 对应的 pred box,预测的BOX根据索引取出
            for i in range(len(gt)):  #
                if gt[i][0] != pre[i][0] and max_value[i] >= 0.75: # 如果预测的类别不同GT且IOU>0.6
                    wrong_label.append(pre[i])
                    flag = True
                    print('wrong_labels:',label_list)
                    # 修改,然后保存到输出文件夹里,这个要自己可视化查看确实是错误的时候使用,第一次使用时注释掉下面的保存文件代码
                    line2 = list(gt[i]) # 修改成元素可变的列表类型
                    temp = list(pre[i])
                    line2[0] = int(temp[0])
                    #
                    xml_file = os.path.join(xml_path, label_list.split('.')[0] + '.xml')
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    size = root.find('size')
                    w = int(size.find('width').text)
                    h = int(size.find('height').text)
                    line3 = convert((w, h), line2[1:])
                    with open(os.path.join(out_path, label_list), 'a') as f2:
                        # 使用" ".join([str(a) for a in line3])能够保留更多小数,str()的效果
                        f2.write(str(line2[0]) + " " + " ".join([str(a) for a in line3]) + '\n') #('%g ' * len(line3)).rstrip() % line3 + '\n')
                else:
                    # 同理，第一次使用时注释掉下面的保存文件代码
                    xml_file = os.path.join(xml_path, label_list.split('.')[0] + '.xml')
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    size = root.find('size')
                    w = int(size.find('width').text)
                    h = int(size.find('height').text)
                    box = convert((w, h), gt[i][1:])
                    with open(os.path.join(out_path, label_list), 'a') as f2:
                        # 使用" ".join([str(a) for a in line3])能够保留更多小数,str()的效果
                        f2.write(str(gt[i][0]) + " " + " ".join([str(a) for a in box]) + '\n')
                    continue
            # 第一次检查错误的标注时候取消注释
            # if flag:    # 说明有错误标注的
            #     diff_boxes = np.array(wrong_label)
            #     # print(imgid2name[imgid])
            #     filename = os.path.join('/home/yzy/datasets/underwater/forUser_A/train/image', label_list.split('.')[0]+'.jpg')
            #     img = cv2.imread(filename)
            #     basename = os.path.basename(filename)   # 返回文件名
            #     imshow_det_bboxes(img, gt[:,1:], gt[:,0], class_names=underwater_classes,
            #                       show=False,
            #                       out_file=os.path.join('/home/yzy/datasets/underwater/forUser_A/train/nosiy_image/' + basename))
            #     imshow_det_bboxes(img, diff_boxes[:, 1:], diff_boxes[:, 0].astype(np.int), class_names=underwater_classes,
            #                       bbox_color='red',
            #                       text_color='red',
            #                       show=False,
            #                       out_file=os.path.join('/home/yzy/datasets/underwater/forUser_A/train/nosiy_image-pred/' + basename))
            #     imshow_det_bboxes(img, pre[:, 1:], pre[:, 0].astype(np.int), class_names=underwater_classes,
            #                       bbox_color='red',
            #                       text_color='red',
            #                       show=False,
            #                       out_file=os.path.join('/home/yzy/datasets/underwater/forUser_A/train/image-pred/' + basename))




def main():
    print('all train data!')
    # 模型预测所有数据集的结果文件夹
    res_path = '/data1/yzycode/yolov5-master/runs/detect/2022underwater/labels/'   #223,没有
    # 真实的标注信息文件夹,直接用VOC格式的
    label_path = '/home/yzy/datasets/underwater/forUser_A/train/labels/'
    print('processing {} ...'.format("labels"))
    # img_root_path = '/data1/yzycode/datasets/yolo_water2020/images/'
    # xml path
    xml_path = '/home/yzy/datasets/yolo_underwater/box/'
    # 输出文件
    out_path = '/home/yzy/datasets/yolo_underwater/labels_bn/'
    if os.path.exists(out_path):
        print('output path is exists!')
    else:
        os.makedirs(out_path, exist_ok=True)
        print(out_path)
    # 进行数据清洗
    data_clean(res_path,label_path,out_path,xml_path)
    print('Done!')

if __name__ == '__main__':
    main()