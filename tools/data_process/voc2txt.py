import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
from tqdm import tqdm

sets=['train', 'val', 'test']   # 划分的数据集名称
classes = ['holothurian','echinus','scallop','starfish']

abs_path = os.getcwd()  # 路径
def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
def convert_annotation(image_id):
    in_file = open('/home/yzy/datasets/yolo_underwater/box/%s.xml'%( image_id))
    out_file = open('/home/yzy/datasets/yolo_underwater/labels/%s.txt'%(image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        #difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes :
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

if __name__ == '__main__':

    # 修改文件路径
    out_path = '/data1/yzycode/datasets/VOCdevkit/VOC2007/labels_change/'
    # xml path
    xml_path = '/data1/yzycode/datasets/VOCdevkit/VOC2007/Annotations/'
    # 最终输出路径
    out_final = '/data1/yzycode/datasets/VOCdevkit/VOC2007/labels_bn/'
    if os.path.exists(out_final):
        print('output path is exists!')
    else:
        os.makedirs(out_final, exist_ok=True)
        print(out_final)
    label_lists = ['000149.txt', '000249.txt', '000147.txt', '001539.txt', '001627.txt', '001677.txt', '001753.txt',
                   '001754.txt', '001767.txt', '001769.txt',
                   '001770.txt', '001773.txt', '001774.txt', '001775.txt', '001839.txt', '001970.txt', '002042.txt',
                   '002053.txt', '002094.txt', '002145.txt', '002331.txt', '002377.txt',
                   '002436.txt', '002558.txt', '002732.txt', '002746.txt', '002758.txt', '002771.txt', '002839.txt',
                   '003014.txt', '003117.txt', '003169.txt', '003170.txt', '003259.txt',
                   '003415.txt', '003529.txt', '003539.txt', '003550.txt', '003672.txt', '003695.txt', '003758.txt',
                   '003795.txt', '003813.txt', '003819.txt', '003820.txt', '003833.txt', '004153.txt', '004158.txt',
                   '004418.txt', '004722.txt', '004723.txt', '004854.txt', '004879.txt', '004880.txt', '004907.txt',
                   '004936.txt', '004949.txt', '005071.txt', '005075.txt', '005115.txt', '005210.txt', '005536.txt']
    for label_list in tqdm(label_lists):
        xml_file = os.path.join(xml_path,label_list.split('.')[0]+'.xml')
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)







