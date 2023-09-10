import os
import os.path as osp
import xml.etree.ElementTree as ET

# import mmcv
import json  # 换成读取
# from mmdet.core import underwater_classes

from glob import glob
from tqdm import tqdm
from PIL import Image

# label_ids = {name: i + 1 for i, name in enumerate(underwater_classes())}

# label_ids = {
#     "holothurian": 1,
#     "echinus": 2,
#     "scallop": 3,
#     "starfish": 4
# }
label_ids = {"cube": 1,"ball": 2,"cylinder": 3,"human body": 4,"tyre":5,"circle cage":6,
             "square cage":7,"metal bucket":8,"plane":9,"rov":10
}


def get_segmentation(points):
    return [points[0], points[1], points[2] + points[0], points[1],
            points[2] + points[0], points[3] + points[1], points[0], points[3] + points[1]]


def parse_xml(xml_path, img_id, anno_id):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotation = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name == 'waterweeds':
            continue
        category_id = label_ids[name]
        bnd_box = obj.find('bndbox')
        xmin = int(bnd_box.find('xmin').text)
        ymin = int(bnd_box.find('ymin').text)
        xmax = int(bnd_box.find('xmax').text)
        ymax = int(bnd_box.find('ymax').text)
        w = xmax - xmin + 1
        h = ymax - ymin + 1
        area = w * h
        segmentation = get_segmentation([xmin, ymin, w, h])
        annotation.append({
            "segmentation": segmentation,
            "area": area,
            "iscrowd": 0,
            "image_id": img_id,
            "bbox": [xmin, ymin, w, h],
            "category_id": category_id,
            "id": anno_id,
            "ignore": 0})
        anno_id += 1
    return annotation, anno_id


def cvt_annotations(img_path, xml_path, out_file):
    images = []
    annotations = []

    # xml_paths = glob(xml_path + '/*.xml')
    img_id = 1
    anno_id = 1
    for img_path in tqdm(glob(img_path + '/*.bmp')):
    # for img_path in tqdm(glob(img_path + '/*.jpg')):
        w, h = Image.open(img_path).size
        img_name = osp.basename(img_path)
        img = {"file_name": img_name, "height": int(h), "width": int(w), "id": img_id}
        images.append(img)

        xml_file_name = img_name.split('.')[0] + '.xml'
        xml_file_path = osp.join(xml_path, xml_file_name)
        annos, anno_id = parse_xml(xml_file_path, img_id, anno_id)
        annotations.extend(annos)
        img_id += 1

    categories = []
    for k, v in label_ids.items():
        categories.append({"name": k, "id": v})
    final_result = {"images": images, "annotations": annotations, "categories": categories}
    # mmcv.dump(final_result, out_file)

    with open(out_file, 'w') as f:  # 用json处理，不用mmcv
        json.dump(final_result, f, indent=1)
    # mmcv.dump(final_result, out_file)
    return annotations
    # return annotations


def main():
    print('all train data!')
    xml_path = '/data/wjh/datasets/forUser_A/voice-train/box'
    img_path = '/data/wjh/datasets/forUser_A/voice-train/image'
    print('processing {} ...'.format("xml format annotations"))
    os.makedirs('/data/wjh/datasets/forUser_A/voice-train/annotations/', exist_ok=True)
    cvt_annotations(img_path, xml_path, '/data/wjh/datasets/forUser_A/voice-train/annotations/trainall.json')
    print('Done!')

    # print("train data!")
    # xml_path = '/data/wjh/datasets/forUser_A/voice-train/train-box'
    # img_path = '/data/wjh/datasets/forUser_A/voice-train/train-image'
    # print('processing {} ...'.format("xml format annotations"))
    # os.makedirs('/data/wjh/datasets/forUser_A/voice-train/annotations/', exist_ok=True)
    # cvt_annotations(img_path, xml_path, '/data/wjh/datasets/forUser_A/voice-train/annotations/train.json')
    # print('Done!')
    #
    # print("val data!")
    # xml_path = '/data/wjh/datasets/forUser_A/voice-train/val-box'
    # img_path = '/data/wjh/datasets/forUser_A/voice-train/val-image'
    # print('processing {} ...'.format("xml format annotations"))
    # os.makedirs('/data/wjh/datasets/forUser_A/voice-train/annotations/', exist_ok=True)
    # cvt_annotations(img_path, xml_path, '/data/wjh/datasets/forUser_A/voice-train/annotations/val.json')
    # print('Done!')


if __name__ == '__main__':
    main()
