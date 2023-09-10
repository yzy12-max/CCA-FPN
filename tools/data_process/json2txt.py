# 原文链接：https://blog.csdn.net/qq_42597373/article/details/118494997

import json
import os
import argparse


# wangzhifneg write in 2021-6-30
def coco2vistxt(json_path, out_folder,imageid2name,cateid2name):
    labels = json.load(open(json_path, 'r', encoding='utf-8'))
    zero_area = 0
    for i in range(len(labels)):
        print(len(labels))
    # print(labels[i]['image_id'])
    #     print(labels[i]['bbox'])
        image_name = imageid2name[labels[i]['image_id']][:-4]
        file_name = image_name + '.txt'
        # file_name = labels[i]['image_id'] + '.txt'
        with open(os.path.join(out_folder, file_name), 'a+', encoding='utf-8') as f:
            # l = labels[i]['bbox']
            # s = [round(i) for i in l]
            category_id = labels[i]['category_id']
            catename = cateid2name[category_id]
            box = labels[i]['bbox']
            xmin, ymin, w, h = box
            if w == 0 or h == 0:
                zero_area += 1
                continue
            xmax = xmin + w
            ymax = ymin + h
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            score = labels[i]['score']
            line = str(catename) + ',' + str(xmin) + ',' + str(
                ymin) + ',' + str(xmax) + ',' + str(ymax) +  ',' + str(score) +'\n'
            # line =str(s)[1:-1].replace(' ','')+ ',' + str(labels[i]['score'])[:6] + ',' + str(labels[i]['category_id']) + ',' + str('-1') + ',' + str('-1')
            f.write(line)
                            # swa_cascade_rcnn_r50_rfp_sac_iou_alldata-v3_e15
if __name__ == '__main__':  # /data/wjh/mmdetection-2.19.0/work_dirs/cascade_rcnn_r50_rfp_sac_iou_e15_alldata_v3/submit3.bbox.json
    # json_path = '/data/wjh/mmdetection-2.19.0/work_dirs/cascade_rcnn_r50_rfp_sac_iou_e15_alldata_v3/submit0.bbox.json'
    # out_folder = '/data/wjh/mmdetection-2.19.0/work_dirs/cascade_rcnn_r50_rfp_sac_iou_e15_alldata_v3/sub_optics/'
    json_path = '/data/wjh/mmdetection-2.19.0/work_dirs/swa_cascade_rcnn_r50_rfp_sac_iou_alldata-v3_e15/submit4.bbox.json'
    out_folder = '/data/wjh/mmdetection-2.19.0/work_dirs/swa_cascade_rcnn_r50_rfp_sac_iou_alldata-v3_e15/sub_optics/'
    # json_path = '/data/wjh/mmdetection-2.19.0/work_dirs/underwater/swa_gfl_r2n101_fpn_dcn_ms_2x/submit4_voice.bbox.json'
    # out_folder = '/data/wjh/mmdetection-2.19.0/work_dirs/underwater/swa_gfl_r2n101_fpn_dcn_ms_2x/sub_acoustics/'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    # raw_anno_file = '/data/wjh/datasets/annotations/testB_acsou.json'  # 原来生成的testA.json
    raw_anno_file = '/data/wjh/datasets/annotations/testB.json'
    with open(raw_anno_file, 'r') as f:
        annos = json.load(f)
    imageid2name = {}
    for image in annos['images']:
        imageid2name[image['id']] = image['file_name']
    # catename2nid = {
    #     "cube": 1, "ball": 2, "cylinder": 3, "human body": 4, "tyre": 5, "circle cage": 6,
    #     "square cage": 7, "metal bucket": 8, "plane": 9, "rov": 10
    # }
    catename2nid = {
            "holothurian": 1, "echinus": 2,"scallop": 3,"starfish": 4
    }
    cateid2name = {}
    for catename, cateid in catename2nid.items():  # 遍历字典
        cateid2name[cateid] = catename
    coco2vistxt(json_path=json_path, out_folder=out_folder,imageid2name=imageid2name,cateid2name=cateid2name)