
import json
import torch
import torch.nn as nn

# model_path = '/data/wjh/weights/detectors_htc_r50_1x_coco-329b1453.pth'
# model_path = '/data/wjh/weights/gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth'
# output_model = torch.load(model_path,map_location='cpu')

json_path = '/data/wjh/datasets/forUser_A/voice-train/annotations/trainall.json'
f = open(json_path,encoding='utf-8')
data = json.load(f)
pass