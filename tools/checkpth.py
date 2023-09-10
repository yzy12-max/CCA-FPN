import torch
import torch.nn as nn

model_path = '/data1/yzycode/mmdetection/work_dirs/underwater/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco/epoch_24.pth'
# model_path = '/data/wjh/weights/gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth'
output_model = torch.load(model_path,map_location='cpu')
pass