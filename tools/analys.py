
#可视化卷集核参数
import torch
import numpy as np
import matplotlib.pyplot as plt
# from mmcv import Config
# from mmdet.models import build_detector


"""
pre_model_dict:'meta','state_dict','optimizer'
"""

def filter_show(filters, nx=16, margin=3, scale=10):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(FN / nx))

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()


path = '../work_dirs/gfl_r50_fpn_1x_coco/epoch_12.pth'
pre_model_dict = torch.load(path, map_location='cpu')
# cfg = Config.fromfile('../configs/pascal_voc/gfl_r50_fpn_1x_coco.py')
# detector = build_detector(cfg.model,train_cfg=cfg.get('train_cfg'),
#         test_cfg=cfg.get('test_cfg'))
# net = detector
#net.load_state_dict(pre_model_dict['state_dict'])
#params = list(net.named_parameters())
#params = net.named_parameters()
save = 0
for key, val in pre_model_dict['state_dict'].items():
    if key == 'backbone.layer2.0.conv2.weight':
        save = val
save = save.numpy()
filter_show(save)
pass
