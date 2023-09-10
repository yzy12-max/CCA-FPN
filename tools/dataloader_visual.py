# -*- coding: utf-8 -*-
# @Time    : 2020/10/2 下午9:55
# @Author  : zxq
# @File    : run_dataloader.py
# @Software: PyCharm
import mmcv
import numpy as np
from mmcv import Config

#from mmcls.datasets import build_dataloader, build_dataset
from mmdet.datasets import build_dataloader, build_dataset


def run_datatloader(cfg):
    """
    可视化数据增强后的效果，同时也可以确认训练样本是否正确
    Args:
        cfg: 配置
    Returns:
    """
    # Build dataset
    dataset = build_dataset(cfg.data.train)  #
    # prepare data loaders

    data_loader = build_dataloader(
        dataset,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        round_up=True,
        seed=cfg.seed)

    for i, data_batch in enumerate(data_loader):
        img_batch = data_batch['img']
        gt_label = data_batch['gt_label']
        for batch_i in range(len(img_batch)):
            img = img_batch[batch_i]
            gt = gt_label[batch_i]

            mean_value = np.array(cfg.img_norm_cfg['mean'])
            std_value = np.array(cfg.img_norm_cfg['std'])
            img_hwc = np.transpose(img.numpy(), [1, 2, 0])
            img_numpy_float = mmcv.imdenormalize(img_hwc, mean_value, std_value)
            img_numpy_uint8 = np.array(img_numpy_float, np.uint8)
            print(gt.numpy())
            mmcv.imshow(img_numpy_uint8, 'img', 0)


if __name__ == '__main__':
    cfg = Config.fromfile('../../configs/imagenet/ciga_call_cfg.py')
    run_datatloader(cfg)