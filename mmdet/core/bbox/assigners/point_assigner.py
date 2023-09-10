# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import BBOX_ASSIGNERS
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class PointAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each point.

    Each proposals will be assigned with `0`, or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    """

    def __init__(self, scale=4, pos_num=3):
        self.scale = scale
        self.pos_num = pos_num

    def assign(self, points, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to points.

        This method assign a gt bbox to every points set, each points set
        will be assigned with  the background_label (-1), or a label number.
        -1 is background, and semi-positive number is the index (0-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every points to the background_label (-1)
        2. A point is assigned to some gt bbox if
            (i) the point is within the k closest points to the gt bbox
            (ii) the distance between this point and the gt is smaller than
                other gt bboxes

        Args:
            points (Tensor): points to be assigned, shape(n, 3) while last
                dimension stands for (x, y, stride).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
                NOTE: currently unused.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_points = points.shape[0]
        num_gts = gt_bboxes.shape[0]

        if num_gts == 0 or num_points == 0: #判断是否有检测目标
            # If no truth assign everything to the background
            assigned_gt_inds = points.new_full((num_points, ),
                                               0,
                                               dtype=torch.long)
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = points.new_full((num_points, ),
                                                  -1,
                                                  dtype=torch.long)
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        points_xy = points[:, :2]
        points_stride = points[:, 2]
        points_lvl = torch.log2(
            points_stride).int()  # [3...,4...,5...,6...,7...]  #取对数的整数，为了匹配下面的计算
        lvl_min, lvl_max = points_lvl.min(), points_lvl.max()

        # assign gt box
        gt_bboxes_xy = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2    #gt_bboxes的中心点
        gt_bboxes_wh = (gt_bboxes[:, 2:] - gt_bboxes[:, :2]).clamp(min=1e-6)
        scale = self.scale    #不是和FCOS一样可学习的学习尺度因子，而是固定值为4
        gt_bboxes_lvl = ((torch.log2(gt_bboxes_wh[:, 0] / scale) +          #论文里的公式，为了计算gt bbox宽高落在哪个尺度
                          torch.log2(gt_bboxes_wh[:, 1] / scale)) / 2).int()  #取整数
        gt_bboxes_lvl = torch.clamp(gt_bboxes_lvl, min=lvl_min, max=lvl_max)

        # stores the assigned gt index of each point
        assigned_gt_inds = points.new_zeros((num_points, ), dtype=torch.long)
        # stores the assigned gt dist (to this point) of each point 存分配点的距离
        assigned_gt_dist = points.new_full((num_points, ), float('inf'))  #填满float('inf')：10000
        points_range = torch.arange(points.shape[0]) #生成点（不是样本点）的索引，相当于给了样本点一个下标

        for idx in range(num_gts):
            gt_lvl = gt_bboxes_lvl[idx]   #获取通过公式编码后的目标值
            # get the index of points in this level
            lvl_idx = gt_lvl == points_lvl   #a=b==c，复合表达式先B值赋给A与c比较相等则返TRUE，否则返F
            points_index = points_range[lvl_idx]  #找出相匹配层的点索引
            # get the points in this level 获取目标值所在特征层的所有样本点
            lvl_points = points_xy[lvl_idx, :]
            # get the center point of gt  获取目标中心
            gt_point = gt_bboxes_xy[[idx], :]    #
            # get width and height of gt
            gt_wh = gt_bboxes_wh[[idx], :]
            # compute the distance between gt center and
            #   all points in this level 计算这层所有样本点和GT中心的距离，因为是原图坐标，后归一化了
            points_gt_dist = ((lvl_points - gt_point) / gt_wh).norm(dim=1)
            # find the nearest k points to gt center in this level 也是找最近
            min_dist, min_dist_index = torch.topk(
                points_gt_dist, self.pos_num, largest=False)  #self.pos_num=1 默认是3
            # the index of nearest k points to gt center in this level
            min_dist_points_index = points_index[min_dist_index]
            # The less_than_recorded_index stores the index
            #   of min_dist that is less then the assigned_gt_dist. Where
            #   assigned_gt_dist stores the dist from previous assigned gt
            #   (if exist) to each point. 和预设的INF比较
            less_than_recorded_index = min_dist < assigned_gt_dist[
                min_dist_points_index]
            # The min_dist_points_index stores the index of points satisfy:
            #   (1) it is k nearest to current gt center in this level.
            #   (2) it is closer to current gt center than other gt center.
            min_dist_points_index = min_dist_points_index[  #如果有多个，那就距离哪个gt bbox中心最近就负责预测谁
                less_than_recorded_index]
            # assign the result
            assigned_gt_inds[min_dist_points_index] = idx + 1     #=第几个样本数，相当于第几个样本的索引，
            assigned_gt_dist[min_dist_points_index] = min_dist[   #样本距离=距离最小值
                less_than_recorded_index]
            # 但是如果topk取得比较大，可能会出现fcos里面描述的模糊样本，对于这类样本的处理办法就是其距离哪个gt bbox中心最近就负责预测谁
        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_points, ), -1)  #初始化为-1
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()  #取出正样本的索引
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[ #将LABEL按索引给值，可以看出目标都是排好序了的
                    assigned_gt_inds[pos_inds] - 1] #0是表示第一个目标索引
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)
