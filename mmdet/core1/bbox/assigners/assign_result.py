import torch


class AssignResult(object):

    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels

    def add_gt_(self, gt_labels):
        self_inds = torch.arange(
            1, len(gt_labels) + 1, dtype=torch.long, device=gt_labels.device) # 对应的侯选框索引,拼接到开头
        self.gt_inds = torch.cat([self_inds, self.gt_inds])     # 后面的不用加num_gt吗?因为本来对因的列就是num_gt多个
        self.max_overlaps = torch.cat(
            [self.max_overlaps.new_ones(self.num_gts), self.max_overlaps])  # IOU为1
        if self.labels is not None:
            self.labels = torch.cat([gt_labels, self.labels])
