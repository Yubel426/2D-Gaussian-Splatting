from torch.utils.data.dataloader import default_collate
import torch
import numpy as np
from lib.config import cfg

_collators = {"3dgs"}

def my_collate_fn(batch):
    # 获取 batch 中每个元素的属性和tensor，然后构建新的 Camera 实例
    colmap_ids = [item.colmap_id for item in batch]
    Rs = [item.R for item in batch]
    Ts = [item.T for item in batch]
    FoVxs = [item.FoVx for item in batch]
    FoVys = [item.FoVy for item in batch]
    images = [item.original_image for item in batch]
    gt_alpha_masks = [item.gt_alpha_mask for item in batch]
    image_names = [item.image_name for item in batch]
    uids = [item.uid for item in batch]
    transes = [item.trans for item in batch]
    scales = [item.scale for item in batch]
    data_devices = [item.data_device for item in batch]

    return Camera(colmap_ids, Rs, Ts, FoVxs, FoVys, images, gt_alpha_masks, image_names, uids, transes, scales, data_devices)
def make_collator(cfg, is_train):
    collator = cfg.train.collator if is_train else cfg.test.collator
    if collator in _collators:
        return _collators[collator]
    else:
        return default_collate
