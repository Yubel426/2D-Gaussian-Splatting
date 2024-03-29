import torch
from lib.utils.optimizer.radam import RAdam


_optimizer_factory = {
    'adam': torch.optim.Adam,
    'radam': RAdam,
    'sgd': torch.optim.SGD
}


def make_optimizer(cfg, net):
    params = []
    lr = cfg.train.lr
    weight_decay = cfg.train.weight_decay
    eps = cfg.train.eps

    # for key, value in net.named_parameters():
    #     if not value.requires_grad:
    #         continue
    #     params += [{"params": [value], "lr": lr, "weight_decay": weight_decay, "eps": eps}]
    #TODO: xyz 学习率调整 from Plenoxels
    params = [
            {'params': [net._xyz], 'lr': cfg.position_lr_init * cfg.spatial_lr_scale, "name": "xyz"},
            {'params': [net._features_dc], 'lr': cfg.feature_lr, "name": "f_dc"},
            {'params': [net._features_rest], 'lr': cfg.feature_lr / 20.0, "name": "f_rest"},
            {'params': [net._opacity], 'lr': cfg.opacity_lr, "name": "opacity"},
            {'params': [net._scaling], 'lr': cfg.scaling_lr, "name": "scaling"},
            {'params': [net._rotation], 'lr': cfg.rotation_lr, "name": "rotation"}
        ]
    if 'adam' in cfg.train.optim:
        optimizer = _optimizer_factory[cfg.train.optim](params, lr, weight_decay=weight_decay, eps=eps)
    else:
        optimizer = _optimizer_factory[cfg.train.optim](params, lr, momentum=0.9)

    return optimizer
