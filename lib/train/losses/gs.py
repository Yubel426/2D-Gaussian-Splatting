import torch
import torch.nn as nn
import os
import torchvision
from lib.networks.gs.render import render
from lib.config import cfg


class NetworkWrapper(nn.Module):
    def __init__(self, net, train_loader):
        super(NetworkWrapper, self).__init__()
        self.net = net
        # self.renderer = render.render()
        self.img2mse = lambda x, y : torch.mean((x - y) ** 2)
        self.acc_crit = torch.nn.functional.smooth_l1_loss
        self.color_crit = nn.MSELoss(reduction='mean')
        self.mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
        bg_color = [1, 1, 1] if cfg.white_background else [0, 0, 0]
        self.bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        #TODO: add random background
        # self.bg = torch.rand((3), device="cuda") if opt.random_background else background



    def forward(self, batch):
        render_pkg = render.render(batch, self.net, cfg.pipe, self.bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        scalar_stats = {}
        loss = 0
        iteration = batch['step']
        img_loss = self.img2mse(image, batch['original_image'])
        scalar_stats.update({'img_loss': img_loss})
        loss += img_loss
        if iteration % 1000 == 0:
            save_path = os.path.join(cfg.result_dir, 'vis/res_{}.jpg'.format(iteration))
            torchvision.utils.save_image(image, save_path)



        scalar_stats.update({'loss': loss})
        image_stats = {}

        return render_pkg, loss, scalar_stats, image_stats
