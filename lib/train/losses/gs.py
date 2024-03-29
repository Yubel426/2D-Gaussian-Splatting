import torch
import torch.nn as nn
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
        # with torch.no_grad():
            # # Densification
            # if iteration < cfg.densify_until_iter:
            #     # Keep track of max radii in image-space for pruning
            #     self.net.max_radii2D[visibility_filter] = torch.max(self.net.max_radii2D[visibility_filter], radii[visibility_filter])
            #     self.net.add_densification_stats(viewspace_point_tensor, visibility_filter)
            #     # 对3D gaussians进行克隆或者切分, 并将opacity小于一定阈值的3D gaussians进行删除            
            #     if iteration > cfg.densify_from_iter and iteration % cfg.densification_interval == 0:
            #         size_threshold = 20 if iteration > cfg.opacity_reset_interval else None
            #         self.net.densify_and_prune(cfg.densify_grad_threshold, 0.005, batch['cameras_extent'], size_threshold)
            #     # 对3D gaussians的不透明度进行重置
            #     if iteration % cfg.opacity_reset_interval == 0 or (cfg.white_background and iteration == cfg.densify_from_iter):
            #         self.net.reset_opacity()


        scalar_stats.update({'loss': loss})
        image_stats = {}

        return render_pkg, loss, scalar_stats, image_stats
