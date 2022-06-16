# encoding: utf-8


import torch
from torch import nn
from .batch_norm import get_norm
from fastreid.modeling.ops import meta_conv2d, meta_norm


class Non_local(nn.Module):
    def __init__(self, in_channels, bn_norm, norm_opt, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = in_channels // reduc_ratio

        self.g = meta_conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            meta_conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            meta_norm(bn_norm, self.in_channels, norm_opt=norm_opt),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = meta_conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

        self.phi = meta_conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, x, opt = None):
        """
                :param x: (b, t, h, w)
                :return x: (b, t, h, w)
        """
        batch_size = x.size(0)
        g_x = self.g(x, opt).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x, opt).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x, opt).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y, opt)
        z = W_y + x
        return z
