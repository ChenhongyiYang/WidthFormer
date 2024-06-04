import math

import torch
import torch.nn as nn
import torch.nn.functional as F



class SEReduceBlock(nn.Module):
    def __init__(self, channels, ratio, out_channels, with_bn=False, extra_conv=False):
        super(SEReduceBlock, self).__init__()

        if extra_conv:
            self.conv1 = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv1 = None

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.se_fc = nn.Sequential(
            nn.Linear(channels, channels // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // ratio, channels)
        )
        if not with_bn:
            self.out_conv = nn.Conv2d(channels, out_channels, kernel_size=1, padding=0)
        else:
            self.out_conv = nn.Sequential(
                nn.Conv2d(channels, out_channels, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.conv1 is not None:
            x = self.conv1(x)
        y = self.avg_pool(x)
        y = y.view(y.size(0), -1)
        y = self.se_fc(y)
        y = torch.sigmoid(y).view(y.size(0), -1, 1, 1)
        x = x * y
        x = self.out_conv(x)
        return x

class Scaling(nn.Module):
    def __init__(self, dist_size):
        super(Scaling, self).__init__()
        self.dist_size = dist_size

    def forward(self, x):
        assert len(x.shape) == 4
        return F.interpolate(x, size=self.dist_size, mode='bilinear')

    def get_resolution(self):
        return tuple(self.dist_size)

def gen_dx_bx(xbound, ybound, zbound):
    # bound: [min, max, interval]
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx


def pos2posemb1d(pos, num_pos_feats, temperature=10000):
    # input: 1D range
    # output: 1, len, num_feats
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., None] / dim_t
    posemb = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    return posemb

def pos2posemb2d(pos, num_pos_feats, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_x, pos_y), dim=-1)
    return posemb


def pos2posemb3d(pos, num_pos_feats, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb
