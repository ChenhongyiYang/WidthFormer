# Copyright (c) Phigent Robotics. All rights reserved.

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.cuda import Event
import math
import numpy as np


from mmcv.cnn import build_norm_layer
from mmcv.runner import force_fp32
from mmdet.models import NECKS
from mmdet3d.ops import bev_pool
from mmcv.cnn import build_conv_layer
from mmdet.models.builder import build_backbone


def gen_dx_bx(xbound, ybound, zbound):
    # bound: [min, max, interval]
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])
    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))
    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


@NECKS.register_module()
class LSS_dev(nn.Module):
    def __init__(self, grid_config=None, data_config=None,
                 numC_input=512, numC_Trans=64, downsample=16,
                 accelerate=False, max_drop_point_rate=0.0, use_bev_pool=True,
                 infer_geo_noise=dict(enable=False, x_std=0., y_std=0., z_std=0., cam_x=0., cam_y=0., cam_z=0.),
                 **kwargs):
        super(LSS_dev, self).__init__()
        if grid_config is None:
            grid_config = {
                'xbound': [-51.2, 51.2, 0.8],
                'ybound': [-51.2, 51.2, 0.8],
                'zbound': [-10.0, 10.0, 20.0],
                'dbound': [1.0, 60.0, 1.0],}
        self.grid_config = grid_config
        dx, bx, nx = gen_dx_bx(self.grid_config['xbound'],
                               self.grid_config['ybound'],
                               self.grid_config['zbound'],
                               )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        if data_config is None:
            data_config = {'input_size': (256, 704)}
        self.data_config = data_config
        self.downsample = downsample

        self.infer_geo_noise = infer_geo_noise

        self.frustum = self.create_frustum() # D x feat_H x feat_W x 3
        self.D, _, _, _ = self.frustum.shape
        self.numC_input = numC_input
        self.numC_Trans = numC_Trans
        self.depthnet = nn.Conv2d(self.numC_input, self.D + self.numC_Trans, kernel_size=1, padding=0)
        self.geom_feats = None
        self.accelerate = accelerate
        self.max_drop_point_rate = max_drop_point_rate
        self.use_bev_pool = use_bev_pool

    def get_depth_dist(self, x):
        return x.softmax(dim=1)

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_config['input_size']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_config['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans, offset=None):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        if offset is not None:
            _,D,H,W = offset.shape
            points[:,:,:,:,:,2] = points[:,:,:,:,:,2]+offset.view(B,N,D,H,W)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        if intrins.shape[3]==4: # for KITTI
            shift = intrins[:,:,:3,3]
            points  = points - shift.view(B,N,1,1,1,3,1)
            intrins = intrins[:,:,:3,:3]
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        # points_numpy = points.detach().cpu().numpy()
        return points

    def get_geometry_test(self, rots, trans, intrins, post_rots, post_trans, offset=None, device=None):
        B, N, _ = trans.shape
        assert B == 1

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        if offset is not None:
            _, D, H, W = offset.shape
            points[:, :, :, :, :, 2] = points[:, :, :, :, :, 2] + offset.view(B, N, D, H, W)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        if intrins.shape[3] == 4:  # for KITTI
            shift = intrins[:, :, :3, 3]
            points = points - shift.view(B, N, 1, 1, 1, 3, 1)
            intrins = intrins[:, :, :3, :3]

        intrin_inv = torch.inverse(intrins).view(B, N, 1, 1, 1, 3, 3)

        points = intrin_inv.matmul(points)
        # ----------------------------------------------------------------
        # add noise here
        trans_noise = np.concatenate([
            np.random.normal(0., self.infer_geo_noise['cam_x'], (N, 1)),
            np.random.normal(0., self.infer_geo_noise['cam_y'], (N, 1)),
            np.random.normal(0., self.infer_geo_noise['cam_z'], (N, 1)),
        ], axis=1)
        trans_noise = torch.tensor(trans_noise).to(device=device).view(B, N, 1, 1, 1, 3, 1)
        points += trans_noise

        # Rotation
        x_degrees = np.random.normal(0., self.infer_geo_noise['x_std'], (N,)) * math.pi / 180.
        y_degrees = np.random.normal(0., self.infer_geo_noise['y_std'], (N,)) * math.pi / 180.
        z_degrees = np.random.normal(0., self.infer_geo_noise['z_std'], (N,)) * math.pi / 180.

        noise_rots = []
        for i in range(N):
            x_degree = x_degrees[i]
            y_degree = y_degrees[i]
            z_degree = z_degrees[i]
            x_rot = torch.tensor([[1., 0., 0.],
                                  [0., math.cos(x_degree), -1 * math.sin(x_degree)],
                                  [0., math.sin(x_degree), math.cos(x_degree)]])
            y_rot = torch.tensor([[math.cos(y_degree), 0., math.sin(y_degree)],
                                  [0., 1., 0.],
                                  [-1 * math.sin(y_degree), 0., math.cos(y_degree)]])
            z_rot = torch.tensor([[math.cos(z_degree), -1 * math.sin(z_degree), 0.],
                                  [math.sin(z_degree), math.cos(z_degree), 0.],
                                  [0., 0., 1.]])
            noise_rot = torch.matmul(torch.matmul(z_rot, y_rot), x_rot)
            noise_rots.append(noise_rot)
        noise_rots = torch.stack(noise_rots, dim=0).to(device=device)
        points = noise_rots.view(B, N, 1, 1, 1, 3, 3).matmul(points)
        # ----------------------------------------------------------------
        points = rots.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        return points  # [x, y, z]


    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        nx = self.nx.to(torch.long)
        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long() # get ego voxel coordinate
        geom_feats = geom_feats.view(Nprime, 3) # [B * n_cam * feat_D * feat_H * feat_W, 3]
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1) # [Nprime, 4]

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])  # remove points outside the voxel coordinate
        x = x[kept]
        geom_feats = geom_feats[kept]

        if self.max_drop_point_rate > 0.0 and self.training:
            drop_point_rate = torch.rand(1)*self.max_drop_point_rate
            kept = torch.rand(x.shape[0])>drop_point_rate
            x, geom_feats = x[kept], geom_feats[kept]

        if self.use_bev_pool:
            final = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0],
                                   self.nx[1])
            final = final.transpose(dim0=-2, dim1=-1)
        else:
            # get tensors from the same voxel next to each other
            ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                    + geom_feats[:, 1] * (self.nx[2] * B) \
                    + geom_feats[:, 2] * B \
                    + geom_feats[:, 3]
            sorts = ranks.argsort()
            x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

            # cumsum trick
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks) # cumsumed_voxel_feature, and their voxel coordinate

            # griddify (B x C x Z x X x Y)
            final = torch.zeros((B, C, nx[2], nx[1], nx[0]), device=x.device)  # BEV feature
            final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x
        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1) # Z x (B, C, Z, X, Y) -> (B, ZxC, X, Y)

        return final

    def voxel_pooling_accelerated(self, rots, trans, intrins, post_rots, post_trans, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        nx = self.nx.to(torch.long)
        # flatten x
        x = x.reshape(Nprime, C)
        max = 300
        # flatten indices
        if self.geom_feats is None:
            geom_feats = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
            geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
            geom_feats = geom_feats.view(Nprime, 3)
            batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                             device=x.device, dtype=torch.long) for ix in range(B)])
            geom_feats = torch.cat((geom_feats, batch_ix), 1)

            # filter out points that are outside box
            kept1 = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
                    & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
                    & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
            idx = torch.range(0, x.shape[0] - 1, dtype=torch.long)
            x = x[kept1]
            idx = idx[kept1]
            geom_feats = geom_feats[kept1]

            # get tensors from the same voxel next to each other
            ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                    + geom_feats[:, 1] * (self.nx[2] * B) \
                    + geom_feats[:, 2] * B \
                    + geom_feats[:, 3]
            sorts = ranks.argsort()
            x, geom_feats, ranks, idx = x[sorts], geom_feats[sorts], ranks[sorts], idx[sorts]
            repeat_id = torch.ones(geom_feats.shape[0], device=geom_feats.device, dtype=geom_feats.dtype)
            curr = 0
            repeat_id[0] = 0
            curr_rank = ranks[0]

            for i in range(1, ranks.shape[0]):
                if curr_rank == ranks[i]:
                    curr += 1
                    repeat_id[i] = curr
                else:
                    curr_rank = ranks[i]
                    curr = 0
                    repeat_id[i] = curr
            kept2 = repeat_id < max
            repeat_id, geom_feats, x, idx = repeat_id[kept2], geom_feats[kept2], x[kept2], idx[kept2]

            geom_feats = torch.cat([geom_feats, repeat_id.unsqueeze(-1)], dim=-1)
            self.geom_feats = geom_feats
            self.idx = idx
        else:
            geom_feats = self.geom_feats
            idx = self.idx
            x = x[idx]

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, nx[2], nx[1], nx[0], max), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0], geom_feats[:, 4]] = x
        final = final.sum(-1)
        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def forward(self, input):
        x, rots, trans, intrins, post_rots, post_trans = input
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depthnet(x)
        depth = self.get_depth_dist(x[:, :self.D])  # [B * n_cam, D, feat_H, feat_W]
        img_feat = x[:, self.D:(self.D + self.numC_Trans)] # [B * n_cam, numC_Trans, feat_H, feat_W]

        start_event = Event(enable_timing=True)
        end_event = Event(enable_timing=True)
        start_event.record()

        # Lift
        volume = depth.unsqueeze(1) * img_feat.unsqueeze(2) # [B * n_cam, 1, D, H, W] x [B * n_cam, numC_Trans, 1, H, W] -> [B * n_cam, numCTrans, D, H, W]
        volume = volume.view(B, N, self.numC_Trans, self.D, H, W)
        volume = volume.permute(0, 1, 3, 4, 5, 2) # [B, n_cam, D, H, W, numC_Trans]

        # Splat
        if self.accelerate:
            bev_feat = self.voxel_pooling_accelerated(rots, trans, intrins, post_rots, post_trans, volume)
        else:
            # geom: [d, x, y] coordinate in ego coordinate system
            if not self.training and self.infer_geo_noise['enable']:
                geom = self.get_geometry_test(rots, trans, intrins, post_rots, post_trans, device=img_feat.device)
            else:
                geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
            bev_feat = self.voxel_pooling(geom, volume)  # [B, Z_voxel * numC_Trans, BEV_H, BEV_W]

        end_event.record()
        torch.cuda.synchronize()
        total_time = start_event.elapsed_time(end_event)
        rets = dict()
        rets['bev_feature'] = bev_feat
        rets['time'] = total_time
        return rets


class SELikeModule(nn.Module):
    def __init__(self, in_channel=512, feat_channel=256, intrinsic_channel=33):
        super(SELikeModule, self).__init__()
        self.input_conv = nn.Conv2d(in_channel, feat_channel, kernel_size=1, padding=0)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(intrinsic_channel),
            nn.Linear(intrinsic_channel, feat_channel),
            nn.Sigmoid() )

    def forward(self, x, cam_params):
        x = self.input_conv(x)
        b,c,_,_ = x.shape
        y = self.fc(cam_params).view(b, c, 1, 1)
        return x * y.expand_as(x)


@NECKS.register_module()
class LSSBevDepth_dev(LSS_dev):
    def __init__(self,
                 extra_depth_net,
                 loss_depth_weight,
                 se_config=dict(),
                 dcn_config=dict(bias=True),
                 **kwargs):
        super(LSSBevDepth_dev, self).__init__(**kwargs)
        self.loss_depth_weight = loss_depth_weight
        self.extra_depthnet = build_backbone(extra_depth_net)
        self.featnet = nn.Conv2d(self.numC_input,
                                 self.numC_Trans,
                                 kernel_size=1,
                                 padding=0)
        self.depthnet = nn.Conv2d(extra_depth_net['num_channels'][0],
                                  self.D,
                                  kernel_size=1,
                                  padding=0)
        self.dcn = nn.Sequential(*[build_conv_layer(dict(type='DCNv2',
                                                        deform_groups=1),
                                                   extra_depth_net['num_channels'][0],
                                                   extra_depth_net['num_channels'][0],
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1,
                                                   dilation=1,
                                                   **dcn_config),
                                   nn.BatchNorm2d(extra_depth_net['num_channels'][0])
                                  ])
        self.se = SELikeModule(self.numC_input,
                               feat_channel=extra_depth_net['num_channels'][0],
                               **se_config)

    def forward(self, input):
        x, rots, trans, intrins, post_rots, post_trans, depth_gt = input
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)

        img_feat = self.featnet(x)
        depth_feat = x
        cam_params = torch.cat([intrins.reshape(B*N,-1),
                               post_rots.reshape(B*N,-1),
                               post_trans.reshape(B*N,-1),
                               rots.reshape(B*N,-1),
                               trans.reshape(B*N,-1)],dim=1)
        depth_feat = self.se(depth_feat, cam_params)
        depth_feat = self.extra_depthnet(depth_feat)[0]
        with autocast(False):
            depth_feat = self.dcn(depth_feat.to(dtype=torch.float32))
        depth_digit = self.depthnet(depth_feat)
        depth_prob = self.get_depth_dist(depth_digit)

        # Lift
        volume = depth_prob.unsqueeze(1) * img_feat.unsqueeze(2)
        volume = volume.view(B, N, self.numC_Trans, self.D, H, W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)

        # Splat
        if self.accelerate:
            bev_feat = self.voxel_pooling_accelerated(rots, trans, intrins, post_rots, post_trans, volume)
        else:
            geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
            bev_feat = self.voxel_pooling(geom, volume)

        rets = dict()
        rets['bev_feature'] = bev_feat
        rets['depth'] = depth_digit
        return rets
