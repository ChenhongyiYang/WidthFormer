# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torchvision
from PIL import Image
import mmcv
import time
import numpy as np
from pyquaternion import Quaternion

import os.path as osp


from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile

from mmdet3d.datasets.pipelines.loading import LoadMultiViewImageFromFiles_BEVDet, LoadPointsFromFile

@PIPELINES.register_module()
class PointToMultiViewDepthDev(object):
    def __init__(self, grid_config, downsample=16):
        self.downsample = downsample
        self.grid_config=grid_config

    def points2depthmap(self, points, height, width, canvas=None):
        height, width = height//self.downsample, width//self.downsample
        depth_map = torch.zeros((height,width), dtype=torch.float32)
        coor = torch.round(points[:,:2]/self.downsample)
        depth = points[:,2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) \
               & (coor[:, 1] >= 0) & (coor[:, 1] < height) \
                & (depth < self.grid_config['dbound'][1]) \
                & (depth >= self.grid_config['dbound'][0])
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks+depth/100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:,1],coor[:,0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        imgs, rots, trans, intrins, post_rots, post_trans = results['img_inputs']
        depth_map_list = []
        for cid in range(rots.shape[0]):
            combine = rots[cid].matmul(torch.inverse(intrins[cid]))
            combine_inv = torch.inverse(combine)
            points_img = (points_lidar.tensor[:,:3] - trans[cid:cid+1,:]).matmul(combine_inv.T)
            points_img = torch.cat([points_img[:,:2]/points_img[:,2:3],
                                   points_img[:,2:3]], 1)
            points_img = points_img.matmul(post_rots[cid].T)+post_trans[cid:cid+1,:]
            depth_map = self.points2depthmap(points_img, imgs.shape[2], imgs.shape[3])
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)
        results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, depth_map)
        return results

