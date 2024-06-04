# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torchvision
from PIL import Image
import mmcv
import numpy as np
from pyquaternion import Quaternion

import os.path as osp


from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile

from .loading import LoadMultiViewImageFromFiles_BEVDet


@PIPELINES.register_module()
class LoadMultiViewImagesFromFilesLMDB_BEVDet(LoadMultiViewImageFromFiles_BEVDet):
    def __init__(
        self,
        data_config,
        is_train=False,
        sequential=False,
        aligned=False,
        trans_only=True,
        file_client_args=dict(backend='lmdb')
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = torchvision.transforms.Compose((
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])))
        self.sequential = sequential
        self.aligned = aligned
        self.trans_only = trans_only

        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def get_inputs(self,results, flip=None, scale=None):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        cams = self.choose_cams()
        for cam in cams:
            cam_data = results['img_info'][cam]
            filename = cam_data['data_path']

            _filename = osp.split(filename)[-1]
            img_bytes = self.file_client.get(_filename)
            img = mmcv.imfrombytes(img_bytes, flag='color')
            img = Image.fromarray(img)

            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            intrin = torch.Tensor(cam_data['cam_intrinsic'])
            rot = torch.Tensor(cam_data['sensor2lidar_rotation'])
            tran = torch.Tensor(cam_data['sensor2lidar_translation'])

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(H=img.height,
                                                                               W=img.width,
                                                                               flip=flip,
                                                                               scale=scale)
            img, post_rot2, post_tran2 = self.img_transform(img, post_rot, post_tran,
                                                            resize=resize,
                                                            resize_dims=resize_dims,
                                                            crop=crop,
                                                            flip=flip,
                                                            rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(self.normalize_img(img))

            if self.sequential:
                assert 'adjacent' in results
                if not type(results['adjacent']) is list:
                    filename_adjacent = results['adjacent']['cams'][cam]['data_path']
                    _filename_adjacent = osp.split(filename_adjacent)[-1]
                    img_bytes_adjacent = self.file_client.get(_filename_adjacent)
                    img_adjacent = mmcv.imfrombytes(img_bytes_adjacent, flag='color')
                    img_adjacent = Image.fromarray(img_adjacent)


                    img_adjacent = self.img_transform_core(img_adjacent,
                                                           resize_dims=resize_dims,
                                                           crop=crop,
                                                           flip=flip,
                                                           rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))
                else:
                    for id in range(len(results['adjacent'])):
                        filename_adjacent = results['adjacent'][id]['cams'][cam]['data_path']
                        _filename_adjacent = osp.split(filename_adjacent)[-1]
                        img_bytes_adjacent = self.file_client.get(_filename_adjacent)
                        img_adjacent = mmcv.imfrombytes(img_bytes_adjacent, flag='color')
                        img_adjacent = Image.fromarray(img_adjacent)

                        img_adjacent = self.img_transform_core(img_adjacent,
                                                               resize_dims=resize_dims,
                                                               crop=crop,
                                                               flip=flip,
                                                               rotate=rotate)
                        imgs.append(self.normalize_img(img_adjacent))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        if self.sequential:
            if self.trans_only:
                if not type(results['adjacent']) is list:
                    rots.extend(rots)
                    post_trans.extend(post_trans)
                    post_rots.extend(post_rots)
                    intrins.extend(intrins)
                    if self.aligned:
                        posi_curr = np.array(results['curr']['ego2global_translation'], dtype=np.float32)
                        posi_adj = np.array(results['adjacent']['ego2global_translation'], dtype=np.float32)
                        shift_global = posi_adj - posi_curr

                        l2e_r = results['curr']['lidar2ego_rotation']
                        e2g_r = results['curr']['ego2global_rotation']
                        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
                        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

                        # shift_global = np.array([*shift_global[:2], 0.0])
                        shift_lidar = shift_global @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                            l2e_r_mat).T
                        trans.extend([tran + shift_lidar for tran in trans])
                    else:
                        trans.extend(trans)
                else:
                    assert False
            else:
                if not type(results['adjacent']) is list:
                    post_trans.extend(post_trans)
                    post_rots.extend(post_rots)
                    intrins.extend(intrins)
                    if self.aligned:
                        egocurr2global = np.eye(4, dtype=np.float32)
                        egocurr2global[:3,:3] = Quaternion(results['curr']['ego2global_rotation']).rotation_matrix
                        egocurr2global[:3,3] = results['curr']['ego2global_translation']

                        egoadj2global = np.eye(4, dtype=np.float32)
                        egoadj2global[:3,:3] = Quaternion(results['adjacent']['ego2global_rotation']).rotation_matrix
                        egoadj2global[:3,3] = results['adjacent']['ego2global_translation']

                        lidar2ego = np.eye(4, dtype=np.float32)
                        lidar2ego[:3, :3] = Quaternion(results['curr']['lidar2ego_rotation']).rotation_matrix
                        lidar2ego[:3, 3] = results['curr']['lidar2ego_translation']

                        lidaradj2lidarcurr = np.linalg.inv(lidar2ego) @ np.linalg.inv(egocurr2global) \
                                             @ egoadj2global @ lidar2ego
                        trans_new = []
                        rots_new =[]
                        for tran,rot in zip(trans, rots):
                            mat = np.eye(4, dtype=np.float32)
                            mat[:3,:3] = rot
                            mat[:3,3] = tran
                            mat = lidaradj2lidarcurr @ mat
                            rots_new.append(torch.from_numpy(mat[:3,:3]))
                            trans_new.append(torch.from_numpy(mat[:3,3]))
                        rots.extend(rots_new)
                        trans.extend(trans_new)
                    else:
                        rots.extend(rots)
                        trans.extend(trans)
                else:
                    assert False
        imgs, rots, trans, intrins, post_rots, post_trans = (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                                                             torch.stack(intrins), torch.stack(post_rots),
                                                             torch.stack(post_trans))
        return imgs, rots, trans, intrins, post_rots, post_trans

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        return results