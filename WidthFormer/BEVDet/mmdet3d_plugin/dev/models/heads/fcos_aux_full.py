# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from abc import abstractmethod
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from mmcv.runner import force_fp32
from torch import nn as nn
from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmdet3d.models.dense_heads.base_mono3d_dense_head import BaseMono3DDenseHead

from mmcv.runner import get_dist_info


INF = 1e8
EPS = 1e-5


@HEADS.register_module()
class FCOSAuxFullHead(BaseMono3DDenseHead):
    """Anchor-free head for monocular 3D object detection.
    """  # noqa: W605

    _version = 1

    def __init__(
            self,
            num_classes,
            in_channels,
            feat_channels=256,
            n_shared_convs=2,
            stacked_convs=4,
            conv_bias='auto',
            background_label=None,
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_delta=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
            loss_depth=dict(type='CrossEntropyLoss', loss_weight=1.0),
            loss_height=dict(type='CrossEntropyLoss', loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
            loss_velo=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
            loss_rot=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
            loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            cls_branch=(256,),
            reg_branch=(
                    (256,),  # delta_x
                    (256,),  # depth
                    (256,),  # height
                    (256,),  # box
            ),
            centerness_branch=(64,),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=None,
            train_cfg=None,
            test_cfg=None,
            init_cfg=None,
            data_config=None):

        super(FCOSAuxFullHead, self).__init__(init_cfg=init_cfg)

        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.n_shared_convs = n_shared_convs
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias

        self.loss_cls = build_loss(loss_cls)
        self.loss_delta = build_loss(loss_delta)
        self.loss_depth = build_loss(loss_depth)
        self.loss_height = build_loss(loss_height)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_velo = build_loss(loss_velo)
        self.loss_rot = build_loss(loss_rot)
        self.loss_centerness = build_loss(loss_centerness)

        input_size = train_cfg.get('input_size', None)
        down_stride = train_cfg.get('down_stride', None)
        depth_bounds = train_cfg.get('depth_bounds', None)  # [1.0, 60.0, 1.0]
        depth_bins = int((depth_bounds[1] - depth_bounds[0]) / depth_bounds[2])

        self.depth_low_bound = depth_bounds[0]
        self.depth_high_bound = depth_bounds[1]
        self.depth_gap = depth_bounds[2]

        reg_out_channels = (1, depth_bins+2, input_size[0]//down_stride, 3, 2, 2)  # delta_x, depth, height, box, rot, volocity
        self.reg_out_channels = list(reg_out_channels)
        assert len(reg_out_channels) == 6
        assert len(reg_branch) == len(reg_out_channels)

        self.bbox_code_size = 10
        self.depth_bins = depth_bins
        self.height_bins = reg_out_channels[2]

        self.cls_branch = cls_branch
        self.reg_branch = reg_branch
        self.centerness_branch = centerness_branch
        self.data_config = data_config

        self.out_channels = []
        for reg_branch_channels in reg_branch:
            if len(reg_branch_channels) > 0:
                self.out_channels.append(reg_branch_channels[-1])
            else:
                self.out_channels.append(-1)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.background_label = (
            num_classes if background_label is None else background_label)
        # background_label should be either 0 or num_classes
        assert (self.background_label == 0
                or self.background_label == num_classes)

        self._init_layers()
        if init_cfg is None:
            self.init_cfg = dict(
                type='Normal',
                layer='Conv1d',
                std=0.01,
                override=dict(
                    type='Normal', name='conv_cls', std=0.01, bias_prob=0.01))

    def _init_layers(self):
        """Initialize layers of the head."""
        self.shared_convs = self._init_convs(self.n_shared_convs, self.in_channels, self.feat_channels)
        self.cls_convs = self._init_convs(self.stacked_convs, self.feat_channels, self.feat_channels)
        self.reg_convs = self._init_convs(self.stacked_convs, self.feat_channels, self.feat_channels)
        self._init_predictor()

    def _init_convs(self, stacked_convs, in_channels, feat_channels):
        if stacked_convs == 0:
            return None

        convs = nn.ModuleList()
        for i in range(stacked_convs):
            chn = in_channels if i == 0 else feat_channels
            convs.append(
                ConvModule(
                    chn,
                    feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))
        return convs

    def _init_branch(self, conv_channels=(64), conv_strides=(1)):
        """Initialize conv layers as a prediction branch."""
        conv_before_pred = nn.ModuleList()
        if isinstance(conv_channels, int):
            conv_channels = [self.feat_channels] + [conv_channels]
            conv_strides = [conv_strides]
        else:
            conv_channels = [self.feat_channels] + list(conv_channels)
            conv_strides = list(conv_strides)
        for i in range(len(conv_strides)):
            conv_before_pred.append(
                ConvModule(
                    conv_channels[i],
                    conv_channels[i + 1],
                    3,
                    stride=conv_strides[i],
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))
        return conv_before_pred

    def _init_predictor(self):
        """Initialize predictor layers of the head."""
        self.conv_cls_prev = self._init_branch(conv_channels=self.cls_branch, conv_strides=(1,) * len(self.cls_branch))
        self.conv_cls = nn.Conv1d(self.cls_branch[-1], self.cls_out_channels, 1)

        self.conv_reg_prevs = nn.ModuleList()
        self.conv_regs = nn.ModuleList()
        for i in range(len(self.reg_out_channels)):
            reg_dim = self.reg_out_channels[i]
            reg_branch_channels = self.reg_branch[i]
            out_channel = self.out_channels[i]
            if len(reg_branch_channels) > 0:
                self.conv_reg_prevs.append(
                    self._init_branch(
                        conv_channels=reg_branch_channels,
                        conv_strides=(1,) * len(reg_branch_channels)))
                self.conv_regs.append(nn.Conv1d(out_channel, reg_dim, 1))
            else:
                self.conv_reg_prevs.append(None)
                self.conv_regs.append(nn.Conv1d(self.feat_channels, reg_dim, 1))

        self.conv_centerness_prev = self._init_branch(
            conv_channels=self.centerness_branch,
            conv_strides=(1,) * len(self.centerness_branch))
        self.conv_centerness = nn.Conv1d(self.centerness_branch[-1], 1, 1)

    def forward(self, feats):
        """Forward features from the upstream network.
        """
        return multi_apply(self.forward_single, feats)

    def forward_single(self, x):
        """Forward features of a single scale levle.
        """
        x = x.permute(0, 2, 1)

        if self.shared_convs is not None:
            for conv_layer in self.shared_convs:
                x = conv_layer(x)

        cls_feat = x
        reg_feat = x

        if self.cls_convs is not None:
            for cls_layer in self.cls_convs:
                cls_feat = cls_layer(cls_feat)
        # clone the cls_feat for reusing the feature map afterwards
        clone_cls_feat = cls_feat.clone()
        for conv_cls_prev_layer in self.conv_cls_prev:
            clone_cls_feat = conv_cls_prev_layer(clone_cls_feat)

        cls_score = self.conv_cls(clone_cls_feat)

        if self.reg_convs is not None:
            for reg_layer in self.reg_convs:
                reg_feat = reg_layer(reg_feat)
        bbox_pred = []
        for i in range(len(self.reg_out_channels)):
            # clone the reg_feat for reusing the feature map afterwards
            clone_reg_feat = reg_feat.clone()
            if len(self.reg_branch[i]) > 0:
                for conv_reg_prev_layer in self.conv_reg_prevs[i]:
                    clone_reg_feat = conv_reg_prev_layer(clone_reg_feat)
            bbox_pred.append(self.conv_regs[i](clone_reg_feat))
        bbox_pred = torch.cat(bbox_pred, dim=1)

        clone_reg_feat = reg_feat.clone()
        for conv_centerness_prev_layer in self.conv_centerness_prev:
            clone_reg_feat = conv_centerness_prev_layer(clone_reg_feat)
        centerness = self.conv_centerness(clone_reg_feat)
        return cls_score, bbox_pred, centerness

    def _split_bbox_preds(self, bbox_preds):
        index_edges = [sum(self.reg_out_channels[:i]) for i in range(len(self.reg_out_channels)+1)]
        delta_preds = bbox_preds[:, index_edges[0]:index_edges[1]]
        depth_preds = bbox_preds[:, index_edges[1]:index_edges[2]]
        height_preds = bbox_preds[:, index_edges[2]:index_edges[3]]
        bbox3d_preds = bbox_preds[:, index_edges[3]:index_edges[4]]
        rots_preds = bbox_preds[:, index_edges[4]:index_edges[5]]
        velo_preds = bbox_preds[:, index_edges[5]:index_edges[6]]
        return delta_preds, depth_preds, height_preds, bbox3d_preds, rots_preds, velo_preds

    @force_fp32(apply_to=('preds'))
    def loss(self,
             gt_bboxes_3d,
             gt_labels_3d,
             preds,
             img_metas,
             img_inputs,
             ):
        """Compute loss of the head.
        """

        labels_3d, bbox_targets_3d, centerness_targets = self.get_targets(gt_bboxes_3d, gt_labels_3d, preds, img_inputs)

        num_batch, num_cam, num_width = labels_3d.shape[:]
        num_point = num_batch * num_cam * num_width
        labels_3d = labels_3d.view(-1)
        centerness_targets = centerness_targets.view(-1)
        bbox_targets_3d = bbox_targets_3d.view(num_point, -1)
        delta_targets = bbox_targets_3d[:, 0:1]
        depth_targets = bbox_targets_3d[:, 1:2].to(dtype=torch.long)
        height_targets = bbox_targets_3d[:,2:3].to(dtype=torch.long)
        bbox3d_targets = bbox_targets_3d[: ,3:6]
        rot_targets = bbox_targets_3d[:, 6:8]
        velo_targets = bbox_targets_3d[:, 8:]

        cls_scores, bbox_preds, centerness = preds[:]
        cls_scores = torch.stack(cls_scores).permute(0, 1, 3, 2).contiguous().view(num_point, -1)
        bbox_preds = torch.stack(bbox_preds).permute(0, 1, 3, 2).contiguous().view(num_point, -1)
        delta_preds, depth_preds, height_preds, bbox3d_preds, rot_preds, velo_preds = self._split_bbox_preds(bbox_preds)
        centerness = torch.stack(centerness).view(-1)

        bg_class_ind = self.num_classes
        pos_inds = ((labels_3d >= 0) & (labels_3d < bg_class_ind)).nonzero().reshape(-1)
        num_pos = len(pos_inds)

        loss_cls = self.loss_cls(cls_scores, labels_3d, avg_factor=num_pos + num_cam)

        pos_delta_preds = delta_preds[pos_inds]
        pos_depth_preds = depth_preds[pos_inds]
        pos_height_preds = height_preds[pos_inds]
        pos_bbox3d_preds = bbox3d_preds[pos_inds]
        pos_rot_preds = rot_preds[pos_inds]
        pos_velo_preds = velo_preds[pos_inds]

        pos_centerness = centerness[pos_inds]

        if num_pos > 0:
            delta_targets = delta_targets[pos_inds]
            depth_targets = depth_targets[pos_inds]
            height_targets = height_targets[pos_inds]
            bbox3d_targets = bbox3d_targets[pos_inds]
            rot_targets = rot_targets[pos_inds]
            velo_targets = velo_targets[pos_inds]
            pos_centerness_targets = centerness_targets[pos_inds]

            equal_weights = pos_centerness_targets.new_ones(
                pos_centerness_targets.shape)

            loss_delta = self.loss_delta(pos_delta_preds, delta_targets, avg_factor=equal_weights.sum())
            loss_depth = self.loss_depth(pos_depth_preds, depth_targets.view(-1), avg_factor=equal_weights.sum())
            loss_height = self.loss_height(pos_height_preds, height_targets.view(-1), avg_factor=equal_weights.sum())
            loss_bbox3d = self.loss_bbox(pos_bbox3d_preds, bbox3d_targets, avg_factor=equal_weights.sum())

            rank, _ = get_dist_info()
            # if rank == 0:
            #     print(torch.cat((pos_rot_preds, rot_targets), dim=-1))

            loss_rot = self.loss_rot(pos_rot_preds, rot_targets, avg_factor=equal_weights.sum())
            loss_velo = self.loss_velo(pos_velo_preds, velo_targets, avg_factor=equal_weights.sum())

            loss_centerness = self.loss_centerness(pos_centerness, pos_centerness_targets)
        else:
            loss_delta = delta_preds.sum() * 0
            loss_depth = depth_preds.sum() * 0
            loss_height = height_preds.sum() * 0
            loss_bbox3d = bbox3d_preds.sum() * 0
            loss_centerness = centerness.sum() * 0
            loss_rot = rot_preds.sum() * 0
            loss_velo = velo_preds.sum() * 0

        loss_dict = dict(
            loss_cls_1d=loss_cls,
            loss_delta_1d=loss_delta,
            loss_depth_1d=loss_depth,
            loss_height_1d=loss_height,
            loss_bbox3d_1d=loss_bbox3d,
            loss_rot_1d=loss_rot,
            loss_velo_1d=loss_velo,
            loss_centerness_1d=loss_centerness)

        return loss_dict

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'dir_cls_preds'))
    def get_bboxes(self,
                   gt_bboxes_3d,
                   gt_labels_3d,
                   preds,
                   img_inputs):

        raise NotImplementedError

    def get_targets(self,
                    gt_bboxes_3d_list,
                    gt_labels_3d_list,
                    preds,
                    img_inputs):
        """Compute regression, classification and centerss targets for points
        in multiple images.
        """

        cls_scores, bbox_preds, centerness = preds[:]
        # cls_scores: list of num_cam, num_class, width
        # bbox_preds: list of num_cam, bbox_code_size, width
        # centerness: list of num_cam, 1, width
        _, rots, trans, intrins, post_rots, post_trans = img_inputs[:6]

        # generate points for each camera
        num_cam, num_class, W = cls_scores[0].shape[:]
        dtype, device = cls_scores[0].dtype, cls_scores[0].device
        mulcam_points = torch.arange(W, dtype=dtype, device=device)
        mulcam_points = (mulcam_points + 0.5) / W
        mulcam_points = mulcam_points[None].repeat(num_cam, 1)
        mulcam_points_list = [mulcam_points for _ in range(len(cls_scores))]

        labels_3d_list, bbox_targets_3d_list, centerness_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_3d_list,
            gt_labels_3d_list,
            rots,
            trans,
            intrins,
            post_rots,
            post_trans,
            mulcam_points_list)

        labels_3d = torch.stack(labels_3d_list)
        bbox_targets_3d = torch.stack(bbox_targets_3d_list)
        centerness_targets = torch.stack(centerness_targets_list)
        return labels_3d, bbox_targets_3d, centerness_targets

    def _get_target_single(
            self,
            gt_bboxes_3d,
            gt_labels_3d,
            rots,
            trans,
            intrins,
            post_rots,
            post_trans,
            mulcam_points,
    ):
        """Compute regression and classification targets for a single image."""
        device = rots.device
        num_cam = rots.shape[0]
        num_gt = gt_bboxes_3d.corners.shape[0]

        # get points for all bboxes
        gt_centers_3d = gt_bboxes_3d.gravity_center  # num_gt, 3
        gt_corners_3d = gt_bboxes_3d.corners  # num_gt, 8, 3

        gt_points_3d = torch.cat([gt_centers_3d.unsqueeze(1), gt_corners_3d], dim=1)  # num_gt, 9, 3
        gt_points_3d = gt_points_3d.view(-1, 3)[None].repeat(num_cam, 1, 1)  # num_cam, num_gt*9, 3
        ref_pts = gt_points_3d.to(device)

        # lidar2img
        ref_pts -= trans.view(num_cam, 1, 3)
        combine = torch.inverse(rots.matmul(torch.inverse(intrins)).float())
        ref_pts = combine.view(num_cam, 1, 3, 3).matmul(ref_pts.unsqueeze(-1)).squeeze(-1)

        ref_pts_depth = ref_pts[..., 2:3]
        eps = 1e-5
        valid_mask = (ref_pts[..., 2] > eps)
        ref_pts = ref_pts[..., 0:2] / torch.maximum(ref_pts[..., 2:3], torch.ones_like(ref_pts[..., 2:3]) * eps)

        # add post aug
        post_rots = post_rots[..., :2, :2]
        ref_pts = post_rots.view(num_cam, 1, 2, 2).matmul(ref_pts.unsqueeze(-1)).squeeze(-1)
        ref_pts = ref_pts + post_trans[..., :2].view(num_cam, 1, 2)  # [n_cam, 9 * n_gt, 2]


        # norm to 0 - 1
        ogfH, ogfW = self.data_config['input_size']
        ref_pts[..., 0] = ref_pts[..., 0] / (ogfW - 1)
        ref_pts[..., 1] = ref_pts[..., 1] / (ogfH - 1)

        # get center and corner points
        ref_pts = torch.cat([ref_pts, ref_pts_depth], dim=2)
        ref_pts = ref_pts.view(num_cam, num_gt, -1, 3)  # [n_cam, n_gt, 9, 3]

        gt_center_2d = ref_pts[:, :, 0, :]   # [n_cam, n_gt, 3]
        gt_corner_2d = ref_pts[:, :, 1:, :]  # [n_cam, n_gt, 8, 3]

        valid_center_mask = (gt_center_2d[..., 0] > 0) & \
                            (gt_center_2d[..., 0] < 1) & \
                            (gt_center_2d[..., 1] > 0) & \
                            (gt_center_2d[..., 1] < 1) & \
                            (gt_center_2d[..., 2] > eps)  # [n_cam, n_gt]

        labels_3d, bbox_targets_3d, centerness_targets = multi_apply(
            self._get_target_cam,
            gt_center_2d,
            gt_corner_2d,
            [gt_bboxes_3d.tensor for _ in range(num_cam)],
            [gt_labels_3d for _ in range(num_cam)],
            valid_center_mask,
            mulcam_points)

        labels_3d = torch.stack(labels_3d)
        bbox_targets_3d = torch.stack(bbox_targets_3d)
        centerness_targets = torch.stack(centerness_targets)

        return labels_3d, bbox_targets_3d, centerness_targets

    def _get_target_cam(self,
                        gt_center_2d,
                        gt_corner_2d,
                        gt_bboxes_3d_full,
                        gt_labels_3d,
                        valid_center_mask,
                        points):
        # gt_center_2d: [num_gt, 3]
        # gt_corner_2d: [num_gt, 8, 3]
        # gt_labels_3d: [num_gt]
        # valid_center_mask: [num_gt]
        # points: [num_points(width)]


        device = gt_center_2d.device
        gt_center_2d = gt_center_2d[valid_center_mask]
        gt_corner_2d = gt_corner_2d[valid_center_mask]
        # for 3d bboxes, only use size currently, (yaw and velo not projected)
        gt_bboxes_3d = gt_bboxes_3d_full[valid_center_mask][:, 3:6].to(device)
        # print( gt_bboxes_3d_full[valid_center_mask][:, 6:])
        gt_rot = gt_bboxes_3d_full[valid_center_mask][:, 6].to(device)
        gt_rot = torch.stack((torch.sin(gt_rot), torch.cos(gt_rot)), dim=-1)
        gt_velo = gt_bboxes_3d_full[valid_center_mask][:, 7:].to(device)
        gt_labels_3d = gt_labels_3d[valid_center_mask]
        num_gt = gt_center_2d.shape[0]
        num_point = points.shape[0]

        if num_gt == 0:
            return gt_labels_3d.new_full((num_point,), self.background_label), \
                   gt_bboxes_3d.new_zeros((num_point, self.bbox_code_size)), \
                   gt_bboxes_3d.new_zeros((num_point,))

        # get bbox_targets_3d
        gt_depth = gt_center_2d[:, 2][None].repeat(num_point, 1)[..., None]
        gt_depth = (gt_depth / self.depth_gap).clamp(min=0, max=self.depth_bins+1)

        gt_height_center = gt_center_2d[:, 1][None].repeat(num_point, 1)[..., None]
        gt_height_center = (gt_height_center * self.height_bins).clamp(min=0, max=self.height_bins-EPS)

        # gt_ymin = torch.min(gt_corner_2d[..., 1], dim=1)[0] # num_gt
        # gt_ymax = torch.max(gt_corner_2d[..., 1], dim=1)[0]
        # gt_height_2d = (gt_ymax - gt_ymin)[None].repeat(num_point, 1)[..., None] * 10

        gt_bbox = gt_bboxes_3d[None].repeat(num_point, 1, 1)
        xs = points
        xs = xs[:, None].expand(num_point, num_gt)  # num_point, num_gt
        delta_xs = (xs - gt_center_2d[:, 0])[..., None]
        gt_rot = gt_rot[None].repeat(num_point, 1, 1)
        gt_velo = gt_velo[None].repeat(num_point, 1, 1)

        bbox_targets_3d = torch.cat([delta_xs, gt_depth, gt_height_center, gt_bbox, gt_rot, gt_velo], dim=-1)
        # bbox_targets_3d = torch.cat([delta_xs, gt_height_center, gt_height_2d, gt_depth, gt_bbox], dim=-1)

        # get bbox_targets_1d
        gt_xmin = torch.min(gt_corner_2d[..., 0], dim=1)[0]  # num_gt
        gt_xmax = torch.max(gt_corner_2d[..., 0], dim=1)[0]
        gap_left = xs - gt_xmin
        gap_right = gt_xmax - xs
        bbox_targets_1d = torch.stack([gap_left, gap_right], dim=-1)


        radius = 0.2
        centerness_alpha = 2.5

        # condition1: limit the regression range for each location
        max_regress_distance = bbox_targets_1d.max(-1)[0]
        inside_regress_range = (max_regress_distance <= radius) & \
                               (gap_left > 0) & \
                               (gap_right > 0)

        dists = torch.abs(bbox_targets_3d[..., 0])
        dists[inside_regress_range == 0] = INF

        min_dist, min_dist_inds = dists.min(dim=1)

        labels_3d = gt_labels_3d[min_dist_inds]
        labels_3d[min_dist == INF] = self.background_label  # set as BG

        bbox_targets_3d = bbox_targets_3d[range(num_point), min_dist_inds]
        relative_dists = torch.abs(bbox_targets_3d[..., 0]) / (1.414 * radius)
        centerness_targets = torch.exp(-centerness_alpha * relative_dists)
        return labels_3d, bbox_targets_3d, centerness_targets