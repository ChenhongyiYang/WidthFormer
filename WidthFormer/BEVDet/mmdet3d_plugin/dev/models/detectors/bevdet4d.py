import torch
import torch.nn.functional as F
from torch.cuda import Event


from mmdet.models import DETECTORS
# from mmdet3d_plugin.models.detectors.bevdet import BEVDet, BEVDepth
from mmdet3d.models.detectors.bevdet import BEVDet, BEVDepth
from mmdet3d.models import builder
from mmdet3d.models.builder import build_head
from torch.cuda.amp import autocast


from mmcv.runner import get_dist_info

@DETECTORS.register_module()
class BEVDetSequential_Dev(BEVDet):
    def __init__(self, aligned=False, distill=None, pre_process=None,
                 pre_process_neck=None, detach=True, test_adj_ids=None, **kwargs):
        super(BEVDetSequential_Dev, self).__init__(**kwargs)
        self.aligned = aligned
        self.distill = distill is not None
        if self.distill:
            self.distill_net = builder.build_neck(distill)
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process)
        self.pre_process_neck = pre_process_neck is not None
        if self.pre_process_neck:
            self.pre_process_neck_net = builder.build_neck(pre_process_neck)
        self.detach = detach
        self.test_adj_ids = test_adj_ids

    def extract_img_feat(self, img, img_metas):
        inputs = img
        """Extract features of images."""
        B, N, _, H, W = inputs[0].shape
        N = N // 2
        imgs = inputs[0].view(B, N, 2, 3, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans = inputs[1:]
        extra = [rots.view(B, 2, N, 3, 3),
                 trans.view(B, 2, N, 3),
                 intrins.view(B, 2, N, 3, 3),
                 post_rots.view(B, 2, N, 3, 3),
                 post_trans.view(B, 2, N, 3)]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra
        bev_feat_list = []
        for img, rot, tran, intrin, post_rot, post_tran in zip(imgs, rots, trans, intrins, post_rots, post_trans):
            x = self.image_encoder(img)
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            x = self.img_view_transformer.depthnet(x)
            geom = self.img_view_transformer.get_geometry(rot, tran, intrin, post_rot, post_tran)
            depth = self.img_view_transformer.get_depth_dist(x[:, :self.img_view_transformer.D])
            img_feat = x[:, self.img_view_transformer.D:(
                    self.img_view_transformer.D + self.img_view_transformer.numC_Trans)]

            # Lift
            volume = depth.unsqueeze(1) * img_feat.unsqueeze(2)
            volume = volume.view(B, N, self.img_view_transformer.numC_Trans, self.img_view_transformer.D, H,
                                 W)
            volume = volume.permute(0, 1, 3, 4, 5, 2)

            # Splat
            bev_feat = self.img_view_transformer.voxel_pooling(geom, volume)

            if self.pre_process:
                bev_feat = self.pre_process_net(bev_feat)
                if self.pre_process_neck:
                    bev_feat = self.pre_process_neck_net(bev_feat)
                else:
                    bev_feat = bev_feat[0]
            bev_feat_list.append(bev_feat)
        if self.detach:
            bev_feat_list[1] = bev_feat_list[1].detach()
        if self.distill:
            bev_feat_list[1] = self.distill_net(bev_feat_list)
        bev_feat = torch.cat(bev_feat_list, dim=1)

        x = self.bev_encoder(bev_feat)
        return [x]


@DETECTORS.register_module()
class BEVDetSequentialES_Dev(BEVDetSequential_Dev):
    def __init__(self, before=False, interpolation_mode='bilinear', **kwargs):
        super(BEVDetSequentialES_Dev, self).__init__(**kwargs)
        self.before = before
        self.interpolation_mode = interpolation_mode

    def shift_feature(self, input, trans, rots):
        with autocast(False):
            n, c, h, w = input.shape
            _, v, _ = trans[0].shape

            # generate grid
            xs = torch.linspace(0, w - 1, w, dtype=input.dtype, device=input.device).view(1, w).expand(h, w)
            ys = torch.linspace(0, h - 1, h, dtype=input.dtype, device=input.device).view(h, 1).expand(h, w)
            grid = torch.stack((xs, ys, torch.ones_like(xs)), -1).view(1, h, w, 3).expand(n, h, w, 3).view(n, h, w, 3, 1)
            grid = grid

            # get transformation from current lidar frame to adjacent lidar frame
            # transformation from current camera frame to current lidar frame
            c02l0 = torch.zeros((n, v, 4, 4), dtype=grid.dtype).to(grid)
            c02l0[:, :, :3, :3] = rots[0]
            c02l0[:, :, :3, 3] = trans[0]
            c02l0[:, :, 3, 3] = 1

            # transformation from adjacent camera frame to current lidar frame
            c12l0 = torch.zeros((n, v, 4, 4), dtype=grid.dtype).to(grid)
            c12l0[:, :, :3, :3] = rots[1]
            c12l0[:, :, :3, 3] = trans[1]
            c12l0[:, :, 3, 3] = 1

            # transformation from current lidar frame to adjacent lidar frame
            l02l1 = c02l0.matmul(torch.inverse(c12l0.to(dtype=torch.float32)).to(dtype=c02l0.dtype))[:, 0, :, :].view(n, 1, 1, 4, 4)
            '''
              c02l0 * inv（c12l0）
            = c02l0 * inv(l12l0 * c12l1)
            = c02l0 * inv(c12l1) * inv(l12l0)
            = l02l1 # c02l0==c12l1
            '''

            l02l1 = l02l1[:, :, :, [True, True, False, True], :][:, :, :, :, [True, True, False, True]]

            feat2bev = torch.zeros((3, 3), dtype=grid.dtype).to(grid)
            feat2bev[0, 0] = self.img_view_transformer.dx[0]
            feat2bev[1, 1] = self.img_view_transformer.dx[1]
            feat2bev[0, 2] = self.img_view_transformer.bx[0] - self.img_view_transformer.dx[0] / 2.
            feat2bev[1, 2] = self.img_view_transformer.bx[1] - self.img_view_transformer.dx[1] / 2.
            feat2bev[2, 2] = 1
            feat2bev = feat2bev.view(1, 3, 3)
            tf = torch.inverse(feat2bev.to(dtype=torch.float32)).to(dtype=l02l1.dtype).matmul(l02l1).matmul(feat2bev)

            # transform and normalize
            grid = tf.matmul(grid)
            normalize_factor = torch.tensor([w - 1.0, h - 1.0], dtype=input.dtype, device=input.device)
            grid = grid[:, :, :, :2, 0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0
            output = F.grid_sample(input, grid.to(input.dtype), align_corners=True, mode=self.interpolation_mode)
            return output

    def extract_img_feat(self, img, img_metas):
        inputs = img
        """Extract features of images."""
        B, N, _, H, W = inputs[0].shape
        N = N // 2
        imgs = inputs[0].view(B, N, 2, 3, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans = inputs[1:]
        extra = [rots.view(B, 2, N, 3, 3),
                 trans.view(B, 2, N, 3),
                 intrins.view(B, 2, N, 3, 3),
                 post_rots.view(B, 2, N, 3, 3),
                 post_trans.view(B, 2, N, 3)]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra
        bev_feat_list = []
        for img, _, _, intrin, post_rot, post_tran in zip(imgs, rots, trans, intrins, post_rots, post_trans):
            tran = trans[0]
            rot = rots[0]
            x = self.image_encoder(img)
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            x = self.img_view_transformer.depthnet(x)
            geom = self.img_view_transformer.get_geometry(rot, tran, intrin, post_rot, post_tran)
            depth = self.img_view_transformer.get_depth_dist(x[:, :self.img_view_transformer.D])
            img_feat = x[:, self.img_view_transformer.D:(
                    self.img_view_transformer.D + self.img_view_transformer.numC_Trans)]

            # Lift
            volume = depth.unsqueeze(1) * img_feat.unsqueeze(2)
            volume = volume.view(B, N, self.img_view_transformer.numC_Trans, self.img_view_transformer.D, H, W)
            volume = volume.permute(0, 1, 3, 4, 5, 2)

            # Splat
            bev_feat = self.img_view_transformer.voxel_pooling(geom, volume)

            bev_feat_list.append(bev_feat)
        if self.before and self.pre_process:
            bev_feat_list = [self.pre_process_net(bev_feat)[0] for bev_feat in bev_feat_list]
        bev_feat_list[1] = self.shift_feature(bev_feat_list[1], trans, rots)
        if self.pre_process and not self.before:
            bev_feat_list = [self.pre_process_net(bev_feat)[0] for bev_feat in bev_feat_list]
        if self.detach:
            bev_feat_list[1] = bev_feat_list[1].detach()
        if self.distill:
            bev_feat_list[1] = self.distill_net(bev_feat_list)
        bev_feat = torch.cat(bev_feat_list, dim=1)

        x = self.bev_encoder(bev_feat)
        return [x]


@DETECTORS.register_module()
class BEVDetSequentialES_NewForward(BEVDetSequentialES_Dev):
    def extract_img_feat(self, img, img_metas):
        inputs = img
        """Extract features of images."""
        B, N, _, H, W = inputs[0].shape
        N = N // 2
        imgs = inputs[0].view(B, N, 2, 3, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans = inputs[1:]
        extra = [rots.view(B, 2, N, 3, 3),
                 trans.view(B, 2, N, 3),
                 intrins.view(B, 2, N, 3, 3),
                 post_rots.view(B, 2, N, 3, 3),
                 post_trans.view(B, 2, N, 3)]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra
        bev_feat_list = []
        for idx, (img, _, _, intrin, post_rot, post_tran) in enumerate(zip(imgs, rots, trans, intrins, post_rots, post_trans)):
            tran = trans[0]
            rot = rots[0]
            if idx > 0 and self.detach:
                with torch.no_grad():
                    x = self.image_encoder(img)
                    inputs = [x, rot, tran , intrin, post_rot, post_tran]
                    vt_dict = self.img_view_transformer(inputs)
                    bev_feat = vt_dict['bev_feature']
            else:
                x = self.image_encoder(img)
                inputs = [x, rot, tran , intrin, post_rot, post_tran]
                vt_dict = self.img_view_transformer(inputs)
                bev_feat = vt_dict['bev_feature']
            bev_feat_list.append(bev_feat)
        if self.before and self.pre_process:
            bev_feat_list = [self.pre_process_net(bev_feat)[0] for bev_feat in bev_feat_list]
        bev_feat_list[1] = self.shift_feature(bev_feat_list[1], trans, rots)
        if self.pre_process and not self.before:
            bev_feat_list = [self.pre_process_net(bev_feat)[0] for bev_feat in bev_feat_list]
        if self.detach:
            bev_feat_list[1] = bev_feat_list[1].detach()
        if self.distill:
            bev_feat_list[1] = self.distill_net(bev_feat_list)
        bev_feat = torch.cat(bev_feat_list, dim=1)

        x = self.bev_encoder(bev_feat)
        return [x]


@DETECTORS.register_module()
class BEVDetSequentialES_WidthSup(BEVDetSequentialES_Dev):
    def __init__(self, before=False, interpolation_mode='bilinear', head_2d_cfg=dict(), **kwargs):
        super(BEVDetSequentialES_Dev, self).__init__(**kwargs)
        self.before = before
        self.interpolation_mode = interpolation_mode
        self.img_flatten_bbox_head = build_head(head_2d_cfg)

    def forward_img_flatten_train(self,
                                  width_feats,
                                  gt_bboxes_3d,
                                  gt_labels_3d,
                                  img_metas,
                                  img_inputs):
        outs = self.img_flatten_bbox_head(width_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs, img_metas, img_inputs]
        losses = self.img_flatten_bbox_head.loss(*loss_inputs)
        return losses

    def extract_img_feat(self, img, img_metas):
        inputs = img
        """Extract features of images."""
        B, N, _, H, W = inputs[0].shape
        N = N // 2
        imgs = inputs[0].view(B, N, 2, 3, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans = inputs[1:]
        extra = [rots.view(B, 2, N, 3, 3),
                 trans.view(B, 2, N, 3),
                 intrins.view(B, 2, N, 3, 3),
                 post_rots.view(B, 2, N, 3, 3),
                 post_trans.view(B, 2, N, 3)]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra
        bev_feat_list = []
        for idx, (img, _, _, intrin, post_rot, post_tran) in enumerate(
                zip(imgs, rots, trans, intrins, post_rots, post_trans)):
            tran = trans[0]
            rot = rots[0]
            if idx > 0 and self.detach:
                with torch.no_grad():
                    x = self.image_encoder(img)
                    inputs = [x, rot, tran, intrin, post_rot, post_tran]
                    vt_dict = self.img_view_transformer(inputs)
                    bev_feat = vt_dict['bev_feature']
            else:
                x = self.image_encoder(img)
                inputs = [x, rot, tran, intrin, post_rot, post_tran]
                if idx == 0:
                    vt_dict = self.img_view_transformer(inputs)
                    bev_feat = vt_dict['bev_feature']
                    width_feat = vt_dict['width_feature']
                    inputs_0 = inputs
                else:
                    vt_dict = self.img_view_transformer(inputs)
                    bev_feat = vt_dict['bev_feature']

            bev_feat_list.append(bev_feat)
        if self.before and self.pre_process:
            bev_feat_list = [self.pre_process_net(bev_feat)[0] for bev_feat in bev_feat_list]
        bev_feat_list[1] = self.shift_feature(bev_feat_list[1], trans, rots)
        if self.pre_process and not self.before:
            bev_feat_list = [self.pre_process_net(bev_feat)[0] for bev_feat in bev_feat_list]
        if self.detach:
            bev_feat_list[1] = bev_feat_list[1].detach()
        if self.distill:
            bev_feat_list[1] = self.distill_net(bev_feat_list)
        bev_feat = torch.cat(bev_feat_list, dim=1)

        x = self.bev_encoder(bev_feat)
        return [x], width_feat, inputs_0

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats, width_feats, inputs_0 = self.extract_img_feat(img, img_metas)
        pts_feats = None
        if self.training:
            return img_feats, width_feats, inputs_0, pts_feats
        else:
            return img_feats, pts_feats

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        img_feats, width_feats, inputs_0, pts_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
        assert self.with_pts_bbox
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses_width = self.forward_img_flatten_train(width_feats, gt_bboxes_3d,
                                                    gt_labels_3d, img_metas, inputs_0)

        losses.update(losses_pts)
        losses.update(losses_width)

        bev_criteria = sum([x.detach() for x in losses_pts.values()])
        losses.update({'bev_criteria': bev_criteria})
        return losses
