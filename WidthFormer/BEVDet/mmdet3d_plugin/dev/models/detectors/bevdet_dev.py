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
class BEVDetDev(BEVDet):
    def __init__(self, benchmark_latency=False, **kwargs):
        super(BEVDetDev, self).__init__(**kwargs)
        self.benchmark_latency = benchmark_latency

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        x = self.image_encoder(img[0])  # [B * n_cam, H/16, W/16]
        vt_dict = self.img_view_transformer([x] + img[1:])
        x = vt_dict['bev_feature']
        x = self.bev_encoder(x)
        return [x]

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = None
        if self.training:
            return img_feats, pts_feats
        else:
            return img_feats, pts_feats

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""

        img_feats, _ = self.extract_feat(points, img=img, img_metas=img_metas)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)

        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

@DETECTORS.register_module()
class BEVDetWidthSup(BEVDet):
    def __init__(self, benchmark_latency=False, head_2d_cfg=dict(), **kwargs):
        super(BEVDetWidthSup, self).__init__(**kwargs)
        self.img_flatten_bbox_head = build_head(head_2d_cfg)
        self.benchmark_latency = benchmark_latency
        self.test_count = 0

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        x = self.image_encoder(img[0])  # [B * n_cam, H/16, W/16]
        vt_dict = self.img_view_transformer([x] + img[1:])

        x = vt_dict['bev_feature']
        width_feats = vt_dict.get('width_feature', None)

        x = self.bev_encoder(x)
        return [x], width_feats

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats, width_feats = self.extract_img_feat(img, img_metas)
        pts_feats = None
        if self.training:
            return img_feats, width_feats, pts_feats
        else:
            return img_feats, pts_feats

    def forward_img_flatten_train(self,
                                  width_feats,
                                  gt_bboxes_3d,
                                  gt_labels_3d,
                                  img_metas,
                                  img_inputs):
        """Forward function for flatten img prediction
        """
        outs = self.img_flatten_bbox_head(width_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs, img_metas, img_inputs]
        losses = self.img_flatten_bbox_head.loss(*loss_inputs)
        return losses

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
        img_feats, width_feats, pts_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
        assert self.with_pts_bbox
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses_width = self.forward_img_flatten_train(width_feats, gt_bboxes_3d,
                                                    gt_labels_3d, img_metas, img_inputs)

        losses.update(losses_pts)
        losses.update(losses_width)

        bev_criteria = sum([x.detach() for x in losses_pts.values()])
        losses.update({'bev_criteria': bev_criteria})
        return losses

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""

        img_feats, _ = self.extract_feat(points, img=img, img_metas=img_metas)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)

        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    

