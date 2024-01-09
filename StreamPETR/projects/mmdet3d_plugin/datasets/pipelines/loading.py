import numpy as np
from mmdet.datasets.builder import PIPELINES
import torch
import cv2

@PIPELINES.register_module()
class PointToMultiViewDepth(object):

    def __init__(self, downsample=1, min_dist=1e-5, max_dist=None):
        self.downsample = downsample
        self.min_dist=min_dist
        self.max_dist=max_dist
        # self.grid_config = grid_config

    def points2depthmap(self, points, height, width, img, cid):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        depth_map_mask = torch.zeros((height, width), dtype=torch.bool)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]

        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) \
            & (coor[:, 1] >= 0) & (coor[:, 1] < height) \
            & (depth >= self.min_dist)
        if self.max_dist is not None:
            kept1=kept1&(depth <  self.max_dist)
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + 1 - depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        depth_map_mask[coor[:, 1], coor[:, 0]] = True
        if False:
            cv2.imwrite(f"./debug_save/loading/depth_raw_img_{cid}.png", img)
            blue = np.array([255,0,0]).reshape(1,1,3)
            red = np.array([0,0,255]).reshape(1,1,3)
            depth_max = depth.max().cpu().numpy()
            img_resize = cv2.resize(img,(depth_map.shape[1],depth_map.shape[0]))
            for ind in range(len(coor)):
                depth_single = depth_map[coor[ind, 1], coor[ind, 0]].cpu().numpy()
                color = blue*depth_single/depth_max + red*(1-depth_single/depth_max)
                cv2.circle(
                    img_resize,
                    center=(int(coor[ind, 0].cpu().numpy()), int(coor[ind, 1].cpu().numpy())),
                    radius=1,
                    color=(int(color[0,0,0]),int(color[0,0,1]),int(color[0,0,2])),
                    thickness=1,
                )
            cv2.imwrite(f"./debug_save/loading/depth_img_{cid}.png", img_resize)
        return depth_map,depth_map_mask


    def __call__(self, results):
        imgs = results['img']
        pts = results['points'].tensor[:, :3]
        lidar2img_rt = results['lidar2img']
        pts = torch.cat(
            [pts, torch.ones((pts.shape[0], 1), dtype=pts.dtype)], -1)
        lidar2img_rt = torch.tensor(lidar2img_rt, dtype=pts.dtype)
        depth_map_list = []
        depth_map_mask_list = []
        for cid in range(len(imgs)):
            points_img = pts.matmul(lidar2img_rt[cid].T)
            points_img[:, :2] /= points_img[:, 2:3]
            depth_map ,depth_mask_map= self.points2depthmap(points_img, imgs[cid].shape[0],
                                             imgs[cid].shape[1], imgs[cid], cid)
            # if False:
                # blue = np.array([255,0,0]).reshape(1,1,3)
                # red = np.array([0,0,255]).reshape(1,1,3)
                # depth_ = depth_map[:,:,None].repeat(1,1,3).cpu().numpy()
                # depth_max = depth_.max()
                # depth_img = red*depth_/depth_max + blue*(1-depth_/depth_max)
                # img = cv2.resize(imgs[cid],(depth_.shape[1],depth_.shape[0]))
                # # depth_img = red*np.ones_like(depth)
                # cv2.imwrite(f"./debug_save/depth_img_{cid}.png", depth_img*0.5+img*0.5)
            depth_map_list.append(depth_map)
            depth_map_mask_list.append(depth_mask_map)


        if False:
            # 1  ori setting in bevdepth
            p = results['img_inputs']
            cid = 0
            rots = p[cid][1]
            trans = p[cid][2]
            intrins = p[cid][3]
            post_rots = p[cid][4]
            post_trans = p[cid][5]

            # pts = points_lidar.tensor[:, :3].numpy()
            pts = np.array([[1, 2, 3]], dtype=np.float64)
            combine = rots@(np.linalg.inv(intrins))
            combine_inv = np.linalg.inv(combine)
            points_img = (pts -
                          trans[None, :])@(combine_inv.T)
            points_img = np.concatenate(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)  # / Z  to (u,v)
            points_img = points_img@(
                post_rots.T) + post_trans[None, :]
            # 2 lidar2image
            # pts = points_lidar.tensor[:, :3].numpy()
            pts = np.array([[1, 2, 3]], dtype=np.float64)
            ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
            pts = np.concatenate([pts, ones], -1)
            points_img_2 = pts@lidar2img_rt[cid].T
            points_img_2[:, :2] /= points_img_2[:, 2]
        depth_map = torch.stack(depth_map_list)
        depth_map_mask = torch.stack(depth_map_mask_list)
        results['depth_map'] = depth_map
        results['depth_map_mask'] = depth_map_mask
        return results

@PIPELINES.register_module()
class LoadDepthByMapplingPoints2Images(object):
    def __init__(self, src_size, input_size, downsample=1, min_dist=1e-5, max_dist=None):
        self.src_size = src_size
        self.input_size = input_size
        self.downsample = downsample
        self.min_dist = min_dist
        self.max_dist = max_dist

    def mask_points_by_range(self, points_2d, depths, img_size):
        """
        Args:
            points2d: (N, 2)
            depths:   (N, )
            img_size: (H, W)
        Returns:
            points2d: (N', 2)
            depths:   (N', )
        """
        H, W = img_size
        mask = np.ones(depths.shape, dtype=np.bool)
        mask = np.logical_and(mask, points_2d[:, 0] >= 0)
        mask = np.logical_and(mask, points_2d[:, 0] < W)
        mask = np.logical_and(mask, points_2d[:, 1] >= 0)
        mask = np.logical_and(mask, points_2d[:, 1] < H)
        points_2d = points_2d[mask]
        depths = depths[mask]
        return points_2d, depths

    def mask_points_by_dist(self, points_2d, depths, min_dist, max_dist):
        mask = np.ones(depths.shape, dtype=np.bool)
        mask = np.logical_and(mask, depths >= min_dist)
        if max_dist is not None:
            mask = np.logical_and(mask, depths <= max_dist)
        points_2d = points_2d[mask]
        depths = depths[mask]
        return points_2d, depths

    def get_rot(self, h):
        return np.array([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def transform_points2d(self, points_2d, depths, resize, crop, flip, rotate):
        points_2d = points_2d * resize
        points_2d = points_2d - crop[:2]  # (N_points, 2)
        points_2d, depths = self.mask_points_by_range(points_2d, depths, (crop[3] - crop[1], crop[2] - crop[0]))

        if flip:
            # A = np.array([[-1, 0], [0, 1]])
            # b = np.array([crop[2] - crop[0] - 1, 0])
            # points_2d = points_2d.dot(A.T) + b
            points_2d[:, 0] = (crop[2] - crop[0]) - 1 - points_2d[:, 0]

        A = self.get_rot(rotate / 180 * np.pi)
        b = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.dot(-b) + b

        points_2d = points_2d.dot(A.T) + b

        points_2d, depths = self.mask_points_by_range(points_2d, depths, self.input_size)

        return points_2d, depths

    def __call__(self, results):
        imgs = results["img"]  # List[(H, W, 3), (H, W, 3), ...]
        # extrinsics = results["extrinsics"]      # List[(4, 4), (4, 4), ...]
        # ori_intrinsics = results["ori_intrinsics"]      # List[(4, 4), (4, 4), ...]
        ori_lidar2imgs = results["lidar2img"]       # List[(4, 4), (4, 4), ...]

        assert len(imgs) == len(ori_lidar2imgs), \
            f'imgs length {len(imgs)} != ori_lidar2imgs length {len(ori_lidar2imgs)}'

        resize = results['multi_view_resize']      # float
        resize_dims = results['multi_view_resize_dims']     # (2, )   (resize_W, resize_H)
        crop = results['multi_view_crop']    # (4, )     (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = results['multi_view_flip']          # bool
        rotate = results['multi_view_rotate']      # float

        # augmentation (resize, crop, horizontal flip, rotate)
        # resize: float, resize的比例
        # resize_dims: Tuple(W, H), resize后的图像尺寸
        # crop: (crop_w, crop_h, crop_w + fW, crop_h + fH)
        # flip: bool
        # rotate: float 旋转角度

        N_views = len(imgs)
        H, W = self.input_size
        dH, dW = H // self.downsample, W // self.downsample
        depth_map_list = []
        depth_map_mask_list = []

        points_lidar = results['points'].tensor[:, :3].numpy()     # (N_points, 3)
        points_lidar = np.concatenate([points_lidar, np.ones((points_lidar.shape[0], 1))], axis=-1)   # (N_points, 4)

        for idx in range(N_views):
            # # lidar --> camera
            # lidar2camera = extrinsics[idx]  # (4, 4)
            # points_camera = points_lidar.dot(lidar2camera.T)   # (N_points, 4)     4: (x, y, z, 1)
            #
            # # camera --> img
            # ori_intrins = ori_intrinsics[idx]   # (4, 4)
            # points_image = points_camera.dot(ori_intrins.T)    # (N_points, 4)     4: (du, dv, d, 1)

            # lidar --> img
            points_image = points_lidar.dot(ori_lidar2imgs[idx].T)
            points_image = points_image[:, :3]  # (N_points, 3)     3: (du, dv, d)
            points_2d = points_image[:, :2] / points_image[:, 2:3]  # (N_points, 2)     2: (u, v)
            depths = points_image[:, 2]  # (N_points, )
            points_2d, depths = self.mask_points_by_range(points_2d, depths, self.src_size)

            # aug
            points_2d, depths = self.transform_points2d(points_2d, depths, resize, crop, flip, rotate)
            points_2d, depths = self.mask_points_by_dist(points_2d, depths, self.min_dist, self.max_dist)

            # downsample
            points_2d = np.round(points_2d / self.downsample)
            points_2d, depths = self.mask_points_by_range(points_2d, depths, (dH, dW))

            depth_map = np.zeros(shape=(dH, dW), dtype=np.float32)  # (dH, dW)
            depth_map_mask = np.zeros(shape=(dH, dW), dtype=np.bool)   # (dH, dW)

            ranks = points_2d[:, 0] + points_2d[:, 1] * dW
            sort = (ranks + depths / 1000.).argsort()
            points_2d, depths, ranks = points_2d[sort], depths[sort], ranks[sort]

            kept = np.ones(points_2d.shape[0], dtype=np.bool)
            kept[1:] = (ranks[1:] != ranks[:-1])
            points_2d, depths = points_2d[kept], depths[kept]
            points_2d = points_2d.astype(np.long)

            depth_map[points_2d[:, 1], points_2d[:, 0]] = depths
            depth_map_mask[points_2d[:, 1], points_2d[:, 0]] = 1
            depth_map_list.append(depth_map)
            depth_map_mask_list.append(depth_map_mask)

        depth_map = np.stack(depth_map_list, axis=0)      # (N_view, dH, dW)
        depth_map_mask = np.stack(depth_map_mask_list, axis=0)    # (N_view, dH, dW)


        # for vis
        # import cv2
        # for idx in range(len(imgs)):
        #     ori_img = imgs[idx]  # (H, W, 3)
        #     ori_img = ori_img.astype(np.uint8)
        #     cv2.imshow("ori_img", ori_img)
        #
        #     curr_img = cv2.resize(src=ori_img,
        #                           dsize=(ori_img.shape[1]//self.downsample, ori_img.shape[0]//self.downsample))
        #     cv2.imshow("curr_img", curr_img)
        #
        #     cv2.imshow("mask", depth_map_mask[idx].astype(np.uint8) * 255)
        #     cur_depth_map = depth_map[idx]
        #     cur_depth_map_mask = depth_map_mask[idx]
        #
        #     cur_depth_map = cur_depth_map / 60 * 255
        #     cur_depth_map = cur_depth_map.astype(np.uint8)
        #     cur_depth_map = cv2.applyColorMap(cur_depth_map, cv2.COLORMAP_RAINBOW)
        #
        #     # cur_depth_map = cv2.resize(src=cur_depth_map, dsize=(img.shape[1], img.shape[0]))
        #
        #     curr_img[cur_depth_map_mask] = cur_depth_map[cur_depth_map_mask]
        #     cv2.imshow("depth map", curr_img)
        #
        #     while(True):
        #         k = cv2.waitKey(0)
        #         if k == 27:
        #             cv2.destroyAllWindows()
        #             break

        results['depth_map'] = depth_map
        results['depth_map_mask'] = depth_map_mask

        results.pop('multi_view_resize')
        results.pop('multi_view_resize_dims')
        results.pop('multi_view_crop')
        results.pop('multi_view_flip')
        results.pop('multi_view_rotate')
        return results