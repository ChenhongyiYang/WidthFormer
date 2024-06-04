import torch
import math
import numpy as np

from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class ParamNoise(object):

    def __init__(self, x_std=0., y_std=0., z_std=0., cam_z=0., random_mode='normal'):
        self.x_std = x_std
        self.y_std = y_std
        self.z_std = z_std
        self.cam_z = cam_z
        self.random_mode = random_mode

    def __call__(self, results):

        imgs, rots, trans, intrins, post_rots, post_trans = results['img_inputs'][:]

        N = rots.shape[0]
        # add noise here
        if self.random_mode == 'normal':
            x_degrees = np.random.normal(0., self.x_std, (N,)) * math.pi / 180.
            y_degrees = np.random.normal(0., self.y_std, (N,)) * math.pi / 180.
            z_degrees = np.random.normal(0., self.z_std, (N,)) * math.pi / 180.
        else:
            x_degrees = np.random.uniform(-self.x_std, self.x_std, (N,)) * math.pi / 180.
            y_degrees = np.random.uniform(-self.y_std, self.y_std, (N,)) * math.pi / 180.
            z_degrees = np.random.uniform(-self.z_std, self.z_std, (N,)) * math.pi / 180.

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
        noise_rots = torch.stack(noise_rots, dim=0)
        rots = rots.matmul(noise_rots)

        trans[:, 2] = trans[:, 2] + self.cam_z

        results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans)
        return results