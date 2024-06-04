import torch
import torch.nn as nn
from torch.cuda import Event

from mmdet3d.models.builder import NECKS
from torch.cuda.amp import autocast

from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import build_positional_encoding, FFN

from ..transformer.transformer_lib import BEVFeatureGenerateModule

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

class Scaling(nn.Module):
    def __init__(self, dist_size):
        super(Scaling, self).__init__()
        self.dist_size = dist_size

    def forward(self, x):
        assert len(x.shape) == 4
        return F.interpolate(x, size=self.dist_size, mode='bilinear')

    def get_resolution(self):
        return tuple(self.dist_size)

class HoriTransformerPE(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 num_self_head,
                 num_cross_head,
                 ffn_dim,
                 op_order,
                 num_attn_layers=1,
                 pe_scale = 1.,
                 cat_dim=0,
                 reduce='max'):

        super().__init__()
        self.merger = nn.Sequential(
            nn.Conv2d(in_channels + cat_dim, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))

        self.attn_layers = nn.ModuleList(
            [
                HorizonRefineFormer(mid_channels, num_self_head, num_cross_head, ffn_dim, op_order, with_pe=True)
                for _ in range(num_attn_layers)
            ]
        )
        if mid_channels != out_channels:
            self.out_proj = nn.Linear(mid_channels, out_channels)
        else:
            self.out_proj = None

        self.horizontal_pe_mlp = nn.Sequential(
            nn.Linear(mid_channels, mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, mid_channels))

        self.vertical_pe_mlp = nn.Sequential(
            nn.Linear(mid_channels, mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, mid_channels))

        self.mid_channels = mid_channels
        self.pe_scale = pe_scale

        self.h_pe_preCompute = None
        self.w_pe_preCompute = None

        self.reduce = reduce


    def forward(self, x, pe=None):
        H, W = x.shape[-2:]

        if self.training or self.h_pe_preCompute is None:
            delta_h = 1. / (H * 2)
            delta_w = 1. / (W * 2)
            h_coor = torch.linspace(delta_h, 1. - delta_h, H, device=x.device)
            w_coor = torch.linspace(delta_w, 1. - delta_w, W, device=x.device)

            h_pe = self.vertical_pe_mlp(pos2posemb1d(h_coor * self.pe_scale, self.mid_channels))
            w_pe = self.horizontal_pe_mlp(pos2posemb1d(w_coor * self.pe_scale, self.mid_channels))
            if not self.training:
                self.h_pe_preCompute = h_pe
                self.w_pe_preCompute = w_pe
        else:
            h_pe = self.h_pe_preCompute
            w_pe = self.w_pe_preCompute


        if pe is not None:
            x = self.merger(torch.cat([x, pe], 1))
        else:
            x = self.merger(x)
        if self.reduce == 'max':
            x_hori = x.max(2)[0].permute(0, 2, 1)
        elif self.reduce == 'mean':
            x_hori = x.mean(2).permute(0, 2, 1)
        else:
            raise NotImplementedError
        for idx, blk in enumerate(self.attn_layers):
            x_hori = blk(x_hori, x, self_pe=w_pe, cross_pe=h_pe)
        if self.out_proj is not None:
            x_hori = self.out_proj(x_hori)
        return x_hori

class HorizonRefineFormer(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_self_head,
                 num_cross_head,
                 ffn_dim,
                 op_order,
                 with_pe=False):

        super(HorizonRefineFormer, self).__init__()
        assert op_order in ('self_first', 'cross_first')
        self.op_order = op_order

        self.num_self_head = num_self_head
        self_head_dim = embed_dims // num_self_head
        self.self_scale = self_head_dim ** -0.5

        self.num_cross_head = num_cross_head
        cross_head_dim = embed_dims // num_cross_head
        self.cross_scale = cross_head_dim ** -0.5

        self.with_pe = with_pe

        if with_pe:
            self.norm_value_self = build_norm_layer(dict(type='LN'), embed_dims)[1]
            self.norm_value_cross = build_norm_layer(dict(type='LN'), embed_dims)[1]

        self.norm_query_self  = build_norm_layer(dict(type='LN'), embed_dims)[1]
        self.norm_query_cross = build_norm_layer(dict(type='LN'), embed_dims)[1]
        self.norm_kv          = build_norm_layer(dict(type='LN'), embed_dims)[1]

        self.q_proj_cross = nn.Linear(embed_dims, embed_dims, bias=True)
        self.k_proj_cross = nn.Linear(embed_dims, embed_dims, bias=False)
        self.v_proj_cross = nn.Linear(embed_dims, embed_dims, bias=True)
        self.proj_cross   = nn.Linear(embed_dims, embed_dims, bias=True)

        self.q_proj_self = nn.Linear(embed_dims, embed_dims, bias=True)
        self.k_proj_self = nn.Linear(embed_dims, embed_dims, bias=False)
        self.v_proj_self = nn.Linear(embed_dims, embed_dims, bias=True)
        self.proj_self   = nn.Linear(embed_dims, embed_dims, bias=True)

        _ffn_cfgs = {
            'embed_dims': embed_dims,
            'feedforward_channels': ffn_dim,
            'num_fcs': 2,
            'ffn_drop': 0.,
            'dropout_layer': dict(type='DropPath', drop_prob=0.),
            'act_cfg': dict(type='GELU'),
        }
        self.ffn = FFN(**_ffn_cfgs)
        self.ffn_norm = build_norm_layer(dict(type='LN'), embed_dims)[1]

    def _self_attn(self, x, pe=None):
        B, W, C = x.shape

        if self.with_pe:
            assert pe is not None

        if self.with_pe:
            x_q_normed = self.norm_query_self(x + pe.view(1,W,C))
            x_k_normed = x_q_normed
            x_v_normed = self.norm_value_self(x)
        else:
            x_q_normed = self.norm_query_self(x)
            x_k_normed = x_q_normed
            x_v_normed = x_q_normed

        q = rearrange(self.q_proj_self(x_q_normed), 'b n (h c)-> b h n c', h=self.num_self_head, b=B, n=W, c=C // self.num_self_head)
        k = rearrange(self.k_proj_self(x_k_normed), 'b n (h c)-> b h n c', h=self.num_self_head, b=B, n=W, c=C // self.num_self_head)
        v = rearrange(self.v_proj_self(x_v_normed), 'b n (h c)-> b h n c', h=self.num_self_head, b=B, n=W, c=C // self.num_self_head)

        attn = (q @ k.transpose(-2, -1)) * self.self_scale
        attn = attn.softmax(dim=-1)
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_self_head, b=B, n=W, c=C // self.num_self_head)
        out = self.proj_self(out)
        out = out + x
        return out

    def _cross_attn(self, x_q, x_kv, pe=None):
        B, W_, C_ = x_q.shape
        _, C, H, W = x_kv.shape
        assert (C == C_) and (W == W_)

        if self.with_pe:
            assert pe is not None

        if self.with_pe:
            x_q_normed = self.norm_query_cross(x_q)
            x_q_normed = x_q_normed.reshape(B * W, 1, C)
            x_kv = x_kv.permute(0, 3, 2, 1).reshape(B * W, H, C)
            x_k_normed = self.norm_kv(x_kv + pe.view(1,H,C))
            x_v_normed = self.norm_value_cross(x_kv)
        else:
            x_q_normed = self.norm_query_cross(x_q)
            x_q_normed = x_q_normed.reshape(B * W, 1, C)
            x_kv = x_kv.permute(0, 3, 2, 1).reshape(B * W, H, C)
            x_k_normed = self.norm_kv(x_kv)
            x_v_normed = x_k_normed

        q = rearrange(self.q_proj_cross(x_q_normed), 'b n (h c)-> b h n c', h=self.num_cross_head, b=B * W, n=1, c=C // self.num_cross_head)
        k = rearrange(self.k_proj_cross(x_k_normed), 'b n (h c)-> b h n c', h=self.num_cross_head, b=B * W, n=H, c=C // self.num_cross_head)
        v = rearrange(self.v_proj_cross(x_v_normed), 'b n (h c)-> b h n c', h=self.num_cross_head, b=B * W, n=H, c=C // self.num_cross_head)

        attn = (q @ k.transpose(-2, -1)) * self.cross_scale
        attn = attn.softmax(dim=-1)
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_cross_head, b=B * W, n=1, c=C // self.num_cross_head)
        out = out.reshape(B, W, C)
        out = self.proj_cross(out)
        out = out + x_q
        return out

    def forward(self, pooled_features, features, self_pe=None, cross_pe=None):
        if self.op_order == 'cross_first':
            x = self._cross_attn(pooled_features, features, cross_pe)
            x = self._self_attn(x, self_pe)
        else:
            x = self._self_attn(pooled_features, self_pe)
            x = self._cross_attn(x, features, cross_pe)
        x = self.ffn(self.ffn_norm(x), identity=x)
        return x

@NECKS.register_module()
class WidthFormer(nn.Module):
    def __init__(self,
                 input_dim=512,
                 embed_dims=128,
                 grid_config=None,
                 data_config=None,
                 LID=False,
                 downsample=16,
                 multiview_positional_encoding=None,
                 positional_embedding_scale=dict(enable=False, type='linear', linear_scale=10),
                 bev_query_shape=(32, 32),
                 transformer_cfgs= [],
                 refine_net_cfg=dict(),
                 norm_img_feats_key=True,
                 norm_img_feats_value=True,
                 positional_encoding='old',
                 positional_noise='none',
                 depth_pred_temp=1.,
                 with_cp=False,
                 return_width_feature=False,
                 **kwargs):
        super(WidthFormer, self).__init__()
        self.embed_dims = embed_dims
        self.with_cp = with_cp

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
        self.positional_embedding_scale = positional_embedding_scale

        self.LID = LID
        self.depth_pred_temp = depth_pred_temp
        self.frustum = self.create_frustum() # D x feat_H x feat_W x 3
        if multiview_positional_encoding is not None:
            self.multiview_positional_encoder = build_positional_encoding(multiview_positional_encoding)
        else:
            self.multiview_positional_encoder = None

        self.D = self.frustum.shape[0]
        self.bev_shape = (int((self.grid_config['xbound'][1] - self.grid_config['xbound'][0]) / self.grid_config['xbound'][2]),
                          int((self.grid_config['ybound'][1] - self.grid_config['ybound'][0]) / self.grid_config['ybound'][2]))

        if bev_query_shape is None:
            self.bev_query_shape = self.bev_shape
        else:
            self.bev_query_shape = bev_query_shape

        # --------------------------------------------------------------------------------------------------------------
        #                                                NN Layers
        # --------------------------------------------------------------------------------------------------------------
        self.img_feat_projection = nn.Conv2d(input_dim, embed_dims, kernel_size=1, padding=0)

        assert positional_encoding in ('old', 'new')
        assert positional_noise in ('none', 'gaussian', 'uniform')

        gap_y = 1. / self.bev_query_shape[0]
        gap_x = 1. / self.bev_query_shape[1]
        ys, xs = torch.meshgrid([torch.linspace(0.5 * gap_y, 1. - gap_y * 0.5, self.bev_query_shape[0]),
                                 torch.linspace(0.5 * gap_x, 1. - gap_x * 0.5, self.bev_query_shape[1])])
        self.gap_y = gap_y
        self.gap_x = gap_x

        self.positional_noise = positional_noise
        self.bev_coor = nn.Parameter(torch.stack((xs, ys), dim=-1), requires_grad=False)

        self.height_net = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims, 1, kernel_size=3, stride=1, padding=1),
        )

        self.depth_net = nn.Conv2d(embed_dims, self.D, kernel_size=1, stride=1, padding=0)

        _refine_net_cfg = dict(
            in_channels=self.embed_dims,
            mid_channels=self.embed_dims,
            out_channels=self.embed_dims,
            num_self_head=1,
            num_cross_head=1,
            ffn_dim=self.embed_dims * 4,
            op_order='self_first',
            num_attn_layers=2,
            pe_scale=10
        )
        _refine_net_cfg.update(refine_net_cfg)
        self.width_net = HoriTransformerPE(**_refine_net_cfg)

        # position embedding MLPs
        self.ego_position_mlp = nn.Sequential(
            nn.Linear(self.embed_dims * 3, self.embed_dims * 4),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 4, self.embed_dims),
        )
        if self.multiview_positional_encoder is not None:
            self.multiview_position_mlp = nn.Sequential(
                nn.Linear(self.embed_dims * 2, self.embed_dims * 4),
                nn.ReLU(),
                nn.Linear(self.embed_dims * 4, self.embed_dims),
            )
        self.bev_position_mlp = nn.Sequential(
            nn.Linear(self.embed_dims * 3, self.embed_dims * 4),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 4, self.embed_dims),
        )
        self.num_transformer_layers = len(transformer_cfgs)

        transformer_layers = []
        for i, tcfg in enumerate(transformer_cfgs):
            resolution = tcfg.get('resolution', None)
            assert resolution is not None
            if i == 0:
                assert tuple(resolution) == self.bev_query_shape
            else:
                assert resolution[0] >= transformer_layers[i-1].get_resolution()[0]
                if resolution[0] > transformer_layers[i-1].get_resolution()[0]:
                    transformer_layers.append(Scaling(resolution))

            attn_cfgs = tcfg.get('attn_cfgs', None)
            ffn_cfg = tcfg.get('ffn_cfg', None)

            layer = BEVFeatureGenerateModule(
                embed_dims=self.embed_dims,
                attn_cfgs=attn_cfgs,
                ffn_cfg=ffn_cfg,
                resolution=resolution,
                with_cp=with_cp)
            transformer_layers.append(layer)

        if transformer_layers[-1].get_resolution() != tuple(self.bev_shape):
            transformer_layers.append(Scaling(self.bev_shape))

        self.transformer_layers = nn.ModuleList(transformer_layers)

        if norm_img_feats_key:
            self.img_key_norm = build_norm_layer(dict(type='LN'), embed_dims)[1]
        else:
            self.img_key_norm = None
        if norm_img_feats_value:
            self.img_value_norm = build_norm_layer(dict(type='LN'), embed_dims)[1]
        else:
            self.img_value_norm = None

        self.return_width_feature = return_width_feature

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_config['input_size']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample

        if not self.LID:
            ds = torch.arange(*self.grid_config['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        else:
            # reference: PETR
            depth_start, depth_end, depth_step = self.grid_config['dbound']
            depth_num = (depth_end - depth_start) // depth_step
            index = torch.arange(start=0, end=depth_num, step=1).float()
            index_1 = index + 1
            bin_size = (depth_end + 1 - depth_start) / (depth_num * (1. + depth_num))
            ds = depth_start + bin_size * index * index_1
            ds = ds.view(-1, 1, 1).expand(-1, fH, fW)
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
        return points  # [x, y, z]

    def get_ego_position_embedding(self, rots, trans, intrins, post_rots, post_trans, heihgt_pred, depth_pred):
        with autocast(False):
            with torch.no_grad():
                
                ego_coor = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
                ego_coor = ((ego_coor - (self.bx - self.dx / 2.)) / self.dx)  # get ego voxel coordinate
                ego_coor = ego_coor / self.nx  # [B, n_cam, D, H, W, 3]

                nan_mask = torch.isnan(ego_coor).to(dtype=ego_coor.dtype).sum(dim=-1)
                nan_mask_r = nan_mask.to(dtype=torch.bool)  # [B, n_cam, D, H, W]
                nan_mask_degree = nan_mask.sum(dim=(2,3)).to(dtype=torch.bool)  # [B, n_cam, H, W]

                ego_coor_xy = ego_coor[..., :2] - 0.5  # [B, n_cam, D, H, W, 2]
                r = torch.sqrt(ego_coor_xy[..., 0] ** 2 + ego_coor_xy[..., 1] ** 2)

                D = ego_coor.shape[2]
                _dr = r[:, :, D - 1, 0, :]  # compute degree using end point to avoid NaN
                _dx = ego_coor_xy[:, :, D - 1, 0, :, 0]
                _dy = ego_coor_xy[:, :, D - 1, 0, :, 1]

                cos = (_dx / _dr).clamp(min=-1., max=1.)
                sin = (_dy / _dr).clamp(min=-1., max=1.)

                r[nan_mask_r] = 10
                cos[nan_mask_degree] = 10
                sin[nan_mask_degree] = 10

                degree = torch.stack((sin, cos), dim=-1)

                if self.positional_embedding_scale['enable'] and self.positional_embedding_scale['type'] == 'linear':
                    degree = degree * self.positional_embedding_scale['linear_scale']
                    r = r * self.positional_embedding_scale['linear_scale']

                degree_embed = pos2posemb2d(degree, self.embed_dims)  # [B, N, W, 2 * n_dim]
                r_embed = pos2posemb1d(r.permute(0, 1, 3, 4, 2), self.embed_dims)  # [B, N, H, W, D, n_dim]

            # depth_pred [B, N, D, H, W]
            height_pred = heihgt_pred.softmax(dim=2)[..., None] # [B, N, H, W, 1]

            r_embed = (depth_pred * r_embed).sum(dim=-2)  # [B, N, H, W, n_dim]
            r_embed = (height_pred * r_embed).sum(dim=2)  # [B, N, W, n_dim]
            polar_coor_embed = torch.cat((r_embed, degree_embed), dim=-1)  # [B, N, W, 3 * n_dim]

        polar_coor_embed = self.ego_position_mlp(polar_coor_embed)   # [B, N, W, n_dim]
        return polar_coor_embed

    def get_multiview_position_embedding(self, x):
        B, N, W, C = x.shape
        mask = torch.zeros_like(x).sum(dim=3) + 1
        position_embedding = self.multiview_positional_encoder(mask)
        position_embedding = self.multiview_position_mlp(position_embedding)
        position_embedding = position_embedding.reshape(B, N, W, C)
        return position_embedding

    def get_bev_position_embedding(self, B):
        with autocast(False):
            with torch.no_grad():
                bev_h, bev_w = self.bev_query_shape
                bev_coor = self.bev_coor[None, :, :, :].repeat(B, 1, 1, 1)  # [B, bev_h, bev_w, 2]

                # add noise here
                if self.training:
                    if self.positional_noise == 'uniform':
                        noise = torch.rand(bev_coor.shape, device=bev_coor.device) - 0.5
                        noise[..., 0] = noise[..., 0] * self.gap_y
                        noise[..., 1] = noise[..., 1] * self.gap_x
                        bev_coor += noise
                    elif self.positional_noise == 'gaussian':
                        noise = torch.normal(0., 0.5, bev_coor.shape, device=bev_coor.device)
                        noise[..., 0] = (noise[..., 0] * self.gap_y * 0.5).clamp(min=-0.5 * self.gap_y, max=0.5 * self.gap_y)
                        noise[..., 1] = (noise[..., 1] * self.gap_x * 0.5).clamp(min=-0.5 * self.gap_x, max=0.5 * self.gap_x)
                        bev_coor += noise

                bev_coor = bev_coor - 0.5
                r = torch.sqrt(bev_coor[..., 0] ** 2 + bev_coor[..., 1] ** 2)

                cos = (bev_coor[..., 0] / r).clamp(min=-1., max=1.)
                sin = (bev_coor[..., 1] / r).clamp(min=-1., max=1.)
                degree = torch.stack((sin, cos), dim=-1)

                if self.positional_embedding_scale['enable'] and self.positional_embedding_scale['type'] == 'linear':
                    r = r * self.positional_embedding_scale['linear_scale']
                    degree = degree * self.positional_embedding_scale['linear_scale']

                r_embed = pos2posemb1d(r, self.embed_dims)
                degree_embed = pos2posemb2d(degree, self.embed_dims)
                position_embedding = torch.cat((r_embed, degree_embed), dim=-1)

            position_embedding = self.bev_position_mlp(position_embedding)
            position_embedding = position_embedding.reshape(B, bev_h, bev_w, self.embed_dims).permute(0, 3, 1, 2)
        return position_embedding

    def get_bev_feature(self, bev_query, width_feats, img_position_embed):
        width_feats_key = width_feats + img_position_embed
        width_feats_value = width_feats

        B, N, W, C = width_feats.shape
        if self.img_key_norm is not None:
            width_feats_key = self.img_key_norm(width_feats_key.view(B * N, W, C)).view(B, N, W, C)
        if self.img_value_norm is not None:
            width_feats_value = self.img_value_norm(width_feats_value.view(B * N, W, C)).view(B, N, W, C)

        x = bev_query
        for idx in range(len(self.transformer_layers)):
            if type(self.transformer_layers[idx]) is Scaling:
                x = self.transformer_layers[idx](x)
                continue
            _x = self.transformer_layers[idx](x, width_feats_key, width_feats_value)
            x = _x
        return x

    def forward(self, input):
        img_feats, rots, trans, intrins, post_rots, post_trans = input
        B, N, _, H, W = img_feats.shape

        img_feats = self.img_feat_projection(img_feats.flatten(0, 1)).reshape(B, N, self.embed_dims, H, W)
        depth_pred = self.depth_net(img_feats.view(B * N, -1, H, W)).view(B, N, self.D, H, W).permute(0, 1, 3, 4, 2).softmax(dim=-1)[..., None]

        height_pred = self.height_net(img_feats.view(B * N, -1, H, W)).view(B, N, H, W)
        ego_pos_embed = self.get_ego_position_embedding(rots, trans, intrins, post_rots, post_trans, height_pred, depth_pred)
        width_features = self.width_net(img_feats.view(B * N, -1, H, W)).view(B, N, W, -1)

        if self.multiview_positional_encoder is not None:
            multiview_pos_embed = self.get_multiview_position_embedding(width_features)
            width_pos_embed = ego_pos_embed + multiview_pos_embed
        else:
            width_pos_embed = ego_pos_embed

        bev_query_pos_embed = self.get_bev_position_embedding(B)
        bev_query = bev_query_pos_embed

        bev_feat = self.get_bev_feature(bev_query, width_features, width_pos_embed)

        rets = dict()
        rets['bev_feature'] = bev_feat.contiguous()
        rets['width_feature'] = width_features
        return rets

