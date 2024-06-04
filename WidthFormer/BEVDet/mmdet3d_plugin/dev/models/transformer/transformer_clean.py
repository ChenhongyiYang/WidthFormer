import copy
import math
from typing import Sequence

import mmcv.cnn
from torch.cuda import Event
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.utils.checkpoint as cp

from einops import rearrange

from mmcv.cnn import build_norm_layer, build_conv_layer, build_activation_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_, constant_init
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.bricks.registry import DROPOUT_LAYERS

from ..utils.fns import pos2posemb1d, pos2posemb2d, pos2posemb3d

from ...amp.checkpoint import checkpoint
from .build import build_selfAttention_layer, build_crossAttention_layer, SELF_ATTENTION, CROSS_ATTENTION


class SHSemanticBiasAttn(nn.Module):
    def __init__(self,
                 dim,
                 out_dim=None,
                 qkv_bias=True,
                 attn_drop=0.,
                 proj_drop=0.,
                 q_extra_dim=0,
                 k_extra_dim=0,
                 v_extra_dim=0,
                 attn_act='softmax',
                 semantic_detach=True):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.scale = dim**-0.5

        self.q_proj = nn.Linear(dim+q_extra_dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim+k_extra_dim, dim, bias=False)
        self.v_proj = nn.Linear(dim+v_extra_dim, dim, bias=qkv_bias)
        self.v_semantic_proj = nn.Linear(dim+v_extra_dim, 1, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.semantic_detach = semantic_detach

        self.attn_act = attn_act

        self.cached_q = None

    def forward(self, query, key, value, att_bias=None, ret_attn=False):
        bq, nq, cq = query.shape
        bk, nk, ck = key.shape
        bv, nv, cv = value.shape

        if not self.training:
            if self.cached_q is None:
                q = self.q_proj(query)
                self.cached_q = q
            else:
                q = self.cached_q
        else:
            q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        if self.semantic_detach:
            semantic = self.v_semantic_proj(value.detach()).reshape(bv, 1, nv)
        else:
            semantic = self.v_semantic_proj(value).reshape(bv, 1, nv)  # [B, nk, 1]

        # [B, nq, nk]
        attn = (q @ k.transpose(-2, -1)) * self.scale + semantic
        if att_bias is not None:
            attn = attn + att_bias.unsqueeze(dim=1)
        if self.attn_act == 'softmax':
            attn = attn.softmax(dim=-1)
        elif self.attn_act == 'sigmoid':
            attn = attn.sigmoid()
        else:
            raise NotImplementedError

        if self.training:
            attn = self.attn_drop(attn)

        out = attn @ v
        out = self.proj(out)
        if self.training:
            out = self.proj_drop(out)

        if not ret_attn:
            return out
        else:
            return out, attn

class ConvFFN(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=1024,
                 num_fcs=2,
                 add_identity=True,
                 init_cfg=None,
                 **kwargs):
        super(ConvFFN, self).__init__(init_cfg)
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = embed_dims

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feedforward_channels, (3,3), (1,1), (1,1), bias=False),
                    nn.BatchNorm2d(feedforward_channels),
                    nn.ReLU(inplace=True))
                )
            in_channels = feedforward_channels
        layers.append(nn.Conv2d(feedforward_channels, embed_dims, (1,1), (1,1)))
        self.layers = nn.Sequential(*layers)
        self.add_identity = add_identity

    def forward(self, x, identity=None):
        """Forward function for `FFN`.
        The function would add x to the output tensor if residue is None.
        """
        out = self.layers(x)
        if not self.add_identity:
            return out
        if identity is None:
            identity = x
        return identity + out

class SHSemanticBiasAttnLayer(nn.Module):
    def __init__(self,
                 embed_dims,
                 ffn_dim=0,
                 key_is_query=False,
                 value_is_key=False,
                 qkv_bias=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 key_normed=False,
                 value_normed=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 use_cffn=False,
                 ffn_cfg=dict(),
                 **kwargs):
        super().__init__()
        self.with_cp = with_cp

        self.norm_query = build_norm_layer(norm_cfg, embed_dims)[1]

        if not key_is_query and not key_normed:
            self.norm_key = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm_key = None
        self.key_is_query = key_is_query

        if not value_is_key and not value_normed:
            self.norm_value = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm_value = None
        self.value_is_key = value_is_key

        self.attn = SHSemanticBiasAttn(
            embed_dims,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_act='softmax',
            semantic_detach=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        _ffn_cfg = copy.deepcopy(ffn_cfg)
        if ffn_cfg.get('ffn_dim', None) is None:
            _ffn_cfg['ffn_dim'] = ffn_dim
        if ffn_cfg.get('use_cffn', None) is None:
            _ffn_cfg['use_cffn'] = use_cffn

        if _ffn_cfg['ffn_dim'] > 0:
            _ffn_cfgs_custom = copy.deepcopy(_ffn_cfg)
            _ffn_cfgs_custom.update({
                'embed_dims': embed_dims,
                'feedforward_channels': _ffn_cfg['ffn_dim'],
                'num_fcs': _ffn_cfg.get('num_fcs', 2),
                'ffn_drop': 0,
                'dropout_layer': dict(type='DropPath', drop_prob=0),
                'act_cfg': act_cfg,
            })
            if not _ffn_cfg['use_cffn']:
                self.ffn = FFN(**_ffn_cfgs_custom)
                self.ffn_norm = build_norm_layer(norm_cfg, embed_dims)[1]
            else:
                self.ffn = ConvFFN(**_ffn_cfgs_custom)

                self.ffn_norm = None
        else:
            self.ffn, self.ffn_norm = None, None
        self.use_cffn = _ffn_cfg['use_cffn']

        self.query_preCompute = None
        # self.normed_query_pre_Compute = None

        self.register_buffer('hit_record', torch.zeros((128, 128, 6)))
        self.register_buffer('N_frame', torch.zeros((1,)))

    def forward(self, query, key, value, query_pos_embed, key_pos_embed, reshape_back):
        if self.training:
            return self.forward_train(query, key, value, query_pos_embed, key_pos_embed, reshape_back)
        else:
            return self.forward_test(query, key, value, query_pos_embed, key_pos_embed, reshape_back)

    def forward_test(self, query, key, value, query_pos_embed, key_pos_embed, reshape_back):
        assert len(query.shape) == 4
        assert len(key.shape) == 3
        assert len(value.shape) == 3
        assert key_pos_embed is None
        assert query_pos_embed is None
        assert self.norm_key is None
        assert self.norm_value is None

        _query = query
        B, C, qH, qW = _query.shape
        _query = _query.reshape(B, C, -1).permute(0, 2, 1)
        if self.query_preCompute is None:
            if self.norm_query is not None:
                q = self.norm_query(_query)
            else:
                q = _query
            self.query_preCompute = q
        else:
            q = self.query_preCompute

        _query = _query + self.attn(q, key, value)
        if self.ffn is not None:
            if self.use_cffn:
                _query = _query.permute(0, 2, 1).reshape(B, C, qH, qW)
                _query = self.ffn(_query, identity=_query)
                if reshape_back:
                    return _query
            else:
                _query = self.ffn(self.ffn_norm(_query), identity=_query)
        if reshape_back:
            _query = _query.permute(0, 2, 1).reshape(B, C, qH, qW)
        return _query

    def forward_train(self, query, key, value, query_pos_embed, key_pos_embed, reshape_back):
        def _inner_forward(query, key, value, query_pos_embed, key_pos_embed):
            assert len(query.shape) == 4
            assert len(key.shape) == 3
            assert len(value.shape) == 3

            assert key_pos_embed is None
            assert query_pos_embed is None
            assert self.norm_key is None
            assert self.norm_value is None

            _query = query

            B, C, qH, qW = _query.shape
            _query = _query.reshape(B, C, -1).permute(0, 2, 1)
            if self.norm_query is not None:
                q = self.norm_query(_query)
            else:
                q = _query
            _query = _query + self.drop_path(self.attn(q, key, value))

            if self.ffn is not None:
                if self.use_cffn:
                    _query = _query.permute(0, 2, 1).reshape(B, C, qH, qW)
                    _query = self.ffn(_query, identity=_query)
                    if reshape_back:
                        return _query
                else:
                    _query = self.ffn(self.ffn_norm(_query), identity=_query)
            if reshape_back:
                _query = _query.permute(0, 2, 1).reshape(B, C, qH, qW)
            return _query

        if self.with_cp:
            return checkpoint(_inner_forward, query, key, value, query_pos_embed, key_pos_embed)
        else:
            return _inner_forward(query, key, value, query_pos_embed, key_pos_embed)

@CROSS_ATTENTION.register_module()
class WidthSHSemanticBiasAttnLayer(SHSemanticBiasAttnLayer):
    def forward(self, query, key, value, query_pos_embed, key_pos_embed, reshape_back):
        if not self.training:
            return self.forward_test(query, key, value, query_pos_embed, key_pos_embed, reshape_back)
        else:
            return self.forward_train(query, key, value, query_pos_embed, key_pos_embed, reshape_back)

    def forward_test(self, query, key, value, query_pos_embed, key_pos_embed, reshape_back):
        assert len(query.shape) == 4
        assert len(key.shape) == 3
        assert len(value.shape) == 3

        assert key_pos_embed is None
        assert query_pos_embed is None
        assert self.norm_key is None
        assert self.norm_value is None

        _query = query
        B, C, qH, qW = _query.shape
        _query = _query.reshape(B, C, -1).permute(0, 2, 1)
        if self.query_preCompute is None:
            if self.norm_query is not None:
                q = self.norm_query(_query)
            else:
                q = _query
            self.query_preCompute = q
        else:
            q = self.query_preCompute

        attn_res, attn_mat = self.attn(q, key, value, ret_attn=True)
        _query = _query + attn_res

        if self.ffn is not None:
            if self.use_cffn:
                _query = _query.permute(0, 2, 1).reshape(B, C, qH, qW)
                _query = self.ffn(_query, identity=_query)
                if reshape_back:
                    return _query, attn_mat
            else:
                _query = self.ffn(self.ffn_norm(_query), identity=_query)
        if reshape_back:
            _query = _query.permute(0, 2, 1).reshape(B, C, qH, qW)
        return _query, attn_mat


    def forward_train(self, query, key, value, query_pos_embed, key_pos_embed, reshape_back):
        def _inner_forward(query, key, value, query_pos_embed, key_pos_embed):
            assert len(query.shape) == 4
            assert len(key.shape) == 3
            assert len(value.shape) == 3

            assert key_pos_embed is None
            assert query_pos_embed is None
            assert self.norm_key is None
            assert self.norm_value is None

            _query = query

            B, C, qH, qW = _query.shape
            _query = _query.reshape(B, C, -1).permute(0, 2, 1)

            if self.norm_query is not None:
                q = self.norm_query(_query)
            else:
                q = _query

            attn_res, attn_mat = self.attn(q, key, value, ret_attn=True)
            _query = _query + self.drop_path(attn_res)

            if self.ffn is not None:
                if self.use_cffn:
                    _query = _query.permute(0, 2, 1).reshape(B, C, qH, qW)
                    _query = self.ffn(_query, identity=_query)
                    if reshape_back:
                        return _query, attn_mat
                else:
                    _query = self.ffn(self.ffn_norm(_query), identity=_query)
            if reshape_back:
                _query = _query.permute(0, 2, 1).reshape(B, C, qH, qW)
            return _query, attn_mat

        if self.with_cp:
            return checkpoint(_inner_forward, query, key, value, query_pos_embed, key_pos_embed)
        else:
            return _inner_forward(query, key, value, query_pos_embed, key_pos_embed)

class Feature2BEVModule(nn.Module):
    def __init__(self,
                 embed_dims,
                 attn_cfgs,
                 with_cp=False,
                 **kwargs):
        super().__init__()
        self.with_cp = with_cp
        self.embed_dims = embed_dims

        attn_layers = []
        attn_types = []
        for cfg in attn_cfgs:
            _cfg = copy.deepcopy(cfg)
            attn_type = _cfg.pop('attn_type', None)
            assert attn_type in ('self', 'cross')
            if _cfg.get('embed_dims', None) is None:
                _cfg['embed_dims'] = embed_dims

            if attn_type == 'self':
                attn_layer = build_selfAttention_layer(_cfg)
            else:
                attn_layer = build_crossAttention_layer(_cfg)
            attn_layers.append(attn_layer)
            attn_types.append(attn_type)

        self.attn_layers = nn.ModuleList(attn_layers)
        self.attn_types = attn_types

    def forward(self, bev_query, img_feats_key, img_feats_value, return_attn=False):
        def _inner_forward(bev_query, img_feats_key, img_feats_value):
            assert len(bev_query.shape) == 4

            B, C, H, W = bev_query.shape
            N = img_feats_key.shape[1]
            assert C == self.embed_dims

            x = bev_query

            for idx, layer in enumerate(self.attn_layers):
                attn_type = self.attn_types[idx]
                if attn_type == 'self':
                    x = layer(x, None, reshape_back=True)
                else:
                    x, attn_mat = layer(x, img_feats_key, img_feats_value, None, None, reshape_back=True)

            assert len(x.shape) == 4
            if not return_attn:
                return x
            else:
                return x, attn_mat

        if self.with_cp:
            return checkpoint(_inner_forward, bev_query, img_feats_key, img_feats_value)
        else:
            return _inner_forward(bev_query, img_feats_key, img_feats_value)


class Feature2BEVModuleForAttnExp(nn.Module):
    def __init__(self,
                 embed_dims,
                 attn_cfgs,
                 with_cp=False,
                 **kwargs):
        super().__init__()
        self.with_cp = with_cp
        self.embed_dims = embed_dims

        attn_layers = []
        attn_types = []
        for cfg in attn_cfgs:
            _cfg = copy.deepcopy(cfg)
            attn_type = _cfg.pop('attn_type', None)
            assert attn_type in ('self', 'cross')
            if _cfg.get('embed_dims', None) is None:
                _cfg['embed_dims'] = embed_dims

            if attn_type == 'self':
                attn_layer = build_selfAttention_layer(_cfg)
            else:
                attn_layer = build_crossAttention_layer(_cfg)
            attn_layers.append(attn_layer)
            attn_types.append(attn_type)

        self.attn_layers = nn.ModuleList(attn_layers)
        self.attn_types = attn_types

    def forward(self, bev_query, img_feats_key, img_feats_value, return_attn=False, inds=None, ret_full_attn=False):
        assert len(bev_query.shape) == 4

        B, C, H, W = bev_query.shape
        N = img_feats_key.shape[1]
        assert C == self.embed_dims

        x = bev_query

        for idx, layer in enumerate(self.attn_layers):
            attn_type = self.attn_types[idx]
            if attn_type == 'self':
                x = layer(x, None, reshape_back=True)
            else:
                x, attn_mat = layer(x, img_feats_key, img_feats_value, None, None, reshape_back=True)


        assert attn_mat.shape[0] == 1
        attn_mat = attn_mat[0]
        if ret_full_attn:
            return attn_mat
        if inds is None:
            topk_inds = torch.topk(attn_mat * 1e10, k=5, dim=-1)[1]
            topk_attns = torch.gather(attn_mat, -1, topk_inds.to(dtype=torch.int64))
            # print(topk_attns[1000])
            return topk_inds, topk_attns
        else:
            topk_attns = torch.gather(attn_mat, -1, inds.to(dtype=torch.int64))
            return topk_attns




class WidthFeatRefineFormer(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_self_head,
                 num_cross_head,
                 ffn_dim,
                 op_order,
                 with_pe=False):
        super(WidthFeatRefineFormer, self).__init__()
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
        return out, attn

    def forward(self, pooled_features, features, self_pe=None, cross_pe=None):
        if self.op_order == 'cross_first':
            x, cross_attn_mat = self._cross_attn(pooled_features, features, cross_pe)
            x = self._self_attn(x, self_pe)
        else:
            x = self._self_attn(pooled_features, self_pe)
            x, cross_attn_mat = self._cross_attn(x, features, cross_pe)
        x = self.ffn(self.ffn_norm(x), identity=x)
        return x, cross_attn_mat

class WidthFeatRefineFormerPostNorm(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_self_head,
                 num_cross_head,
                 ffn_dim,
                 op_order,
                 with_pe=False):
        super(WidthFeatRefineFormerPostNorm, self).__init__()
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
            self.self_norm = build_norm_layer(dict(type='LN'), embed_dims)[1]
            self.cross_norm = build_norm_layer(dict(type='LN'), embed_dims)[1]


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
            q = x + pe.view(1,W,C)
        else:
            q = x
        k = q
        v = x

        q = rearrange(self.q_proj_self(q), 'b n (h c)-> b h n c', h=self.num_self_head, b=B, n=W, c=C // self.num_self_head)
        k = rearrange(self.k_proj_self(k), 'b n (h c)-> b h n c', h=self.num_self_head, b=B, n=W, c=C // self.num_self_head)
        v = rearrange(self.v_proj_self(v), 'b n (h c)-> b h n c', h=self.num_self_head, b=B, n=W, c=C // self.num_self_head)

        attn = (q @ k.transpose(-2, -1)) * self.self_scale
        attn = attn.softmax(dim=-1)
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_self_head, b=B, n=W, c=C // self.num_self_head)
        out = self.proj_self(out)
        out = out + x
        out = self.self_norm(out)
        return out

    def _cross_attn(self, x_q, x_kv, pe=None):
        B, W_, C_ = x_q.shape
        _, C, H, W = x_kv.shape
        assert (C == C_) and (W == W_)

        if self.with_pe:
            assert pe is not None

        q = x_q.reshape(B * W, 1, C)
        x_kv = x_kv.permute(0, 3, 2, 1).reshape(B * W, H, C)
        k = x_kv + pe.view(1,H,C)
        v = x_kv

        q = rearrange(self.q_proj_cross(q), 'b n (h c)-> b h n c', h=self.num_cross_head, b=B * W, n=1, c=C // self.num_cross_head)
        k = rearrange(self.k_proj_cross(k), 'b n (h c)-> b h n c', h=self.num_cross_head, b=B * W, n=H, c=C // self.num_cross_head)
        v = rearrange(self.v_proj_cross(v), 'b n (h c)-> b h n c', h=self.num_cross_head, b=B * W, n=H, c=C // self.num_cross_head)

        attn = (q @ k.transpose(-2, -1)) * self.cross_scale
        attn = attn.softmax(dim=-1)
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_cross_head, b=B * W, n=1, c=C // self.num_cross_head)
        out = out.reshape(B, W, C)
        out = self.proj_cross(out)
        out = out + x_q
        out = self.cross_norm(out)
        return out, attn

    def forward(self, pooled_features, features, self_pe=None, cross_pe=None):
        if self.op_order == 'cross_first':
            x, cross_attn_mat = self._cross_attn(pooled_features, features, cross_pe)
            x = self._self_attn(x, self_pe)
        else:
            x = self._self_attn(pooled_features, self_pe)
            x, cross_attn_mat = self._cross_attn(x, features, cross_pe)
        x = self.ffn(self.ffn_norm(x), identity=x)
        return x, cross_attn_mat





class WidthFeatRefineModule(nn.Module):
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
                 reduce='max',
                 with_conv_refine=False,
                 post_norm=False,
                 dummy=False):

        super().__init__()
        self.reduce = reduce
        self.dummy = dummy

        self.merger = nn.Sequential(
            nn.Conv2d(in_channels + cat_dim, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True))

        if mid_channels != out_channels:
            self.out_proj = nn.Linear(mid_channels, out_channels)
        else:
            self.out_proj = None
        if dummy:
            return

        if post_norm:
            self.attn_layers = nn.ModuleList(
                [
                    WidthFeatRefineFormerPostNorm(mid_channels, num_self_head, num_cross_head, ffn_dim, op_order, with_pe=True)
                    for _ in range(num_attn_layers)
                ]
            )
        else:
            self.attn_layers = nn.ModuleList(
                [
                    WidthFeatRefineFormer(mid_channels, num_self_head, num_cross_head, ffn_dim, op_order, with_pe=True)
                    for _ in range(num_attn_layers)
                ]
            )
        self.horizontal_pe_mlp = nn.Sequential(
            nn.Linear(mid_channels, mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, mid_channels))

        self.vertical_pe_mlp = nn.Sequential(
            nn.Linear(mid_channels, mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, mid_channels))

        if with_conv_refine:
            self.conv1 = nn.Sequential(
                nn.Conv1d(
                    mid_channels,
                    mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm1d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    mid_channels,
                    mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm1d(mid_channels),
                nn.ReLU(inplace=True),
            )

            self.conv2 = nn.Sequential(
                nn.Conv1d(
                    mid_channels,
                    mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm1d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    mid_channels,
                    mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm1d(mid_channels),
                nn.ReLU(inplace=True),
            )

            self.conv3 = nn.Sequential(
                nn.Conv1d(
                    mid_channels,
                    mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm1d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    mid_channels,
                    mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm1d(mid_channels),
                nn.ReLU(inplace=True),
            )
        self.with_conv_refine = with_conv_refine

        self.mid_channels = mid_channels
        self.pe_scale = pe_scale

        self.h_pe_preCompute = None
        self.w_pe_preCompute = None


    def forward(self, x, pe=None, ret_attn=False):
        H, W = x.shape[-2:]

        if pe is not None:
            x = self.merger(torch.cat([x, pe], 1))
        else:
            x = self.merger(x)
        if self.reduce == 'max':
                x_width = x.max(2)[0].permute(0, 2, 1)
        elif self.reduce == 'mean':
                x_width = x.mean(2).permute(0, 2, 1)
        else:
            raise NotImplementedError

        if self.dummy:
            if self.out_proj is not None:
                x_width = self.out_proj(x_width)
            return x_width


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

        if self.with_conv_refine:
            x_width = x_width.permute(0, 2, 1)
            x_width = x_width + self.conv1(x_width)
            x_width = x_width + self.conv2(x_width)
            x_width = x_width + self.conv3(x_width)
            x_width = x_width.permute(0, 2, 1)


        for idx, blk in enumerate(self.attn_layers):
            x_width, cross_attn_mat = blk(x_width, x, self_pe=w_pe, cross_pe=h_pe)
        if self.out_proj is not None:
            x_width = self.out_proj(x_width)

        if ret_attn:
            return x_width, cross_attn_mat
        else:
            return x_width