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


from ...amp.checkpoint import checkpoint
from .build import build_selfAttention_layer, build_crossAttention_layer, SELF_ATTENTION, CROSS_ATTENTION

scale_factor = 8
expand = scale_factor ** 2


class CFFN(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=1024,
                 num_fcs=2,
                 add_identity=True,
                 init_cfg=None,
                 **kwargs):
        super(CFFN, self).__init__(init_cfg)
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
    
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class FullAttnModule(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 out_dim=None,
                 qkv_bias=True,
                 attn_drop=0.,
                 proj_drop=0.,
                 q_extra_dim=0,
                 k_extra_dim=0,
                 v_extra_dim=0,
                 attn_act='softmax'):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim+q_extra_dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim+k_extra_dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim+v_extra_dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.embed_dims = dim

        self.attn_act = attn_act

    def forward(self, query, key, value, att_bias=None, ret_attn=False):
        bq, nq, cq = query.shape
        bk, nk, ck = key.shape
        bv, nv, cv = value.shape

        q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=bq, n=nq, c=self.embed_dims // self.num_heads)
        k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=bk, n=nk, c=self.embed_dims // self.num_heads)
        v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=bv, n=nv, c=self.embed_dims // self.num_heads)

        # [B, nh, N, S]
        attn = (q @ k.transpose(-2, -1)) * self.scale
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
        assert attn.shape == (bq, self.num_heads, nq, nk)

        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=bq, n=nq, c=cv // self.num_heads)
        out = self.proj(out)
        if self.training:
            out = self.proj_drop(out)
        if not ret_attn:
            return out
        else:
            return out, attn


class SemanticAugSingleHeadAttnModule(nn.Module):
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
                 semantic_detach=False):
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

    def forward(self, query, key, value, att_bias=None, ret_attn=False):
        bq, nq, cq = query.shape
        bk, nk, ck = key.shape
        bv, nv, cv = value.shape

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

class CompressFullAttnModule(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 out_dim=None,
                 qkv_bias=True,
                 attn_drop=0.,
                 proj_drop=0.,
                 qk_compress=1,
                 scale='v1'): # v1: scaling using dim before compress, v2: scaling using dim after compress
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if scale == 'v1':
            self.scale = head_dim**-0.5
        elif scale == 'v2':
            self.scale = (head_dim / qk_compress)**-0.5
        else:
            raise NotImplementedError

        qk_dim = dim // qk_compress
        self.qk_dim = qk_dim

        self.q_proj = nn.Linear(dim, qk_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, qk_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value, att_bias=None):
        bq, nq, cq = query.shape
        bk, nk, ck = key.shape
        bv, nv, cv = value.shape

        q = self.q_proj(query)
        k = self.k_proj(key)

        q = rearrange(q, 'b n (h c)-> b h n c', h=self.num_heads, b=bq, n=nq, c=self.qk_dim // self.num_heads)
        k = rearrange(k, 'b n (h c)-> b h n c', h=self.num_heads, b=bk, n=nk, c=self.qk_dim // self.num_heads)
        v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=bv, n=nv, c=cv // self.num_heads)

        # [B, nh, N, S]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if att_bias is not None:
            attn = attn + att_bias.unsqueeze(dim=1)

        attn = attn.softmax(dim=-1)
        if self.training:
            attn = self.attn_drop(attn)
        assert attn.shape == (bq, self.num_heads, nq, nk)

        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=bq, n=nq, c=cv // self.num_heads)
        out = self.proj(out)
        if self.training:
            out = self.proj_drop(out)
        return out

@CROSS_ATTENTION.register_module()
class BasicCrossAttnLayer(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads,
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
                 final_dw_conv_cfg=dict(enable=False, non_linear=False),
                 light_weight_cfg=dict(enable=False, qk_compress=4, scale='v1'),
                 attn_act='softmax',
                 residual=True,
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

        self.residual = residual

        if light_weight_cfg['enable']:
            self.attn = CompressFullAttnModule(
                embed_dims,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                qk_compress=light_weight_cfg["qk_compress"],
                scale=light_weight_cfg["scale"]
            )
        else:
            self.attn = FullAttnModule(
                embed_dims,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                attn_act=attn_act)

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
                cffn_version = _ffn_cfg.get('cffn_version', 'v1')
                if cffn_version == 'v1':
                    self.ffn = CFFN(**_ffn_cfgs_custom)
                else:
                    raise NotImplementedError
                self.ffn_norm = None
        else:
            self.ffn, self.ffn_norm = None, None
        self.use_cffn = _ffn_cfg['use_cffn']

        self.register_buffer('hit_record', torch.zeros((128, 128, 6)))
        self.register_buffer('N_frame', torch.zeros((1,)))

    def forward(self, query, key, value, query_pos_embed, key_pos_embed, reshape_back):
        def _inner_forward(query, key, value, query_pos_embed, key_pos_embed):
            assert len(query.shape) == 4
            assert len(key.shape) == 5
            assert len(value.shape) == 5
            if query_pos_embed is not None:
                assert len(query_pos_embed.shape) == 4
                _query = query + query_pos_embed
            else:
                _query = query
            if key_pos_embed is not None:
                assert len(key_pos_embed.shape) == 5
                _key = key + key_pos_embed
            else:
                _key = key
            _value = value

            B, C, qH, qW = _query.shape
            B, N, _, kH, kW = _key.shape

            _query = _query.reshape(B, C, -1).permute(0, 2, 1)
            _key = _key.permute(0, 1, 3, 4, 2).reshape(B, N * kH * kW, C)
            _value = _value.permute(0, 1, 3, 4, 2).reshape(B, N * kH * kW, C)

            if self.norm_query is not None:
                q = self.norm_query(_query)
            else:
                q = _query
            if self.norm_key is not None:
                k = self.norm_key(_key)
            else:
                k = _key
            if self.norm_value is not None:
                v = self.norm_value(_value)
            else:
                v = _value

            if self.training:
                if self.residual:
                    _query = _query + self.drop_path(self.attn(q, k, v))
                else:
                    _query = self.drop_path(self.attn(q, k, v))
            else:
                if self.residual:
                    _query = _query + self.attn(q, k, v)
                else:
                    _query = self.attn(q, k, v)

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

    def blockwise_forward(self, query, key, value, query_pos_embed, key_pos_embed, reshape_back):
        assert len(query.shape) == 4
        assert len(key.shape) == 5
        assert len(value.shape) == 5

        # print(query.shape, key.shape, value.shape)
        # exit()

        if query_pos_embed is not None:
            assert len(query_pos_embed.shape) == 4
            _query = query + query_pos_embed
        else:
            _query = query
        if key_pos_embed is not None:
            assert len(key_pos_embed.shape) == 5
            _key = key + key_pos_embed
        else:
            _key = key
        _value = value

        B, C, qH, qW = _query.shape
        B, N, _, kH, kW = _key.shape

        _query = _query.reshape(B, C, -1).permute(0, 2, 1)
        _key = _key.permute(0, 1, 3, 4, 2).reshape(B, N * kH * kW, C)
        _value = _value.permute(0, 1, 3, 4, 2).reshape(B, N * kH * kW, C)

        if self.norm_query is not None:
            q = self.norm_query(_query)
        else:
            q = _query
        if self.norm_key is not None:
            k = self.norm_key(_key)
        else:
            k = _key
        if self.norm_value is not None:
            v = self.norm_value(_value)
        else:
            v = _value

        _attn_ret, _attn = self.attn(q, k, v, ret_attn=True)
        _attn = _attn[0,0].reshape(128, 128, 6, 16, 44)
        _attn = _attn.sum(dim=(3,4))
        self.hit_record += _attn
        self.N_frame += 1

        if self.N_frame > 750:
            # print(self.hit_record / self.N_frame)
            hit_record = self.hit_record / self.N_frame
            hit_record = hit_record.detach().cpu().data.numpy()

            import numpy as np
            import seaborn as sns
            import matplotlib.pyplot as plt

            plt.figure(figsize=(35, 5))

            plt.subplot(1, 6, 1)
            sns.heatmap(hit_record[:, :, 0])
            plt.xticks([64], ['X'])
            plt.yticks([64], ['Y'])
            plt.title('FRONT_LEFT')

            plt.subplot(1, 6, 2)
            sns.heatmap(hit_record[:, :, 1])
            plt.xticks([64], ['X'])
            plt.yticks([64], ['Y'])
            plt.title('FRONT')

            plt.subplot(1, 6, 3)
            sns.heatmap(hit_record[:, :, 2])
            plt.xticks([64], ['X'])
            plt.yticks([64], ['Y'])
            plt.title('FRONT_RIGHT')

            plt.subplot(1, 6, 4)
            sns.heatmap(hit_record[:, :, 3])
            plt.xticks([64], ['X'])
            plt.yticks([64], ['Y'])
            plt.title('BACK_LEFT')

            plt.subplot(1, 6, 5)
            sns.heatmap(hit_record[:, :, 4])
            plt.xticks([64], ['X'])
            plt.yticks([64], ['Y'])
            plt.title('BACK')

            plt.subplot(1, 6, 6)
            sns.heatmap(hit_record[:, :, 5])
            plt.xticks([64],['X'])
            plt.yticks([64],['Y'])
            plt.title('BACK_RIGHT')

            plt.savefig('/home/s2139448/multiview_hit.png')
            exit()
        _query = _query + _attn_ret
        if self.ffn is not None:
            _query = self.ffn(self.ffn_norm(_query), identity=_query)
        if reshape_back:
            _query = _query.permute(0, 2, 1).reshape(B, C, qH, qW)
            if self.dw is not None:
                _query = self.dw(_query)
        else:
            if self.dw is not None:
                _query = _query.permute(0, 2, 1).reshape(B, C, qH, qW)
                _query = self.dw(_query)
                _query = _query.reshape(B, C, -1).permute(0, 2, 1)
        return _query


class BasicSingleHeadSemanticAugAttnLayer(nn.Module):
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

        self.attn = SemanticAugSingleHeadAttnModule(
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
                cffn_version = _ffn_cfg.get('cffn_version', 'v1')
                if cffn_version == 'v1':
                    self.ffn = CFFN(**_ffn_cfgs_custom)
                else:
                    raise NotImplementedError
                self.ffn_norm = None
        else:
            self.ffn, self.ffn_norm = None, None
        self.use_cffn = _ffn_cfg['use_cffn']

        self.register_buffer('hit_record', torch.zeros((128, 128, 6)))
        self.register_buffer('N_frame', torch.zeros((1,)))

    def forward(self, query, key, value, query_pos_embed, key_pos_embed, reshape_back):
        def _inner_forward(query, key, value, query_pos_embed, key_pos_embed):
            assert len(query.shape) == 4
            assert len(key.shape) == 5
            assert len(value.shape) == 5
            if query_pos_embed is not None:
                assert len(query_pos_embed.shape) == 4
                _query = query + query_pos_embed
            else:
                _query = query
            if key_pos_embed is not None:
                assert len(key_pos_embed.shape) == 5
                _key = key + key_pos_embed
            else:
                _key = key
            _value = value

            B, C, qH, qW = _query.shape
            B, N, _, kH, kW = _key.shape

            _query = _query.reshape(B, C, -1).permute(0, 2, 1)
            _key = _key.permute(0, 1, 3, 4, 2).reshape(B, N * kH * kW, C)
            _value = _value.permute(0, 1, 3, 4, 2).reshape(B, N * kH * kW, C)

            if self.norm_query is not None:
                q = self.norm_query(_query)
            else:
                q = _query
            if self.norm_key is not None:
                k = self.norm_key(_key)
            else:
                k = _key
            if self.norm_value is not None:
                v = self.norm_value(_value)
            else:
                v = _value

            if self.training:
                _query = _query + self.drop_path(self.attn(q, k, v))
            else:
                _query = _query + self.attn(q, k, v)

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
class BasicWidthSingleHeadSemanticAugAttnLayer(BasicSingleHeadSemanticAugAttnLayer):
    def forward(self, query, key, value, query_pos_embed, key_pos_embed, reshape_back):
        def _inner_forward(query, key, value, query_pos_embed, key_pos_embed):
            assert len(query.shape) == 4
            assert len(key.shape) == 4
            assert len(value.shape) == 4

            if query_pos_embed is not None:
                assert len(query_pos_embed.shape) == 4
                _query = query + query_pos_embed
            else:
                _query = query
            if key_pos_embed is not None:
                assert len(key_pos_embed.shape) == 4
                _key = key + key_pos_embed
            else:
                _key = key
            _value = value

            B, C, qH, qW = _query.shape
            B, N, kN, _ = _key.shape

            _key = _key.reshape(B, N * kN, -1)
            _value = _value.reshape(B, N * kN, -1)

            _query = _query.reshape(B, C, -1).permute(0, 2, 1)

            if self.norm_query is not None:
                q = self.norm_query(_query)
            else:
                q = _query
            if self.norm_key is not None:
                k = self.norm_key(_key)
            else:
                k = _key
            if self.norm_value is not None:
                v = self.norm_value(_value)
            else:
                v = _value

            if self.training:
                _query = _query + self.drop_path(self.attn(q, k, v))
            else:
                _query = _query + self.attn(q, k, v)
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
class HorizontalCrossAttnLayer(BasicCrossAttnLayer):
    def __init__(self, pool='mean', **kwargs):
        super(HorizontalCrossAttnLayer, self).__init__(**kwargs)
        self.pool = pool

    def forward(self, query, key, value, query_pos_embed, key_pos_embed, reshape_back):
        def _inner_forward(query, key, value, query_pos_embed, key_pos_embed):
            assert len(query.shape) == 4
            assert len(key.shape) == 5
            assert len(value.shape) == 5

            # print(query.shape, key.shape, value.shape)
            # exit()

            if query_pos_embed is not None:
                assert len(query_pos_embed.shape) == 4
                _query = query + query_pos_embed
            else:
                _query = query
            if key_pos_embed is not None:
                assert len(key_pos_embed.shape) == 5
                _key = key + key_pos_embed
            else:
                _key = key
            _value = value

            B, C, qH, qW = _query.shape
            B, N, _, kH, kW = _key.shape

            if self.pool == 'mean':
                _key = _key.mean(dim=-2)
                _value = _value.mean(dim=-2)
            elif self.pool == 'max':
                _key = _key.max(dim=-2)[0]
                _value = _value.max(dim=-2)[0]
            else:
                raise NotImplementedError

            _query = _query.reshape(B, C, -1).permute(0, 2, 1)
            _key = _key.permute(0, 1, 3, 2).reshape(B, N * kW, C)
            _value = _value.permute(0, 1, 3, 2).reshape(B, N * kW, C)

            q = self.norm_query(_query)
            k = self.norm_key(_key)
            v = self.norm_value(_value)

            _query = _query + self.drop_path(self.attn(q, k, v))
            if self.ffn is not None:
                _query = self.ffn(self.ffn_norm(_query), identity=_query)
            if reshape_back:
                _query = _query.permute(0, 2, 1).reshape(B, C, qH, qW)
            return _query
        if self.with_cp:
            return checkpoint(_inner_forward, query, key, value, query_pos_embed, key_pos_embed)
        else:
            return _inner_forward(query, key, value, query_pos_embed, key_pos_embed)


class BEVFeatureGenerateModule(nn.Module):
    def __init__(self,
                 embed_dims,
                 attn_cfgs,
                 ffn_cfg=None,
                 ffn_norm_cfg=dict(type='LN'),
                 resolution=None,
                 with_cp=False,
                 **kwargs):
        super().__init__()
        self.with_cp = with_cp

        if resolution is not None:
            assert type(resolution) is tuple
        self.resolution = resolution

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

        if ffn_cfg is not None:
            _ffn_cfgs = {
                'embed_dims': embed_dims,
                'feedforward_channels': ffn_cfg.get('ffn_dim'),
                'num_fcs': ffn_cfg.get('num_fcs', 2),
                'ffn_drop': ffn_cfg.get('drop', 0),
                'dropout_layer': ffn_cfg.get('drop_path', 0),
                'act_cfg': ffn_cfg.get('act_cfg', dict(type='GELU'),),
            }
            self.ffn = FFN(**_ffn_cfgs)
            assert ffn_norm_cfg is not None
            self.ffn_norm = build_norm_layer(ffn_norm_cfg, embed_dims)[1]
        else:
            self.ffn, self.ffn_norm = None, None

    def get_resolution(self):
        return tuple(self.resolution)

    def forward(self, bev_query, img_feats_key, img_feats_value):
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
                    x = layer(x, img_feats_key, img_feats_value, None, None, reshape_back=True)

            if self.ffn is not None:
                x = x.reshape(B, C, -1).permute(0, 2, 1)
                x = self.ffn(self.ffn_norm(x), identity=x)
                x = x.permute(0, 2, 1).reshape(B, C, H, W)

            assert len(x.shape) == 4
            return x

        if self.with_cp:
            return checkpoint(_inner_forward, bev_query, img_feats_key, img_feats_value)
        else:
            return _inner_forward(bev_query, img_feats_key, img_feats_value)

