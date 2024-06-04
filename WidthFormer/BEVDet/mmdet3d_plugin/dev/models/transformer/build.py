from mmcv.utils import Registry, build_from_cfg
from torch import nn
import warnings


SELF_ATTENTION = Registry('self_attn_layer')
CROSS_ATTENTION = Registry('cross_attn_layer')


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)

def build_selfAttention_layer(cfg):
    """Build distill loss."""
    return build(cfg, SELF_ATTENTION)

def build_crossAttention_layer(cfg):
    """Build distill loss."""
    return build(cfg, CROSS_ATTENTION)
