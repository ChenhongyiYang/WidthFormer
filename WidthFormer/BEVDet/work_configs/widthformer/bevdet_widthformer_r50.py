_base_ = ['../template/bevdet_r50_bs8.py']

data_config={
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test':0.04,
}

# Model
grid_config={
        'xbound': [-51.2, 51.2, 0.8], # +-51.2, x BEV plain range
        'ybound': [-51.2, 51.2, 0.8],  # +-51.2, y BEV plain range
        'zbound': [-10.0, 10.0, 20.0], # +-51.2, z BEV plain range
        'dbound': [1.0, 60.0, 1.0],} # +-51.2, depth BEV plain range

embed_dims=64
model = dict(
    type='BEVDetWidthSup',
    img_backbone=dict(with_cp=True),
    img_neck=dict(with_cp=True),
    img_view_transformer=dict(
        type='WidthFormer',
        input_dim=512,
        embed_dims=embed_dims,
        grid_config=grid_config,
        data_config=data_config,
        LID=False,
        downsample=16,
        multiview_positional_encoding=dict(
            type='SinePositionalEncoding2D',
            num_feats=embed_dims,
            normalize=True),
        bev_query_shape=(128, 128),
        positional_embedding_scale=dict(enable=True, type='linear', linear_scale=10),
        transformer_cfgs=[
            dict(
                resolution=(128, 128),
                attn_cfgs=[
                    dict(
                        type='BasicWidthSingleHeadSemanticAugAttnLayer',
                        attn_type='cross',
                        embed_dims=embed_dims,
                        key_normed=True,
                        value_normed=True,
                        attn_act='softmax',
                        ffn_cfg=dict(
                            ffn_dim=512,
                            use_cffn=True,
                            cffn_version='v1',
                        )
                    )
                ],
            )],
        refine_net_cfg=dict(
            num_self_head=1,
            num_cross_head=1,
            ffn_dim=embed_dims * 4,
            op_order='cross_first',
            num_attn_layers=1),
        norm_img_feats_key=True,
        norm_img_feats_value=True,
        positional_encoding='new',
        positional_noise='none',
        with_cp=False,
        return_width_feature=True),
    head_2d_cfg=dict(
        type='FCOSAuxFlattenHeadDev',
        num_classes=10,
        in_channels=embed_dims,
        feat_channels=256,
        stacked_convs=2,
        n_shared_convs=1,
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_delta=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_depth=dict(type='CrossEntropyLoss', loss_weight=1.0),
        loss_height=dict(type='CrossEntropyLoss', loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
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
        conv_bias=False,
        norm_cfg=dict(type='BN1d', requires_grad=True),
        train_cfg=dict(
            input_size=data_config['input_size'],
            down_stride=16,
            depth_bounds=grid_config['dbound']),
        data_config=data_config),
    img_bev_encoder_backbone=dict(numC_input=embed_dims, with_cp=False),
    img_bev_encoder_neck=dict(in_channels=embed_dims*8+embed_dims*2),
)

base_bs = 8
bs = 8
data = dict(
    samples_per_gpu=bs, # 8
    workers_per_gpu=2)
paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys={
        '.ego_position_mlp': dict(decay_mult=0.0),
        '.multiview_position_mlp': dict(decay_mult=0.0),
        '.bev_position_mlp': dict(decay_mult=0.0),
        '.width_depth_proj': dict(decay_mult=0.0)
    }
)
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=0.01, paramwise_cfg=paramwise_cfg)