"""
    Config file for OpenLane/Lane3D data
"""
# 1. USE CULANE DATASET (It reads the .lines.txt files)
dataset_type = 'CulaneDataset' 

# 2. YOUR DATA ROOT
data_root = "/scratch/engin_root/engin1/aphatke/conditional-lane-detection"

# 3. SET YOUR IMAGE RESOLUTION
ori_scale = (1920, 1280) 

# 4. CROP SETTINGS
crop_bbox = [0, 270, 1920, 1280]

# Standard settings

mask_down_scale = 8
hm_down_scale = 16
line_width = 3
radius = 6
nms_thr = 4
batch_size = 1  # Testing uses batch size 1
mask_size = (1, 68, 100)


img_scale = (800, 544) 

img_norm_cfg = dict(
    mean=[75.3, 76.6, 77.6], std=[50.5, 53.8, 54.3], to_rgb=False)


train_cfg = dict(out_scale=mask_down_scale)
test_cfg = dict(out_scale=mask_down_scale)


model = dict(
    type='CondLaneNet',  # Use standard CondLaneNet for stability
    pretrained='torchvision://resnet18',
    train_cfg=train_cfg,
    test_cfg=test_cfg,
    num_classes=1,
    backbone=dict(
        type='ResNet',
        depth=18,
        strides=(1, 2, 2, 2),
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),

        neck=dict(
        type='TransConvFPN',
        in_channels=[128, 256, 64],
        out_channels=64,
        num_outs=3,
        trans_idx=-1,
        trans_cfg=dict(
            in_dim=512,
            attn_in_dims=[512, 64],
            attn_out_dims=[64, 64],
            strides=[1, 1],
            ratios=[4, 4],
            pos_shape=(1, 17, 25),  # Use batch_size=1 for testing
        ),
    ),
    head=dict(
        type='CondLaneHead',
        heads=dict(hm=1),
        in_channels=(64, ),
        num_classes=1,
        head_channels=64,
        head_layers=1,
        disable_coords=False,
        branch_in_channels=64,
        branch_channels=64,
        branch_out_channels=64,
        reg_branch_channels=64,
        branch_num_conv=1,
        hm_idx=1,
        mask_idx=0,
        compute_locations_pre=True,
        location_configs=dict(size=(1, 1, 68, 100), device='cuda:0')),  # Use batch_size=1 for testing
    loss_weights=dict(
        hm_weight=1,
        kps_weight=0.4,
        row_weight=1.,
        range_weight=1.,
    ),
)
train_compose = dict(bboxes=False, keypoints=True, masks=False)


train_al_pipeline = [
    dict(type='Compose', params=train_compose),
    dict(
        type='Crop',
        x_min=crop_bbox[0],
        x_max=crop_bbox[2],
        y_min=crop_bbox[1],
        y_max=crop_bbox[3],
        p=1),
    dict(type='Resize', height=img_scale[1], width=img_scale[0], p=1),
    dict(type='RandomBrightness', limit=0.2, p=0.6),
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.1,
        scale_limit=(-0.2, 0.2),
        rotate_limit=10,
        border_mode=0,
        p=0.6),
]

val_al_pipeline = [
    dict(type='Compose', params=train_compose),
    dict(
        type='Crop',
        x_min=crop_bbox[0],
        x_max=crop_bbox[2],
        y_min=crop_bbox[1],
        y_max=crop_bbox[3],
        p=1),
    dict(type='Resize', height=img_scale[1], width=img_scale[0], p=1),
]

train_pipeline = [
    dict(type='albumentation', pipelines=train_al_pipeline),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='CollectLane', # Must match CondLaneNet
        down_scale=mask_down_scale,
        hm_down_scale=hm_down_scale,
        max_mask_sample=5,
        line_width=line_width,
        radius=radius,
        keys=['img', 'gt_hm'],
        meta_keys=[
            'filename', 'sub_img_name', 'gt_masks', 'mask_shape', 'hm_shape',
            'ori_shape', 'img_shape', 'down_scale', 'hm_down_scale',
            'img_norm_cfg', 'gt_points'
        ]),
]

val_pipeline = [
    dict(type='albumentation', pipelines=val_al_pipeline),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='CollectLane',
        down_scale=mask_down_scale,
        hm_down_scale=hm_down_scale,
        radius=radius,
        keys=['img', 'gt_hm'],
        meta_keys=[
            'filename', 'sub_img_name', 'gt_masks', 'mask_shape', 'hm_shape',
            'ori_shape', 'img_shape', 'down_scale', 'hm_down_scale',
            'img_norm_cfg', 'gt_points'
        ]),
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        data_list=data_root + '/images/list/training_list_small.txt', # Point to your list
        pipeline=train_pipeline,
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        data_list=data_root + '/images/list/validation_list.txt', # Point to your list
        pipeline=val_pipeline,
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        # Option 1: Use validation_list.txt (3905 images) - currently active
        data_list=data_root + '/images/list/validation_list.txt',
        # Option 2: Use combined_test_list.txt (intersection + training samples) - run create_combined_test_list.py first
        # data_list=data_root + '/images/list/combined_test_list.txt',
        test_suffix='.jpg',
        pipeline=val_pipeline,
        test_mode=True,
    ))

optimizer = dict(type='Adam', lr=3e-4, betas=(0.9, 0.999), eps=1e-8)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    step=[8, 14])

checkpoint_config = dict(interval=1)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

total_epochs = 1
device_ids = [0]
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/openlane/small2'
load_from = None
resume_from = None
workflow = [('train', 200), ('val', 1)]
find_unused_parameters = True # IDK WHY WE NEED THIS

