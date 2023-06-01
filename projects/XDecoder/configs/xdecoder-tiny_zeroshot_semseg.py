_base_ = 'mmdet::_base_/default_runtime.py'

custom_imports = dict(
    imports=['projects.XDecoder.xdecoder'], allow_failed_imports=False)

model = dict(
    type='XDecoder',
    task='semseg',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(type='FocalNet'),
    semseg_head=dict(
        type='XDecoderOVSemSegHead',
        in_channels=(96, 192, 384, 768),
        num_classes=133,
        pixel_decoder=dict(type='TransformerEncoderPixelDecoder'),
        transformer_decoder=dict(type='XDecoderTransformerDecoder'),
    ),
    test_cfg=dict(mask_thr=0.5, keep_bg=True))

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='LoadSemSegAnnotations', reduce_zero_label=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor',
                   'seg_map_path', 'reduce_zero_label', 'seg_fields', 'img',
                   'gt_seg_map', 'keep_ratio', 'text'))
]

dataset_type = 'ADE20KDataset'
data_root = 'data/ade/ADEChallengeData2016'

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        return_classess=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='MIoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
