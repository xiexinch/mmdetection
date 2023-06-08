_base_ = [
    '_base_/xdecoder-tiny_open-vocab-panoptic.py',
    'mmdet::_base_/datasets/ade20k_panoptic.py'
]
model = dict()
backend_args = None
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='LoadPanopticAnnotations', backend_args=backend_args),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text'))
]

train_dataloader = None
val_dataloader = dict(
    dataset=dict(return_classes=True, pipeline=test_pipeline))
test_dataloader = val_dataloader

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
