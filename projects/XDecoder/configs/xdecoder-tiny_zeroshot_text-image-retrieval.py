_base_ = 'xdecoder-tiny_zeroshot_open-vocab-semseg.py'

model = dict(task='retrieval')

grounding_resize_scale = 512

test_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='pillow', backend_args=_base_.backend_args),
    dict(type='FixScaleResize',
         scale=224,
         keep_ratio=True,
         short_side_mode=True,
         backend='pillow',
         interpolation='bicubic'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text'))
]

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
