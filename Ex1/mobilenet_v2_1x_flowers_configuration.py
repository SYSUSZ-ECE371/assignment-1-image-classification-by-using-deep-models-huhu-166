
_base_ = [
    '../_base_/models/mobilenet_v2_1x.py',
    '../_base_/schedules/imagenet_bs256_epochstep.py',
    '../_base_/default_runtime.py'
]

# ---- model settings ----
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth',
            prefix='backbone')
    ),
    head=dict(num_classes=5)  # 修改为你的类别数量
)

# ---- data settings ----
dataset_type = 'ImageNet'
data_preprocessor = dict(
    mean=[124.508, 116.050, 106.438],
    std=[58.577, 57.310, 57.437],
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs'),
]

# 修改数据路径为花卉数据集路径
train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='/content/flower_dataset/flower_dataset/train',
        classes=['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'],
        pipeline=train_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='/content/flower_dataset/flower_dataset/val',
        classes=['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'],
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)







# ---- evaluation metric ----
val_evaluator = dict(type='Accuracy', topk=1)
test_evaluator = None
# ---- schedule settings ----
# 微调建议使用较小学习率
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),
    clip_grad=None,
)

param_scheduler = dict(type='StepLR', by_epoch=True, step_size=3, gamma=0.1)

train_cfg = dict(by_epoch=True, max_epochs=10, val_interval=1)
val_cfg = dict()
test_cfg = None
# ---- runtime settings ----
default_hooks = dict(logger=dict(interval=10))
randomness = dict(seed=0, deterministic=False)
