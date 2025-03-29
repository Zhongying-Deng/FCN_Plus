_base_ = './fcn_r50-d8_512x512_160k_ade20k.py'
model = dict(
    pretrained='open-mmlab://resnet101_v1c', 
    backbone=dict(
        depth=101,
        plugins=[
            dict(cfg=dict(type='GlobalShift', scale=4, portion=0.5),
                 stages=(True, True, True, True),
                 position='after_conv1'),]))

test_cfg = dict(mode='slide', crop_size=(512, 512), stride=(341, 341))
total_iters = 320000
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU')
