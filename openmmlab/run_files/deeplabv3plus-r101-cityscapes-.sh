GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=1 tools/slurm_train.sh mediaf mmseg1 configs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes.py --work-dir='./openmmlab/work_dirs_rerun'  --seed=0

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=1 tools/slurm_train.sh mediaf mmseg2 configs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_40k_cityscapes.py --work-dir='./openmmlab/work_dirs_rerun'  --seed=0

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=1 tools/slurm_train.sh mediaf mmseg3 configs/deeplabv3plus/deeplabv3plus_r101-d8_769x769_40k_cityscapes.py --work-dir='./openmmlab/work_dirs_rerun'  --seed=0  --options model.backbone.with_cp=True

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=1 tools/slurm_train.sh mediaf mmseg4 configs/deeplabv3plus/deeplabv3plus_r101-d8_769x769_80k_cityscapes.py --work-dir='./openmmlab/work_dirs_rerun'  --seed=0  --options model.backbone.with_cp=True

