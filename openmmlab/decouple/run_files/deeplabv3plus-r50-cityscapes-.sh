GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=1 tools/slurm_train.sh mediaf1 mmseg_1 openmmlab/decouple/configs/deeplabv3plus_r50-d8_769x769_80k_cityscapes.py --work-dir='./openmmlab/decouple/work_dirs'  --seed=0

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=1 tools/slurm_train.sh mediaf1 mmseg_2 openmmlab/decouple/configs/deeplabv3plus_r50-d8_512x1024_40k_cityscapes.py --work-dir='./openmmlab/decouple/work_dirs'  --seed=0

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=1 tools/slurm_train.sh mediaf1 mmseg_3 openmmlab/decouple/configs/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py --work-dir='./openmmlab/decouple/work_dirs'  --seed=0

GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=1 tools/slurm_train.sh mediaf1 mmseg_4 openmmlab/decouple/configs/deeplabv3plus_r50-d8_769x769_40k_cityscapes.py --work-dir='./openmmlab/decouple/work_dirs'  --seed=0

