MASTER_PORT=3622 GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=2 tools/slurm_train.sh mediaa mmseg_1 shift_exps/configs_ade/fcn_r101-plugs_shift_s4_p05_c1-d8_512x512-scale_2048_320-ratio_05_20-320k_ade20k.py --seed=0

MASTER_PORT=1246 GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=2 tools/slurm_train.sh mediaa mmseg_2 shift_exps/configs_ade/fcn_r101-plugs_shift_s4_p05_c1-d8_512x512-scale_1024_320-ratio_05_20-320k_ade20k.py --seed=0

# MASTER_PORT=3309 GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=2 tools/slurm_train.sh mediaa mmseg_3 shift_exps/configs_ade/fcn_r101-plugs_shift_s4_p05_c1-d8_512x512_320k_ade20k.py --seed=0

MASTER_PORT=3958 GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=2 tools/slurm_train.sh mediaa mmseg_4 shift_exps/configs_ade/fcn_r101-plugs_shift_s4_p05_c1-d8_512x512-scale_2048_384-ratio_05_20-320k_ade20k.py --seed=0

MASTER_PORT=4833 GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=2 tools/slurm_train.sh mediaa mmseg_5 shift_exps/configs_ade/fcn_r101-plugs_shift_s4_p05_c1-d8_512x512-scale_1024_384-ratio_05_20-320k_ade20k.py --seed=0

