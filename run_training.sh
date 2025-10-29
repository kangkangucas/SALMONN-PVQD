#!/bin/bash
#SBATCH --job-name=train-pvqd
#SBATCH --account=project_2014832
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# # 1) 准备环境
# module purge
# 按你的环境需要加载模块（例如 gcc/cuda 等），下行仅示例：
module load gcc/11.2.0 cuda/11.5.0

# 2) 激活你的虚拟环境
export PATH="/projappl/project_2014832/fang_env/salmonn/bin:$PATH"

# 3) 打印信息（可选）
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
nvidia-smi
python -V

# 4) 单卡启动（两种二选一）

# 4a) 直接 python（建议你在 config 里 use_distributed=False）
# python -u train.py --cfg-path configs/config_pvqd.yaml

# 4b) 如果保留 use_distributed=True，用 torchrun 单卡启动：
torchrun --standalone --nproc_per_node=1 \
  train.py --cfg-path configs/config_pvqd.yaml