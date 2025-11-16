#!/bin/bash
#SBATCH --job-name=train-pvqd
#SBATCH --account=project_2014832
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=25:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# 1) 加载必要的模块
echo "Loading modules..."
module load gcc/11.2.0 cuda/11.5.0


# Create logs directory if it doesn't exist
mkdir -p logs


# 2) 设置 CUDA 库路径 (关键!)
echo "Setting CUDA environment variables..."
export CUDA_HOME=/appl/spack/v017/install-tree/gcc-11.2.0/cuda-11.5.0-mg4ztb
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

# 3) 验证 CUDA 设置
echo "Verifying CUDA setup..."
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA version:"
nvcc --version | head -1
echo ""

# 4) 激活虚拟环境
echo "Activating virtual environment..."
export PATH="/projappl/project_2014832/fang_env/salmonn/bin:$PATH"

# 5) 验证 Python 和 PyTorch
echo "Verifying Python and PyTorch..."
python --version
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# 6) 显示 GPU 信息
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# 7) 禁用 bitsandbytes CUDA 检查 (可选)
export BNB_CUDA_VERSION=115

# 8) 设置 CUDA 可见设备
export CUDA_VISIBLE_DEVICES=0

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

# Change to working directory
cd /scratch/project_2014832/fang/code/SALMONN

# Run inference
python infer_pseudo_labels.py \
    --audio-dir /scratch/project_2014832/fang/data/DH-FaceVid-1K_extracted_wav \
    --output-csv /scratch/project_2014832/fang/data/pseudo_labels_dh_facevid_1k_202511131805.csv \
    --ckpt-path output_dir/202511131805/checkpoint_1.pth \
    --cfg-path configs/config_pvqd.yaml \
    --device cuda:0 \
    --batch-size 1

echo ""
echo "=========================================="
echo "Job completed!"
echo "=========================================="

