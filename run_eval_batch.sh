#!/bin/bash

# SLURM submission script for batch model evaluation
# Submit with: sbatch run_eval_batch.sh

#SBATCH --job-name=eval-pvqd-batch
#SBATCH --account=project_2011211
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=========================================="
echo "SLURM Batch Evaluation Job"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "Nodes: ${SLURM_JOB_NUM_NODES}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Load environment modules
echo "Loading environment modules..."
module load gcc/11.2.0 cuda/11.5.0

# Activate virtual environment
echo "Activating Python environment..."
export PATH="/projappl/project_2014832/fang_env/salmonn/bin:$PATH"

# Print environment info
echo ""
echo "Environment Info:"
echo "  Python: $(python --version)"
echo "  CUDA: $(nvcc --version | grep release)"
echo ""

# Show GPU info
echo "GPU Info:"
nvidia-smi
echo ""

# Run batch evaluation
echo "=========================================="
echo "Starting batch evaluation..."
echo "=========================================="
bash batch_eval.sh

echo ""
echo "=========================================="
echo "Job completed at $(date)"
echo "=========================================="

