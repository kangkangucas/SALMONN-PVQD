# Remote Evaluation Guide - Testing Multiple Models

## Overview
This guide explains how to evaluate different trained model checkpoints on the test data remotely using the updated `eval_multi_task.py` script.

## Key Feature: Model Checkpoint as Command-Line Argument

The evaluation script now accepts `--ckpt-path` argument to specify different model checkpoints without modifying the config file.

## Usage

### Basic Usage (Using Config Checkpoint)
```bash
python eval_multi_task.py --cfg-path configs/config_pvqd.yaml --device cuda:0
```

### With Custom Checkpoint (Recommended for Testing Multiple Models)
```bash
python eval_multi_task.py \
    --cfg-path configs/config_pvqd.yaml \
    --ckpt-path /path/to/your/checkpoint.pth \
    --device cuda:0
```

## Remote Execution Steps

### Step 1: Prepare Your Checkpoints

First, identify all the model checkpoints you want to evaluate. They are typically saved in:
```
output_dir_breathiness/TIMESTAMP/checkpoint_best.pth
output_dir_breathiness/TIMESTAMP/checkpoint_0.pth
output_dir_breathiness/TIMESTAMP/checkpoint_1.pth
...
```

### Step 2: Create a Batch Evaluation Script

Create a file `batch_eval.sh` to evaluate multiple models:

```bash
#!/bin/bash

# Batch evaluation script for multiple models

CONFIG="configs/config_pvqd.yaml"
DEVICE="cuda:0"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="eval_results_batch_${TIMESTAMP}"

mkdir -p "$RESULTS_DIR"

# List of checkpoints to evaluate
CHECKPOINTS=(
    "output_dir_breathiness/20241030_120000/checkpoint_best.pth"
    "output_dir_breathiness/20241030_120000/checkpoint_0.pth"
    "output_dir_breathiness/20241030_120000/checkpoint_1.pth"
    "output_dir_breathiness/20241030_120000/checkpoint_2.pth"
)

echo "Starting batch evaluation at $(date)"
echo "Results will be saved to: $RESULTS_DIR"
echo "=========================================="

for ckpt in "${CHECKPOINTS[@]}"; do
    if [ ! -f "$ckpt" ]; then
        echo "Warning: Checkpoint not found: $ckpt"
        continue
    fi
    
    # Extract checkpoint name for output file
    ckpt_name=$(basename "$ckpt" .pth)
    output_file="$RESULTS_DIR/eval_results_${ckpt_name}.json"
    log_file="$RESULTS_DIR/eval_log_${ckpt_name}.txt"
    
    echo ""
    echo "=========================================="
    echo "Evaluating: $ckpt"
    echo "Output: $output_file"
    echo "=========================================="
    
    # Run evaluation
    python eval_multi_task.py \
        --cfg-path "$CONFIG" \
        --ckpt-path "$ckpt" \
        --device "$DEVICE" \
        2>&1 | tee "$log_file"
    
    # Move results to batch directory
    if [ -f "eval_results_multi_task.json" ]; then
        mv eval_results_multi_task.json "$output_file"
    fi
    
    echo "Completed: $ckpt_name"
done

echo ""
echo "=========================================="
echo "Batch evaluation completed at $(date)"
echo "Results saved to: $RESULTS_DIR"
echo "=========================================="
```

### Step 3: Submit to Remote Cluster (SLURM)

Create `run_eval_batch.sh` for SLURM submission:

```bash
#!/bin/bash
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

# Load environment
module load gcc/11.2.0 cuda/11.5.0
export PATH="/projappl/project_2014832/fang_env/salmonn/bin:$PATH"

# Print info
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
nvidia-smi
python -V

# Run batch evaluation
bash batch_eval.sh
```

### Step 4: Submit the Job

```bash
# Make scripts executable
chmod +x batch_eval.sh run_eval_batch.sh

# Submit to SLURM
sbatch run_eval_batch.sh

# Check job status
squeue -u $USER

# View output
tail -f logs/eval-pvqd-batch-JOBID.out
```

## Single Model Evaluation (Quick Test)

For testing a single model quickly:

```bash
python eval_multi_task.py \
    --cfg-path configs/config_pvqd.yaml \
    --ckpt-path output_dir_breathiness/20241030_120000/checkpoint_best.pth \
    --device cuda:0
```

## Comparing Results

After evaluation, compare results from different checkpoints:

```bash
# Analyze single checkpoint
python analyze_eval_results.py eval_results_multi_task.json

# Compare two checkpoints
python analyze_eval_results.py eval_results_checkpoint_best.json \
    --compare eval_results_checkpoint_0.json
```

## Finding Your Checkpoints

### List all available checkpoints:
```bash
find output_dir_breathiness -name "checkpoint_*.pth" -type f
```

### Get checkpoint details:
```bash
ls -lh output_dir_breathiness/*/checkpoint_*.pth
```

### Check checkpoint creation time:
```bash
stat output_dir_breathiness/TIMESTAMP/checkpoint_best.pth
```

## Output Structure

After batch evaluation, you'll have:
```
eval_results_batch_YYYYMMDD_HHMMSS/
├── eval_results_checkpoint_best.json
├── eval_log_checkpoint_best.txt
├── eval_results_checkpoint_0.json
├── eval_log_checkpoint_0.txt
├── eval_results_checkpoint_1.json
├── eval_log_checkpoint_1.txt
└── ...
```

## Analyzing Batch Results

Create a comparison script `compare_batch_results.py`:

```python
import json
import os
from pathlib import Path
import numpy as np

results_dir = "eval_results_batch_YYYYMMDD_HHMMSS"
results_files = sorted(Path(results_dir).glob("eval_results_*.json"))

print("\n" + "="*70)
print("BATCH EVALUATION COMPARISON")
print("="*70)

all_results = {}
for results_file in results_files:
    with open(results_file, 'r') as f:
        results = json.load(f)
        ckpt_name = results_file.stem.replace("eval_results_", "")
        all_results[ckpt_name] = results

# Compare RMSE across checkpoints
print("\nRMSE Comparison:")
print("-" * 70)
print(f"{'Checkpoint':<30} {'Breathiness':<15} {'Roughness':<15} {'Strain':<15}")
print("-" * 70)

for ckpt_name, results in sorted(all_results.items()):
    breathiness_rmse = results.get('breathiness', {}).get('rmse', 'N/A')
    roughness_rmse = results.get('roughness', {}).get('rmse', 'N/A')
    strain_rmse = results.get('strain', {}).get('rmse', 'N/A')
    
    print(f"{ckpt_name:<30} {breathiness_rmse:<15.4f} {roughness_rmse:<15.4f} {strain_rmse:<15.4f}")

print("="*70 + "\n")
```

## Troubleshooting

### Checkpoint not found
```bash
# Verify checkpoint exists
ls -lh /path/to/checkpoint.pth

# Check if path is correct
file /path/to/checkpoint.pth
```

### Out of memory during evaluation
```bash
# Use smaller batch size (modify config)
# Or reduce num_workers
python eval_multi_task.py \
    --cfg-path configs/config_pvqd.yaml \
    --ckpt-path checkpoint.pth \
    --device cuda:0 \
    --options run.batch_size_eval 4 run.num_workers 2
```

### Job timeout
Increase time in SLURM script:
```bash
#SBATCH --time=08:00:00  # Increase from 04:00:00
```

## Performance Tips

1. **Parallel Evaluation**: Submit multiple SLURM jobs for different checkpoints
2. **Batch Processing**: Use the batch script to evaluate all checkpoints in one job
3. **Monitor GPU**: Check GPU usage during evaluation
   ```bash
   watch -n 1 nvidia-smi
   ```

## Example Workflow

```bash
# 1. List all checkpoints
find output_dir_breathiness -name "checkpoint_*.pth" | head -5

# 2. Create batch_eval.sh with your checkpoints

# 3. Submit batch job
sbatch run_eval_batch.sh

# 4. Monitor progress
squeue -u $USER
tail -f logs/eval-pvqd-batch-JOBID.out

# 5. Analyze results
python analyze_eval_results.py eval_results_batch_YYYYMMDD_HHMMSS/eval_results_checkpoint_best.json

# 6. Compare checkpoints
python compare_batch_results.py
```

## Next Steps

1. Identify checkpoints to evaluate
2. Create batch_eval.sh with your checkpoint paths
3. Submit to SLURM cluster
4. Monitor and analyze results
5. Compare performance across models

