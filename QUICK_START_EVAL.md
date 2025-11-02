# Quick Start: Evaluate Multiple Models Remotely

## TL;DR - Quick Commands

### Evaluate Single Model
```bash
python eval_multi_task.py \
    --cfg-path configs/config_pvqd.yaml \
    --ckpt-path /path/to/checkpoint.pth \
    --device cuda:0
```

### Evaluate Multiple Models (Batch)
```bash
# 1. Edit batch_eval.sh and add your checkpoint paths
nano batch_eval.sh

# 2. Run locally
bash batch_eval.sh

# 3. Or submit to SLURM
sbatch run_eval_batch.sh

# 4. Compare results
python compare_batch_results.py eval_results_batch_YYYYMMDD_HHMMSS
```

---

## Step-by-Step Guide

### Step 1: Find Your Checkpoints

```bash
# List all available checkpoints
find output_dir_breathiness -name "checkpoint_*.pth" -type f

# Example output:
# output_dir_breathiness/20241030_120000/checkpoint_best.pth
# output_dir_breathiness/20241030_120000/checkpoint_0.pth
# output_dir_breathiness/20241030_120000/checkpoint_1.pth
# output_dir_breathiness/20241030_120000/checkpoint_2.pth
```

### Step 2: Test Single Checkpoint (Quick Verification)

```bash
python eval_multi_task.py \
    --cfg-path configs/config_pvqd.yaml \
    --ckpt-path output_dir_breathiness/20241030_120000/checkpoint_best.pth \
    --device cuda:0
```

**Expected Output:**
```
Loading model...
Loading checkpoint from: output_dir_breathiness/20241030_120000/checkpoint_best.pth
============================================================
Evaluating Task: breathiness
============================================================
Total test samples: 30
Evaluating breathiness: 100%|████████| 30/30 [00:45<00:00,  1.50s/it]

Successfully evaluated: 30/30 samples
Failed samples: 0

Metrics for breathiness:
  RMSE: 12.3456
  MAE:  10.1234

============================================================
EVALUATION SUMMARY
============================================================
breathiness     - RMSE: 12.3456, MAE: 10.1234
roughness       - RMSE: 15.6789, MAE: 12.3456
strain          - RMSE: 14.2345, MAE: 11.5678

Average RMSE across all tasks: 14.0863

Results saved to eval_results_multi_task.json
```

### Step 3: Prepare Batch Evaluation

Edit `batch_eval.sh` and update the CHECKPOINTS list:

```bash
# Open the file
nano batch_eval.sh

# Find this section and update with your checkpoints:
CHECKPOINTS=(
    "output_dir_breathiness/20241030_120000/checkpoint_best.pth"
    "output_dir_breathiness/20241030_120000/checkpoint_0.pth"
    "output_dir_breathiness/20241030_120000/checkpoint_1.pth"
    "output_dir_breathiness/20241030_120000/checkpoint_2.pth"
)
```

### Step 4: Run Batch Evaluation

#### Option A: Run Locally
```bash
bash batch_eval.sh
```

#### Option B: Submit to SLURM (Recommended for Remote)
```bash
# Make scripts executable
chmod +x batch_eval.sh run_eval_batch.sh

# Submit job
sbatch run_eval_batch.sh

# Check status
squeue -u $USER

# View output in real-time
tail -f logs/eval-pvqd-batch-JOBID.out
```

### Step 5: Analyze Results

After batch evaluation completes:

```bash
# Compare all checkpoints
python compare_batch_results.py eval_results_batch_YYYYMMDD_HHMMSS

# Analyze specific checkpoint
python analyze_eval_results.py eval_results_batch_YYYYMMDD_HHMMSS/eval_results_checkpoint_best.json

# Compare two checkpoints
python analyze_eval_results.py \
    eval_results_batch_YYYYMMDD_HHMMSS/eval_results_checkpoint_best.json \
    --compare eval_results_batch_YYYYMMDD_HHMMSS/eval_results_checkpoint_0.json
```

---

## Remote Execution (Mahti Cluster)

### Complete Workflow

```bash
# 1. SSH to cluster
ssh user@mahti.csc.fi

# 2. Navigate to project
cd /scratch/project_2011211/fang/SALMONN

# 3. Find checkpoints
find output_dir_breathiness -name "checkpoint_*.pth" | head -5

# 4. Edit batch_eval.sh with your checkpoints
nano batch_eval.sh

# 5. Submit batch job
sbatch run_eval_batch.sh

# 6. Monitor job
watch squeue -u $USER

# 7. Check output
tail -f logs/eval-pvqd-batch-JOBID.out

# 8. After completion, analyze results
python compare_batch_results.py eval_results_batch_YYYYMMDD_HHMMSS

# 9. Download results locally
scp -r user@mahti.csc.fi:/scratch/project_2011211/fang/SALMONN/eval_results_batch_* .
```

---

## Understanding the Output

### eval_results_batch_YYYYMMDD_HHMMSS/
```
├── eval_results_checkpoint_best.json    # Results for best checkpoint
├── eval_log_checkpoint_best.txt         # Detailed log
├── eval_results_checkpoint_0.json       # Results for epoch 0
├── eval_log_checkpoint_0.txt
├── eval_results_checkpoint_1.json       # Results for epoch 1
├── eval_log_checkpoint_1.txt
└── ...
```

### JSON Result Format
```json
{
  "breathiness": {
    "task": "breathiness",
    "rmse": 12.3456,
    "mae": 10.1234,
    "num_samples": 30,
    "failed_samples": 0,
    "predictions": [45.2, 67.8, ...],
    "ground_truths": [50.0, 65.0, ...]
  },
  "roughness": { ... },
  "strain": { ... }
}
```

---

## Troubleshooting

### Problem: "Checkpoint not found"
```bash
# Verify checkpoint exists
ls -lh /path/to/checkpoint.pth

# Check full path
file /path/to/checkpoint.pth
```

### Problem: Job timeout
Edit `run_eval_batch.sh`:
```bash
#SBATCH --time=08:00:00  # Increase from 04:00:00
```

### Problem: Out of memory
Edit `batch_eval.sh` to add options:
```bash
python eval_multi_task.py \
    --cfg-path "$CONFIG" \
    --ckpt-path "$ckpt" \
    --device "$DEVICE" \
    --options run.batch_size_eval 4 run.num_workers 2
```

### Problem: Can't find results
```bash
# List all result directories
ls -d eval_results_batch_*

# Check specific directory
ls -lh eval_results_batch_YYYYMMDD_HHMMSS/
```

---

## Performance Benchmarks

| Metric | Value |
|--------|-------|
| Time per checkpoint | ~2-5 minutes |
| GPU memory | ~20-25GB |
| Samples per task | 30 |
| Total samples | 90 |

---

## Key Files

| File | Purpose |
|------|---------|
| `eval_multi_task.py` | Main evaluation script |
| `batch_eval.sh` | Batch evaluation wrapper |
| `run_eval_batch.sh` | SLURM submission script |
| `compare_batch_results.py` | Results comparison tool |
| `analyze_eval_results.py` | Detailed analysis tool |

---

## Next Steps

1. ✅ Find your checkpoints
2. ✅ Edit batch_eval.sh
3. ✅ Submit batch job
4. ✅ Monitor progress
5. ✅ Analyze results
6. ✅ Compare checkpoints
7. ✅ Download results

Ready to evaluate! 🚀

