# Remote Evaluation Commands - Copy & Paste Ready

## 🚀 Quick Reference - All Commands You Need

### 1. SSH to Cluster
```bash
ssh user@mahti.csc.fi
cd /scratch/project_2011211/fang/SALMONN
```

### 2. Find Your Checkpoints
```bash
find output_dir_breathiness -name "checkpoint_*.pth" -type f
```

**Example Output:**
```
output_dir_breathiness/20241030_120000/checkpoint_best.pth
output_dir_breathiness/20241030_120000/checkpoint_0.pth
output_dir_breathiness/20241030_120000/checkpoint_1.pth
output_dir_breathiness/20241030_120000/checkpoint_2.pth
```

### 3. Test Single Checkpoint (Quick Verification)
```bash
python eval_multi_task.py \
    --cfg-path configs/config_pvqd.yaml \
    --ckpt-path output_dir_breathiness/20241030_120000/checkpoint_best.pth \
    --device cuda:0
```

### 4. Edit Batch Script
```bash
nano batch_eval.sh
```

**Find this section and update:**
```bash
CHECKPOINTS=(
    "output_dir_breathiness/20241030_120000/checkpoint_best.pth"
    "output_dir_breathiness/20241030_120000/checkpoint_0.pth"
    "output_dir_breathiness/20241030_120000/checkpoint_1.pth"
    "output_dir_breathiness/20241030_120000/checkpoint_2.pth"
)
```

**Save and exit:** Press `Ctrl+X`, then `Y`, then `Enter`

### 5. Submit Batch Job
```bash
sbatch run_eval_batch.sh
```

**Output:**
```
Submitted batch job 12345678
```

### 6. Monitor Job Status
```bash
# Check if job is running
squeue -u $USER

# View real-time output
tail -f logs/eval-pvqd-batch-12345678.out

# Watch continuously
watch squeue -u $USER
```

### 7. After Job Completes - Analyze Results
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

### 8. Download Results to Local Machine
```bash
# From your local machine (NOT on cluster)
scp -r user@mahti.csc.fi:/scratch/project_2011211/fang/SALMONN/eval_results_batch_* .
```

---

## 📋 Complete Workflow (Copy & Paste)

```bash
# 1. Connect to cluster
ssh user@mahti.csc.fi
cd /scratch/project_2011211/fang/SALMONN

# 2. Find checkpoints
find output_dir_breathiness -name "checkpoint_*.pth" -type f

# 3. Edit batch script (update CHECKPOINTS list)
nano batch_eval.sh

# 4. Submit job
sbatch run_eval_batch.sh

# 5. Get job ID from output, then monitor
squeue -u $USER
tail -f logs/eval-pvqd-batch-JOBID.out

# 6. After completion, analyze
python compare_batch_results.py eval_results_batch_YYYYMMDD_HHMMSS

# 7. Download results (from local machine)
scp -r user@mahti.csc.fi:/scratch/project_2011211/fang/SALMONN/eval_results_batch_* .
```

---

## 🔍 Useful Commands

### Check Job Status
```bash
# All your jobs
squeue -u $USER

# Specific job
squeue -j JOBID

# Detailed job info
scontrol show job JOBID
```

### View Logs
```bash
# Last 50 lines
tail -50 logs/eval-pvqd-batch-JOBID.out

# Real-time monitoring
tail -f logs/eval-pvqd-batch-JOBID.out

# Search for errors
grep -i error logs/eval-pvqd-batch-JOBID.out
```

### Cancel Job
```bash
scancel JOBID
```

### List Results
```bash
# All result directories
ls -d eval_results_batch_*

# Files in specific directory
ls -lh eval_results_batch_YYYYMMDD_HHMMSS/

# Count results
ls eval_results_batch_YYYYMMDD_HHMMSS/eval_results_*.json | wc -l
```

---

## 📊 Analyzing Results

### Compare All Checkpoints
```bash
python compare_batch_results.py eval_results_batch_YYYYMMDD_HHMMSS
```

**Output Example:**
```
RMSE COMPARISON (Lower is Better)
Checkpoint                     Breathiness     Roughness       Strain
checkpoint_best                12.3456         15.6789         14.2345
checkpoint_0                   13.2345         16.5678         15.1234
checkpoint_1                   12.8901         15.9012         14.5678
checkpoint_2                   12.5678         15.7890         14.3456

BEST CHECKPOINTS (by Average RMSE)
1. checkpoint_best             Avg RMSE: 14.0863
2. checkpoint_2                Avg RMSE: 14.2341
3. checkpoint_1                Avg RMSE: 14.4537
```

### Detailed Analysis of One Checkpoint
```bash
python analyze_eval_results.py eval_results_batch_YYYYMMDD_HHMMSS/eval_results_checkpoint_best.json
```

### Compare Two Specific Checkpoints
```bash
python analyze_eval_results.py \
    eval_results_batch_YYYYMMDD_HHMMSS/eval_results_checkpoint_best.json \
    --compare eval_results_batch_YYYYMMDD_HHMMSS/eval_results_checkpoint_0.json
```

---

## 🛠️ Troubleshooting Commands

### Check if Checkpoint Exists
```bash
ls -lh output_dir_breathiness/20241030_120000/checkpoint_best.pth
```

### Check GPU Availability
```bash
nvidia-smi
```

### Check Disk Space
```bash
df -h
```

### Check Job Error
```bash
cat logs/eval-pvqd-batch-JOBID.err
```

### Verify Python Environment
```bash
python --version
which python
```

---

## 📝 Example: Full Session

```bash
# 1. Connect
ssh user@mahti.csc.fi

# 2. Navigate
cd /scratch/project_2011211/fang/SALMONN

# 3. Find checkpoints
find output_dir_breathiness -name "checkpoint_*.pth" -type f
# Output:
# output_dir_breathiness/20241030_120000/checkpoint_best.pth
# output_dir_breathiness/20241030_120000/checkpoint_0.pth
# output_dir_breathiness/20241030_120000/checkpoint_1.pth

# 4. Quick test
python eval_multi_task.py \
    --cfg-path configs/config_pvqd.yaml \
    --ckpt-path output_dir_breathiness/20241030_120000/checkpoint_best.pth \
    --device cuda:0

# 5. Edit batch script
nano batch_eval.sh
# Update CHECKPOINTS list with the 3 paths above

# 6. Submit
sbatch run_eval_batch.sh
# Output: Submitted batch job 12345678

# 7. Monitor
squeue -u $USER
tail -f logs/eval-pvqd-batch-12345678.out

# 8. After ~10-15 minutes, analyze
python compare_batch_results.py eval_results_batch_20241030_183000

# 9. Download (from local machine)
scp -r user@mahti.csc.fi:/scratch/project_2011211/fang/SALMONN/eval_results_batch_20241030_183000 .
```

---

## ⏱️ Typical Timeline

| Step | Time | Command |
|------|------|---------|
| SSH + Navigate | 1 min | `ssh` + `cd` |
| Find Checkpoints | 1 min | `find` |
| Quick Test | 5 min | `python eval_multi_task.py` |
| Edit Script | 2 min | `nano batch_eval.sh` |
| Submit Job | 1 min | `sbatch` |
| Batch Evaluation | 10-15 min | (running on cluster) |
| Analyze Results | 2 min | `python compare_batch_results.py` |
| Download | 2 min | `scp` |
| **Total** | **~25-30 min** | |

---

## 💡 Tips

1. **Keep terminal open** while monitoring job
2. **Use `watch`** for continuous monitoring: `watch squeue -u $USER`
3. **Save checkpoint paths** in a text file for reference
4. **Check results immediately** after job completes
5. **Download results** before disconnecting from cluster

---

## 🆘 Quick Help

**Job not starting?**
```bash
scontrol show job JOBID
```

**Out of memory?**
Edit `batch_eval.sh` and add:
```bash
--options run.batch_size_eval 4 run.num_workers 2
```

**Job timeout?**
Edit `run_eval_batch.sh`:
```bash
#SBATCH --time=08:00:00
```

**Can't find results?**
```bash
ls -d eval_results_batch_*
```

---

Ready to evaluate! 🚀

