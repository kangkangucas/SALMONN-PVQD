# Evaluating 3D Model - Setup Guide

## What is the 3D Model?

The **3D model** is trained on **3 tasks combined**:
- **Breathiness** (dimension 1)
- **Roughness** (dimension 2)
- **Strain** (dimension 3)

It's a **multi-task model** that learns to predict all 3 voice quality dimensions simultaneously.

## Problem

You got this error when trying to evaluate the 3D model:
```
Error processing sample 9: Missing key generate
full_key: generate
object_type=dict
```

This occurred because:
- The 3D model was trained on multi-task data with a different JSON structure
- The test data has a "generate" key instead of "text"
- The evaluation script needed to handle this format

## Solution

The evaluation script has been updated to handle **multiple data formats**:

### Supported Data Formats

**Format 1: PVQD Format (Original)**
```json
{
  "annotation": [
    {
      "path": "/path/to/audio.wav",
      "text": "12.5"
    }
  ]
}
```

**Format 2: 3D Format (with "generate" key)**
```json
{
  "annotation": [
    {
      "path": "/path/to/audio.wav",
      "generate": "12.5"  // or ["12.5"]
    }
  ]
}
```

**Format 3: Alternative Format**
```json
{
  "annotation": [
    {
      "audio": "/path/to/audio.wav",
      "score": "12.5"
    }
  ]
}
```

## Setup Steps

### Step 1: Verify Test Data Files Exist

The 3D model uses the same PVQD test data files:

```bash
ssh user@mahti.csc.fi
cd /scratch/project_2011211/fang/SALMONN

# Verify test files exist
ls -lh fang_data_prepare/splits_pvqd/*_test.json
```

Expected output:
```
fang_data_prepare/splits_pvqd/breathiness_test.json
fang_data_prepare/splits_pvqd/roughness_test.json
fang_data_prepare/splits_pvqd/strain_test.json
```

### Step 2: Config is Already Set Up

The `configs/config_3d.yaml` is already configured to use the PVQD test files:

```yaml
datasets:
  test_task_paths:
    breathiness: "fang_data_prepare/splits_pvqd/breathiness_test.json"
    roughness: "fang_data_prepare/splits_pvqd/roughness_test.json"
    strain: "fang_data_prepare/splits_pvqd/strain_test.json"
```

No changes needed!

### Step 3: Test Single Checkpoint

```bash
python eval_multi_task.py \
    --cfg-path configs/config_3d.yaml \
    --ckpt-path output_dir_3d/202510292335/checkpoint_best.pth \
    --device cuda:0
```

### Step 4: Batch Evaluate (if multiple checkpoints)

Edit `batch_eval.sh`:
```bash
nano batch_eval.sh
```

Update:
```bash
CONFIG="configs/config_3d.yaml"

CHECKPOINTS=(
    "output_dir_3d/202510292335/checkpoint_best.pth"
    "output_dir_3d/202510292335/checkpoint_1.pth"
    "output_dir_3d/202510292335/checkpoint_3.pth"
)
```

Then submit:
```bash
sbatch run_eval_batch.sh
```

## Troubleshooting

### Error: "Missing key generate"

**Cause:** The test data has a "generate" key but the script doesn't know how to extract it.

**Solution:** 
1. Check the actual structure of your test data
2. Update `get_annotation_fields()` function in `eval_multi_task.py` if needed

### Error: "Cannot extract path and ground truth from annotation"

**Cause:** The test data has an unexpected structure.

**Solution:**
1. Print the first sample to see the structure:
```bash
python3 << 'EOF'
import json
with open("your_test_file.json") as f:
    data = json.load(f)
    print(json.dumps(data['annotation'][0], indent=2))
EOF
```

2. Update `get_annotation_fields()` function to handle your format

### Error: "Prompt key not found"

**Cause:** The test prompts file doesn't have the right keys for your tasks.

**Solution:**
1. Check what prompts are available:
```bash
python3 << 'EOF'
import json
with open("prompts/test_prompts_pvqd.json") as f:
    prompts = json.load(f)
    print("Available prompts:")
    for key in prompts.keys():
        print(f"  - {key}")
EOF
```

2. Update `test_prompt_path` in config_3d.yaml to point to the correct prompts file

## What Gets Evaluated

For each task in `test_task_paths`:
- Loads test data from JSON file
- Processes each audio sample
- Generates prediction using the model
- Compares with ground truth
- Calculates RMSE, MAE, Correlation

## Output

Results are saved to `eval_results_multi_task.json`:
```json
{
  "task1": {
    "task": "task1",
    "rmse": 12.34,
    "mae": 10.56,
    "num_samples": 30,
    "failed_samples": 0,
    "predictions": [...],
    "ground_truths": [...]
  },
  "task2": {...},
  "task3": {...}
}
```

## Next Steps

1. **Find your 3D test data files** on the remote cluster
2. **Check the data structure** to understand the format
3. **Update config_3d.yaml** with correct paths
4. **Test with a single checkpoint** first
5. **Batch evaluate** if everything works

## Questions?

If you encounter issues:
1. Check the error message carefully
2. Print the test data structure to understand it
3. Update the config file accordingly
4. Test again

The evaluation script is now flexible enough to handle different data formats!

