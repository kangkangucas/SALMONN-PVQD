# 3D Model Evaluation - Error Fix Summary

## Error Message
```
Error processing sample 9: Missing key generate
full_key: generate
object_type=dict
```

## Root Cause

The evaluation script was hardcoded to expect PVQD test data format:
```json
{
  "annotation": [
    {"path": "...", "text": "12.5"}  // PVQD format
  ]
}
```

But the 3D model test data has a different format:
```json
{
  "annotation": [
    {"path": "...", "generate": "12.5"}  // 3D format
  ]
}
```

## Solution

### 1. Updated `eval_multi_task.py`

Added a new function `get_annotation_fields()` that handles multiple data formats:

```python
def get_annotation_fields(ann):
    """Extract path and ground truth from annotation, handling different formats."""
    # Try PVQD format first
    if 'path' in ann and 'text' in ann:
        return ann['path'], float(ann['text'])
    
    # Try 3D format (with 'generate' key)
    if 'path' in ann and 'generate' in ann:
        generate_val = ann['generate']
        if isinstance(generate_val, list) and len(generate_val) > 0:
            return ann['path'], float(generate_val[0])
        else:
            return ann['path'], float(generate_val)
    
    # Try other common formats
    if 'audio' in ann and 'score' in ann:
        return ann['audio'], float(ann['score'])
    
    raise ValueError(f"Cannot extract path and ground truth from annotation: {ann}")
```

### 2. Updated `evaluate_task()` function

Made test data loading more flexible:

```python
# Handle different test data formats
if isinstance(test_data, dict) and 'annotation' in test_data:
    annotations = test_data['annotation']
elif isinstance(test_data, list):
    annotations = test_data
else:
    raise ValueError(f"Unexpected test data format: {type(test_data)}")
```

### 3. Updated `main()` function

Made config loading more flexible:

```python
# Load test prompts - use config path if available
prompt_path = getattr(cfg.config.model, 'test_prompt_path', 
                      'prompts/test_prompts_pvqd.json')

# Get test paths from config if available
if hasattr(cfg.config.datasets, 'test_task_paths'):
    test_task_paths = cfg.config.datasets.test_task_paths
    if isinstance(test_task_paths, dict):
        tasks = dict(test_task_paths)
    else:
        tasks = dict(test_task_paths.items())

# Fallback to PVQD if not in config
if not tasks:
    tasks = {
        'breathiness': 'fang_data_prepare/splits_pvqd/breathiness_test.json',
        'roughness': 'fang_data_prepare/splits_pvqd/roughness_test.json',
        'strain': 'fang_data_prepare/splits_pvqd/strain_test.json'
    }
```

### 4. Created `configs/config_3d.yaml`

New configuration file for 3D model with placeholder for test data paths:

```yaml
datasets:
  test_task_paths:
    task1: "path/to/3d_test_task1.json"
    task2: "path/to/3d_test_task2.json"
    task3: "path/to/3d_test_task3.json"
```

### 5. Created `EVAL_3D_GUIDE.md`

Comprehensive guide for setting up 3D model evaluation.

## Supported Data Formats

The updated script now handles:

**Format 1: PVQD**
```json
{"annotation": [{"path": "...", "text": "12.5"}]}
```

**Format 2: 3D with "generate" key**
```json
{"annotation": [{"path": "...", "generate": "12.5"}]}
```

**Format 3: 3D with list value**
```json
{"annotation": [{"path": "...", "generate": ["12.5"]}]}
```

**Format 4: Alternative format**
```json
{"annotation": [{"audio": "...", "score": "12.5"}]}
```

## How to Use

### For PVQD Model (No changes needed)
```bash
python eval_multi_task.py \
    --cfg-path configs/config_pvqd.yaml \
    --ckpt-path path/to/checkpoint.pth \
    --device cuda:0
```

### For 3D Model (New)
```bash
# 1. Update config_3d.yaml with your test data paths
nano configs/config_3d.yaml

# 2. Run evaluation
python eval_multi_task.py \
    --cfg-path configs/config_3d.yaml \
    --ckpt-path output_dir_3d/202510292335/checkpoint_best.pth \
    --device cuda:0
```

## Files Modified/Created

| File | Status | Changes |
|------|--------|---------|
| `eval_multi_task.py` | Modified | Added `get_annotation_fields()`, flexible data loading |
| `configs/config_3d.yaml` | Created | New config for 3D model |
| `EVAL_3D_GUIDE.md` | Created | Setup and troubleshooting guide |
| `FIX_3D_EVAL_SUMMARY.md` | Created | This file |

## Next Steps

1. **Find your 3D test data** on the remote cluster
2. **Check the data structure** to confirm format
3. **Update `configs/config_3d.yaml`** with correct paths
4. **Test with single checkpoint** first
5. **Batch evaluate** if needed

## Troubleshooting

### Still getting "Missing key generate"?

1. Check the actual structure of your test data:
```bash
python3 << 'EOF'
import json
with open("your_test_file.json") as f:
    data = json.load(f)
    print(json.dumps(data['annotation'][0], indent=2))
EOF
```

2. If the structure is different, update `get_annotation_fields()` function

### Getting "Cannot extract path and ground truth"?

The test data format is not recognized. Update `get_annotation_fields()` to handle your format.

## Questions?

Refer to `EVAL_3D_GUIDE.md` for detailed setup instructions and troubleshooting.

