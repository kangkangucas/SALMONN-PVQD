# Fix: Missing Key "generate" Error

## Problem

You were getting this error when running evaluation:
```
Error processing sample 9: Missing key generate
full_key: generate
object_type=dict
```

## Root Cause

The error was coming from line 132 in `eval_multi_task.py`:
```python
output = model.generate(samples, cfg.config.generate, prompts=prompt)[0]
```

The code was trying to access `cfg.config.generate`, but this section was **missing from the config files**:
- `configs/config_3d.yaml` - ❌ No `generate` section
- `configs/config_pvqd.yaml` - ❌ No `generate` section

The `decode_config.yaml` had it, but the evaluation configs didn't.

## Solution

Added the `generate` section to both config files:

### configs/config_3d.yaml
```yaml
generate:
  max_new_tokens: 200
  num_beams: 4
  do_sample: False
  min_length: 1
  temperature: 1.0
  top_p: 0.9
  repetition_penalty: 1.0
  length_penalty: 1.0
```

### configs/config_pvqd.yaml
```yaml
generate:
  max_new_tokens: 200
  num_beams: 4
  do_sample: False
  min_length: 1
  temperature: 1.0
  top_p: 0.9
  repetition_penalty: 1.0
  length_penalty: 1.0
```

## What These Parameters Do

- **max_new_tokens**: Maximum number of tokens to generate (200)
- **num_beams**: Number of beams for beam search (4)
- **do_sample**: Whether to use sampling (False = use greedy/beam search)
- **min_length**: Minimum length of generated sequence (1)
- **temperature**: Controls randomness (1.0 = no randomness)
- **top_p**: Nucleus sampling parameter (0.9)
- **repetition_penalty**: Penalty for repeating tokens (1.0 = no penalty)
- **length_penalty**: Penalty for length (1.0 = no penalty)

## Files Modified

1. ✅ `configs/config_3d.yaml` - Added `generate` section
2. ✅ `configs/config_pvqd.yaml` - Added `generate` section

## How to Test

Now you can run evaluation without the "Missing key generate" error:

```bash
# Test 3D model
python eval_multi_task.py \
    --cfg-path configs/config_3d.yaml \
    --ckpt-path output_dir_3d/202510292335/checkpoint_best.pth \
    --device cuda:0

# Test PVQD model
python eval_multi_task.py \
    --cfg-path configs/config_pvqd.yaml \
    --ckpt-path path/to/checkpoint.pth \
    --device cuda:0
```

## Why This Happened

The evaluation script was copied from `cli_inference.py` which uses `cfg.config.generate`. However, the config files used for evaluation didn't include this section. The fix ensures all config files have the necessary generation parameters.

## Next Steps

1. Run evaluation again with the fixed configs
2. Check the results in `eval_results_multi_task.json`
3. If you want to adjust generation parameters, edit the `generate` section in the config files

