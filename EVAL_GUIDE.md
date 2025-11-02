# Multi-Task Model Evaluation Guide

## Overview
The `eval_multi_task.py` script evaluates your trained SALMONN model on all three voice quality assessment tasks (breathiness, roughness, strain) and calculates RMSE and MAE metrics.

## Features
- ✅ Automatically loads all test samples from 3 tasks
- ✅ Calculates RMSE (Root Mean Square Error) for each task
- ✅ Calculates MAE (Mean Absolute Error) for each task
- ✅ Provides overall average RMSE across all tasks
- ✅ Saves detailed results to JSON file
- ✅ Handles errors gracefully with detailed logging
- ✅ Progress bar for monitoring evaluation

## Usage

### Basic Usage
```bash
python eval_multi_task.py --cfg-path configs/config_pvqd.yaml --device cuda:0
```

### With Custom Options
```bash
python eval_multi_task.py \
    --cfg-path configs/config_pvqd.yaml \
    --device cuda:0 \
    --options model.ckpt_path ./ckpt/your_model.pth
```

## What the Script Does

### 1. Model Loading
- Loads the SALMONN model from the checkpoint specified in config
- Initializes Whisper feature extractor for audio processing
- Sets model to evaluation mode

### 2. Test Data Processing
For each task (breathiness, roughness, strain):
- Loads test samples from `fang_data_prepare/splits_pvqd/{task}_test.json`
- Processes audio files using Whisper processor
- Generates predictions using the model
- Extracts numerical scores from model outputs

### 3. Metrics Calculation
- **RMSE**: Root Mean Square Error between predictions and ground truth
  - Formula: sqrt(mean((predictions - ground_truth)^2))
  - Lower is better
  
- **MAE**: Mean Absolute Error
  - Formula: mean(|predictions - ground_truth|)
  - Lower is better

### 4. Results Output
Prints:
```
============================================================
EVALUATION SUMMARY
============================================================
breathiness     - RMSE: 12.3456, MAE: 10.1234
roughness       - RMSE: 15.6789, MAE: 12.3456
strain          - RMSE: 14.2345, MAE: 11.5678

Average RMSE across all tasks: 14.0863

Results saved to eval_results_multi_task.json
```

## Output Files

### eval_results_multi_task.json
Contains detailed results for each task:
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
  ...
}
```

## Configuration

The script uses settings from `configs/config_pvqd.yaml`:
- Model checkpoint path
- Model architecture parameters
- Generation parameters (temperature, top_p, etc.)
- Whisper model path

## Important Notes

### Audio Path Handling
The test data contains absolute paths like:
```
/scratch/project_2011211/fang/data/PerceptualVoiceQualitiesDatabase/AudioFiles_16k/PT090.wav
```

If you're running on a different system:
1. **Option 1**: Update the paths in test JSON files
2. **Option 2**: Modify the script to handle path mapping
3. **Option 3**: Create symlinks to match the original paths

### Score Extraction
The script extracts the first number from model output using regex:
```python
numbers = re.findall(r'\d+\.?\d*', output_text.strip())
```

If your model outputs scores differently, modify the `extract_score()` function.

### GPU Memory
- Batch size: 1 (processes one sample at a time)
- Typical memory usage: ~20-25GB on A100 40G
- Evaluation time: ~2-5 minutes per task (30 samples each)

## Troubleshooting

### Issue: "Audio file not found"
**Solution**: Update audio paths in test JSON files or create symlinks

### Issue: "Could not extract score from output"
**Solution**: Check model output format and modify `extract_score()` function

### Issue: Out of memory
**Solution**: Reduce model precision or use smaller batch processing

### Issue: Low RMSE but visually wrong predictions
**Solution**: Check if score extraction is working correctly

## Example Results Interpretation

```
breathiness - RMSE: 12.34
```
- On average, predictions are off by ±12.34 points (on 0-100 scale)
- This is ~12.3% error rate

## Next Steps

1. **Analyze Results**: Check which task has best/worst performance
2. **Error Analysis**: Look at predictions vs ground truth in JSON
3. **Fine-tuning**: If RMSE is high, consider:
   - More training epochs
   - Different learning rate
   - Data augmentation
   - Prompt engineering

## Advanced Usage

### Evaluate Single Task
Modify the `tasks` dictionary in `main()`:
```python
tasks = {
    'breathiness': 'fang_data_prepare/splits_pvqd/breathiness_test.json'
}
```

### Custom Prompts
Modify prompts in `prompts/test_prompts_pvqd.json` and re-run

### Batch Processing
For faster evaluation, modify `prepare_one_sample()` to support batching

## Performance Benchmarks

Typical RMSE values for reference:
- **Excellent**: RMSE < 10
- **Good**: RMSE 10-15
- **Fair**: RMSE 15-20
- **Poor**: RMSE > 20

## Support

For issues or questions:
1. Check the error messages in console output
2. Review the eval_results_multi_task.json for detailed results
3. Verify audio paths are correct
4. Check model checkpoint is loaded correctly

