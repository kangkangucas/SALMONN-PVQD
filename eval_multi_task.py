# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import numpy as np
import torch
from transformers import WhisperFeatureExtractor
from tqdm import tqdm

from config import Config
from models.salmonn import SALMONN
from utils import prepare_one_sample


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate multi-task SALMONN model')
    parser.add_argument("--cfg-path", type=str, required=True, help='path to configuration file')
    parser.add_argument("--ckpt-path", type=str, default=None, help='path to model checkpoint (overrides config ckpt)')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    return parser.parse_args()


def extract_score(output_text):
    """Extract numerical score from model output."""
    try:
        # Try to extract the first number from the output
        import re
        numbers = re.findall(r'\d+\.?\d*', output_text.strip())
        if numbers:
            return float(numbers[0])
        else:
            return None
    except:
        return None


def calculate_rmse(predictions, ground_truths):
    """Calculate RMSE between predictions and ground truths."""
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    # Filter out None predictions
    valid_mask = predictions != None
    if not np.any(valid_mask):
        return None
    
    predictions = predictions[valid_mask].astype(float)
    ground_truths = ground_truths[valid_mask].astype(float)
    
    rmse = np.sqrt(np.mean((predictions - ground_truths) ** 2))
    return rmse


def get_annotation_fields(ann):
    """Extract path and ground truth from annotation, handling different formats."""
    # Try PVQD format first
    if 'path' in ann and 'text' in ann:
        return ann['path'], float(ann['text'])

    # Try 3D format (with 'generate' key)
    if 'path' in ann and 'generate' in ann:
        # 'generate' might be a list or string
        generate_val = ann['generate']
        if isinstance(generate_val, list) and len(generate_val) > 0:
            return ann['path'], float(generate_val[0])
        else:
            return ann['path'], float(generate_val)

    # Try other common formats
    if 'audio' in ann and 'score' in ann:
        return ann['audio'], float(ann['score'])

    raise ValueError(f"Cannot extract path and ground truth from annotation: {ann}")


def evaluate_task(model, wav_processor, test_data_path, task_name, prompt_template, device, cfg):
    """Evaluate model on a single task."""
    print(f"\n{'='*60}")
    print(f"Evaluating Task: {task_name}")
    print(f"{'='*60}")

    # Load test data
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)

    # Handle different test data formats
    if isinstance(test_data, dict) and 'annotation' in test_data:
        annotations = test_data['annotation']
    elif isinstance(test_data, list):
        annotations = test_data
    else:
        raise ValueError(f"Unexpected test data format: {type(test_data)}")

    print(f"Total test samples: {len(annotations)}")

    predictions = []
    ground_truths = []
    failed_samples = 0

    # Evaluate each sample
    for idx, ann in enumerate(tqdm(annotations, desc=f"Evaluating {task_name}")):
        try:
            wav_path, ground_truth = get_annotation_fields(ann)
            
            # Prepare sample
            samples = prepare_one_sample(wav_path, wav_processor, cuda_enabled=True)
            
            # Create prompt
            prompt = [cfg.config.model.prompt_template.format(prompt_template)]
            
            # Generate prediction
            with torch.cuda.amp.autocast(dtype=torch.float16):
                output = model.generate(samples, cfg.config.generate, prompts=prompt)[0]
            
            # Extract score
            pred_score = extract_score(output)
            
            if pred_score is not None:
                predictions.append(pred_score)
                ground_truths.append(ground_truth)
            else:
                failed_samples += 1
                print(f"  Warning: Could not extract score from output: {output}")
        
        except Exception as e:
            failed_samples += 1
            print(f"  Error processing sample {idx}: {str(e)}")
            continue
    
    # Calculate metrics
    print(f"\nSuccessfully evaluated: {len(predictions)}/{len(annotations)} samples")
    print(f"Failed samples: {failed_samples}")
    
    if len(predictions) > 0:
        rmse = calculate_rmse(predictions, ground_truths)
        mae = np.mean(np.abs(np.array(predictions) - np.array(ground_truths)))
        
        print(f"\nMetrics for {task_name}:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        
        return {
            'task': task_name,
            'rmse': rmse,
            'mae': mae,
            'num_samples': len(predictions),
            'failed_samples': failed_samples,
            'predictions': predictions,
            'ground_truths': ground_truths
        }
    else:
        print(f"No valid predictions for {task_name}")
        return None


def main():
    args = parse_args()
    cfg = Config(args)

    # Load model
    print("Loading model...")
    model = SALMONN.from_config(cfg.config.model)

    # Load checkpoint (from args or config)
    ckpt_path = args.ckpt_path if args.ckpt_path else cfg.config.model.ckpt
    if ckpt_path:
        print(f"Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt['model'], strict=False)
    else:
        print("Warning: No checkpoint specified. Using model with default weights.")

    model.to(args.device)
    model.eval()
    
    # Load whisper processor
    wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)
    
    # Load test prompts
    prompt_path = getattr(cfg.config.model, 'test_prompt_path', 'prompts/test_prompts_pvqd.json')
    with open(prompt_path, 'r') as f:
        test_prompts = json.load(f)

    # Define tasks to evaluate - try to get from config first
    tasks = {}
    if hasattr(cfg.config.datasets, 'test_task_paths'):
        # Multi-task mode
        test_task_paths = cfg.config.datasets.test_task_paths
        if isinstance(test_task_paths, dict):
            tasks = dict(test_task_paths)
        else:
            # Handle OmegaConf DictConfig
            tasks = dict(test_task_paths.items())

    # Fallback to default PVQD tasks if not in config
    if not tasks:
        tasks = {
            'breathiness': 'fang_data_prepare/splits_pvqd/breathiness_test.json',
            'roughness': 'fang_data_prepare/splits_pvqd/roughness_test.json',
            'strain': 'fang_data_prepare/splits_pvqd/strain_test.json'
        }
    
    # Evaluate each task
    results = {}
    with torch.no_grad():
        for task_name, test_data_path in tasks.items():
            # Get prompt for this task
            prompt_key = f"{task_name}_evaluation_onlyscore_test"
            if prompt_key not in test_prompts:
                print(f"Warning: Prompt key '{prompt_key}' not found in test_prompts_pvqd.json")
                continue
            
            prompt_template = test_prompts[prompt_key]
            
            # Evaluate task
            result = evaluate_task(
                model, wav_processor, test_data_path, task_name,
                prompt_template, args.device, cfg
            )
            
            if result:
                results[task_name] = result
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    overall_rmse_values = []
    for task_name, result in results.items():
        print(f"{task_name:15} - RMSE: {result['rmse']:.4f}, MAE: {result['mae']:.4f}")
        overall_rmse_values.append(result['rmse'])
    
    if overall_rmse_values:
        avg_rmse = np.mean(overall_rmse_values)
        print(f"\nAverage RMSE across all tasks: {avg_rmse:.4f}")
    
    # Save results to file
    output_file = "eval_results_multi_task.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()

