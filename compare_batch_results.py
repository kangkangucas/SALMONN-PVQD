#!/usr/bin/env python3
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

import json
import argparse
from pathlib import Path
import numpy as np


def compare_batch_results(results_dir):
    """Compare evaluation results from multiple checkpoints."""
    
    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory '{results_dir}' not found")
        return
    
    # Find all result files
    results_files = sorted(results_dir.glob("eval_results_*.json"))
    
    if not results_files:
        print(f"Error: No result files found in '{results_dir}'")
        return
    
    print("\n" + "="*80)
    print("BATCH EVALUATION COMPARISON")
    print("="*80)
    print(f"Results directory: {results_dir}")
    print(f"Number of checkpoints: {len(results_files)}")
    print("="*80)
    
    all_results = {}
    for results_file in results_files:
        with open(results_file, 'r') as f:
            results = json.load(f)
            ckpt_name = results_file.stem.replace("eval_results_", "")
            all_results[ckpt_name] = results
    
    # Extract metrics for each task
    tasks = ['breathiness', 'roughness', 'strain']
    metrics = {}
    
    for task in tasks:
        metrics[task] = {'rmse': [], 'mae': [], 'names': []}
        for ckpt_name, results in sorted(all_results.items()):
            if task in results:
                metrics[task]['rmse'].append(results[task]['rmse'])
                metrics[task]['mae'].append(results[task]['mae'])
                metrics[task]['names'].append(ckpt_name)
    
    # Print RMSE comparison
    print("\n" + "-"*80)
    print("RMSE COMPARISON (Lower is Better)")
    print("-"*80)
    print(f"{'Checkpoint':<30} {'Breathiness':<15} {'Roughness':<15} {'Strain':<15}")
    print("-"*80)
    
    for ckpt_name, results in sorted(all_results.items()):
        breathiness_rmse = results.get('breathiness', {}).get('rmse', 'N/A')
        roughness_rmse = results.get('roughness', {}).get('rmse', 'N/A')
        strain_rmse = results.get('strain', {}).get('rmse', 'N/A')
        
        if isinstance(breathiness_rmse, (int, float)):
            print(f"{ckpt_name:<30} {breathiness_rmse:<15.4f} "
                  f"{roughness_rmse:<15.4f} {strain_rmse:<15.4f}")
        else:
            print(f"{ckpt_name:<30} {str(breathiness_rmse):<15} "
                  f"{str(roughness_rmse):<15} {str(strain_rmse):<15}")
    
    # Print MAE comparison
    print("\n" + "-"*80)
    print("MAE COMPARISON (Lower is Better)")
    print("-"*80)
    print(f"{'Checkpoint':<30} {'Breathiness':<15} {'Roughness':<15} {'Strain':<15}")
    print("-"*80)
    
    for ckpt_name, results in sorted(all_results.items()):
        breathiness_mae = results.get('breathiness', {}).get('mae', 'N/A')
        roughness_mae = results.get('roughness', {}).get('mae', 'N/A')
        strain_mae = results.get('strain', {}).get('mae', 'N/A')
        
        if isinstance(breathiness_mae, (int, float)):
            print(f"{ckpt_name:<30} {breathiness_mae:<15.4f} "
                  f"{roughness_mae:<15.4f} {strain_mae:<15.4f}")
        else:
            print(f"{ckpt_name:<30} {str(breathiness_mae):<15} "
                  f"{str(roughness_mae):<15} {str(strain_mae):<15}")
    
    # Print statistics
    print("\n" + "-"*80)
    print("STATISTICS")
    print("-"*80)
    
    for task in tasks:
        if metrics[task]['rmse']:
            rmse_values = np.array(metrics[task]['rmse'])
            mae_values = np.array(metrics[task]['mae'])
            
            print(f"\n{task.upper()}:")
            print(f"  RMSE - Mean: {np.mean(rmse_values):.4f}, "
                  f"Std: {np.std(rmse_values):.4f}, "
                  f"Min: {np.min(rmse_values):.4f}, "
                  f"Max: {np.max(rmse_values):.4f}")
            print(f"  MAE  - Mean: {np.mean(mae_values):.4f}, "
                  f"Std: {np.std(mae_values):.4f}, "
                  f"Min: {np.min(mae_values):.4f}, "
                  f"Max: {np.max(mae_values):.4f}")
    
    # Find best checkpoint
    print("\n" + "-"*80)
    print("BEST CHECKPOINTS (by Average RMSE)")
    print("-"*80)
    
    avg_rmse_per_ckpt = {}
    for ckpt_name, results in all_results.items():
        rmse_values = []
        for task in tasks:
            if task in results:
                rmse_values.append(results[task]['rmse'])
        if rmse_values:
            avg_rmse_per_ckpt[ckpt_name] = np.mean(rmse_values)
    
    if avg_rmse_per_ckpt:
        sorted_ckpts = sorted(avg_rmse_per_ckpt.items(), key=lambda x: x[1])
        for i, (ckpt_name, avg_rmse) in enumerate(sorted_ckpts[:3], 1):
            print(f"{i}. {ckpt_name:<30} Avg RMSE: {avg_rmse:.4f}")
    
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compare batch evaluation results'
    )
    parser.add_argument(
        'results_dir',
        help='Path to batch results directory'
    )
    
    args = parser.parse_args()
    compare_batch_results(args.results_dir)


if __name__ == "__main__":
    main()

