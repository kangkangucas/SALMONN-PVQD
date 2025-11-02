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
import numpy as np
import argparse
from pathlib import Path


def analyze_results(results_file):
    """Analyze evaluation results and print detailed statistics."""
    
    if not Path(results_file).exists():
        print(f"Error: Results file '{results_file}' not found")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("\n" + "="*70)
    print("DETAILED EVALUATION ANALYSIS")
    print("="*70)
    
    all_rmse = []
    all_mae = []
    
    for task_name, task_results in results.items():
        print(f"\n{'Task: ' + task_name:^70}")
        print("-" * 70)
        
        rmse = task_results['rmse']
        mae = task_results['mae']
        num_samples = task_results['num_samples']
        failed = task_results['failed_samples']
        
        predictions = np.array(task_results['predictions'])
        ground_truths = np.array(task_results['ground_truths'])
        
        # Calculate additional statistics
        errors = predictions - ground_truths
        abs_errors = np.abs(errors)
        
        print(f"Samples evaluated: {num_samples}")
        print(f"Failed samples: {failed}")
        print(f"\nMetrics:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"\nError Statistics:")
        print(f"  Min error:    {np.min(errors):+.4f}")
        print(f"  Max error:    {np.max(errors):+.4f}")
        print(f"  Mean error:   {np.mean(errors):+.4f}")
        print(f"  Std error:    {np.std(errors):.4f}")
        print(f"  Median error: {np.median(errors):+.4f}")
        print(f"\nAbsolute Error Statistics:")
        print(f"  Min abs error:    {np.min(abs_errors):.4f}")
        print(f"  Max abs error:    {np.max(abs_errors):.4f}")
        print(f"  Mean abs error:   {np.mean(abs_errors):.4f}")
        print(f"  Median abs error: {np.median(abs_errors):.4f}")
        
        # Percentiles
        print(f"\nError Percentiles:")
        for p in [25, 50, 75, 90, 95]:
            val = np.percentile(abs_errors, p)
            print(f"  {p}th percentile: {val:.4f}")
        
        # Prediction range
        print(f"\nPrediction Range:")
        print(f"  Min prediction: {np.min(predictions):.2f}")
        print(f"  Max prediction: {np.max(predictions):.2f}")
        print(f"  Mean prediction: {np.mean(predictions):.2f}")
        
        print(f"\nGround Truth Range:")
        print(f"  Min ground truth: {np.min(ground_truths):.2f}")
        print(f"  Max ground truth: {np.max(ground_truths):.2f}")
        print(f"  Mean ground truth: {np.mean(ground_truths):.2f}")
        
        # Correlation
        correlation = np.corrcoef(predictions, ground_truths)[0, 1]
        print(f"\nCorrelation: {correlation:.4f}")
        
        all_rmse.append(rmse)
        all_mae.append(mae)
    
    # Overall summary
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    print(f"Average RMSE: {np.mean(all_rmse):.4f}")
    print(f"Average MAE:  {np.mean(all_mae):.4f}")
    print(f"Std RMSE:     {np.std(all_rmse):.4f}")
    print(f"Std MAE:      {np.std(all_mae):.4f}")
    
    # Performance rating
    avg_rmse = np.mean(all_rmse)
    if avg_rmse < 10:
        rating = "Excellent"
    elif avg_rmse < 15:
        rating = "Good"
    elif avg_rmse < 20:
        rating = "Fair"
    else:
        rating = "Poor"
    
    print(f"\nPerformance Rating: {rating}")
    print("="*70 + "\n")


def compare_results(results_file1, results_file2):
    """Compare two evaluation results."""
    
    with open(results_file1, 'r') as f:
        results1 = json.load(f)
    
    with open(results_file2, 'r') as f:
        results2 = json.load(f)
    
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    print(f"File 1: {results_file1}")
    print(f"File 2: {results_file2}")
    print("-" * 70)
    
    for task_name in results1.keys():
        if task_name not in results2:
            print(f"Warning: Task '{task_name}' not in second results file")
            continue
        
        rmse1 = results1[task_name]['rmse']
        rmse2 = results2[task_name]['rmse']
        mae1 = results1[task_name]['mae']
        mae2 = results2[task_name]['mae']
        
        rmse_diff = rmse2 - rmse1
        mae_diff = mae2 - mae1
        rmse_pct = (rmse_diff / rmse1) * 100 if rmse1 != 0 else 0
        mae_pct = (mae_diff / mae1) * 100 if mae1 != 0 else 0
        
        print(f"\n{task_name}:")
        print(f"  RMSE: {rmse1:.4f} → {rmse2:.4f} ({rmse_diff:+.4f}, {rmse_pct:+.1f}%)")
        print(f"  MAE:  {mae1:.4f} → {mae2:.4f} ({mae_diff:+.4f}, {mae_pct:+.1f}%)")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze evaluation results')
    parser.add_argument('results_file', help='Path to eval_results_multi_task.json')
    parser.add_argument('--compare', help='Compare with another results file')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_results(args.results_file, args.compare)
    else:
        analyze_results(args.results_file)


if __name__ == "__main__":
    main()

