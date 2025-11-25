#!/bin/bash

# Batch evaluation script for multiple SALMONN models
# This script evaluates multiple model checkpoints and saves results

set -euo pipefail

# Configuration
CONFIG="configs/config_pvqd.yaml"
DEVICE="cuda:0"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="eval_results_batch_${TIMESTAMP}"

# Create results directory
mkdir -p "$RESULTS_DIR"

# List of checkpoints to evaluate
# MODIFY THIS LIST WITH YOUR CHECKPOINT PATHS
CHECKPOINTS=(
    # "output_dir_3d/202510292335/checkpoint_best.pth"
    # "output_dir_3d/202510292335/checkpoint_1.pth" # no aug best one
    # "output_dir_3d/202510292335/checkpoint_3.pth"
    # "output_dir/202511092248/checkpoint_best.pth"
    # "output_dir/202511092248/checkpoint_3.pth"
    # "output_dir/202511082251/checkpoint_best.pth"
    # "output_dir/202511082251/checkpoint_1.pth" # no aug 2rd best one
    # "output_dir/202511091143/checkpoint_best.pth"
    # "output_dir/202511091143/checkpoint_1.pth"   
    # "output_dir/202511101148/checkpoint_3.pth"  
    "output_dir/202511162149/checkpoint_4.pth" # aug 2rd best one
    "output_dir/202511162233/checkpoint_3.pth" # aug best one

)

echo "=========================================="
echo "Batch Model Evaluation"
echo "=========================================="
echo "Config: $CONFIG"
echo "Device: $DEVICE"
echo "Results Directory: $RESULTS_DIR"
echo "Number of checkpoints: ${#CHECKPOINTS[@]}"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Counter for tracking progress
total=${#CHECKPOINTS[@]}
current=0

for ckpt in "${CHECKPOINTS[@]}"; do
    current=$((current + 1))
    
    # Check if checkpoint exists
    if [ ! -f "$ckpt" ]; then
        echo "[${current}/${total}] ❌ Checkpoint not found: $ckpt"
        continue
    fi
    
    # Extract checkpoint name for output files
    ckpt_name=$(basename "$ckpt" .pth)
    output_file="$RESULTS_DIR/eval_results_${ckpt_name}.json"
    log_file="$RESULTS_DIR/eval_log_${ckpt_name}.txt"
    
    echo "[${current}/${total}] ⏳ Evaluating: $ckpt"
    echo "    Output: $output_file"
    echo "    Log: $log_file"
    
    # Run evaluation
    if python eval_multi_task.py \
        --cfg-path "$CONFIG" \
        --ckpt-path "$ckpt" \
        --device "$DEVICE" \
        2>&1 | tee "$log_file"; then
        
        # Move results to batch directory
        if [ -f "eval_results_multi_task.json" ]; then
            mv eval_results_multi_task.json "$output_file"
            echo "[${current}/${total}] ✅ Completed: $ckpt_name"
        else
            echo "[${current}/${total}] ⚠️  No results file generated for: $ckpt_name"
        fi
    else
        echo "[${current}/${total}] ❌ Failed: $ckpt_name"
    fi
    
    echo ""
done

echo "=========================================="
echo "Batch Evaluation Summary"
echo "=========================================="
echo "Results saved to: $RESULTS_DIR"
echo "End time: $(date)"
echo ""

# List all results
echo "Generated result files:"
ls -lh "$RESULTS_DIR"/eval_results_*.json 2>/dev/null || echo "No result files found"

echo ""
echo "To analyze results, run:"
echo "  python analyze_eval_results.py $RESULTS_DIR/eval_results_checkpoint_best.json"
echo ""
echo "To compare two checkpoints, run:"
echo "  python analyze_eval_results.py $RESULTS_DIR/eval_results_checkpoint_best.json \\"
echo "      --compare $RESULTS_DIR/eval_results_checkpoint_0.json"
echo "=========================================="

