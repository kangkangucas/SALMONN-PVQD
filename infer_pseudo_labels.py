#!/usr/bin/env python3
"""
Inference script to generate pseudo labels for a new dataset using a trained SALMONN model.
Supports checkpoint resumption for long-running inference jobs.

Usage:
    python infer_pseudo_labels.py \
        --audio-dir /path/to/audio/directory \
        --output-csv results.csv \
        --ckpt-path output_dir_3d/202510292335/checkpoint_1.pth \
        --cfg-path configs/config_pvqd.yaml \
        --device cuda:0
"""

import argparse
import os
import json
import csv
import glob
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from transformers import WhisperFeatureExtractor

from config import Config
from models.salmonn import SALMONN

from utils import prepare_one_sample
def parse_args():
    parser = argparse.ArgumentParser(description='Generate pseudo labels for new dataset')
    parser.add_argument("--audio-dir", type=str, required=True, 
                        help='Path to directory containing WAV files')
    parser.add_argument("--output-csv", type=str, default="pseudo_labels.csv",
                        help='Output CSV file path')
    parser.add_argument("--ckpt-path", type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument("--cfg-path", type=str, default="configs/config_pvqd.yaml",
                        help='Path to configuration file')
    parser.add_argument("--device", type=str, default="cuda:0",
                        help='Device to use for inference')
    parser.add_argument("--batch-size", type=int, default=1,
                        help='Batch size for inference')
    return parser.parse_args()


def extract_score(output_text):
    """Extract numerical score from model output."""
    try:
        import re
        numbers = re.findall(r'\d+\.?\d*', output_text.strip())
        if numbers:
            return float(numbers[0])
        else:
            return None
    except:
        return None


def load_checkpoint_state(csv_path):
    """Load previously processed files from CSV."""
    processed_files = set()
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            if reader is not None:
                for row in reader:
                    if 'filename' in row:
                        processed_files.add(row['filename'])
    return processed_files


def save_result(csv_path, filename, breathiness, roughness, strain, error_msg=None):
    """Append result to CSV file."""
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'filename', 'breathiness', 'roughness', 'strain', 'error'
        ])
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'filename': filename,
            'breathiness': breathiness if breathiness is not None else '',
            'roughness': roughness if roughness is not None else '',
            'strain': strain if strain is not None else '',
            'error': error_msg if error_msg else ''
        })


def main():
    args = parse_args()
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("SALMONN Pseudo Label Generation")
    print("=" * 80)
    print(f"Audio directory: {args.audio_dir}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Device: {args.device}")
    print()
    
    # Load model
    print("Loading model...")
    cfg = Config(argparse.Namespace(cfg_path=args.cfg_path, options=None))
    model = SALMONN.from_config(cfg.config.model)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(args.device)
    model.eval()
    
    # Load whisper processor
    wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)
    
    # Load prompts
    prompt_path = getattr(cfg.config.model, 'test_prompt_path', 'prompts/test_prompts_pvqd.json')
    with open(prompt_path, 'r') as f:
        prompts = json.load(f)
    
    # Get list of WAV files
    audio_dir = Path(args.audio_dir)
    wav_files = sorted(glob.glob(str(audio_dir / "**/*.wav"), recursive=True))
    print(f"Found {len(wav_files)} WAV files")
    
    # Load checkpoint state
    processed_files = load_checkpoint_state(args.output_csv)
    remaining_files = [f for f in wav_files if os.path.basename(f) not in processed_files]
    
    print(f"Already processed: {len(processed_files)} files")
    print(f"Remaining to process: {len(remaining_files)} files")
    print()
    
    # Process files
    print("Starting inference...")
    print("=" * 80)
    
    for wav_path in tqdm(remaining_files, desc="Processing"):
        filename = os.path.basename(wav_path)

        try:
            # Prepare sample
            samples = prepare_one_sample(wav_path, wav_processor, cuda_enabled=True)

            # Generate predictions for each task
            results = {}
            for task_name in ['breathiness', 'roughness', 'strain']:
                try:
                    # Get the correct prompt key
                    prompt_key = f"{task_name}_evaluation_onlyscore_test"
                    prompt_template = prompts.get(prompt_key, "")

                    if not prompt_template:
                        print(f"  Warning: Prompt key '{prompt_key}' not found in prompts")
                        results[task_name] = None
                        continue

                    prompt = [cfg.config.model.prompt_template.format(prompt_template)]

                    with torch.no_grad():
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            output = model.generate(samples, cfg.config.generate, prompts=prompt)[0]

                    score = extract_score(output)
                    results[task_name] = score

                except Exception as e:
                    print(f"  Error processing {task_name} for {filename}: {str(e)}")
                    results[task_name] = None
            
            # Save result
            save_result(
                args.output_csv,
                filename,
                results.get('breathiness'),
                results.get('roughness'),
                results.get('strain')
            )
            
        except Exception as e:
            print(f"  Error processing {filename}: {str(e)}")
            save_result(args.output_csv, filename, None, None, None, str(e))
    
    print("=" * 80)
    print(f"Inference complete! Results saved to: {args.output_csv}")
    
    # Print summary
    processed_files = load_checkpoint_state(args.output_csv)
    print(f"Total processed: {len(processed_files)} files")


if __name__ == "__main__":
    main()

