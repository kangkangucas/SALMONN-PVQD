#!/usr/bin/env python3
import json
import numpy as np
import soundfile as sf
from dataset import SALMONNDataset
from configs.config import Config

# Load config
config = Config("configs/config_pvqd.yaml")

# Load dataset
dataset = SALMONNDataset(
    annotation_path=config.annotation_path,
    wav_processor=config.wav_processor,
    enable_augmentation=config.augmentation.enable,
    augmentation_types=config.augmentation.types,
    augmentation_prob=config.augmentation.prob,
)

print(f"Dataset size: {len(dataset)}")
print(f"Augmentation enabled: {config.augmentation.enable}")
print()

# Check first 5 samples
for i in range(5):
    sample = dataset[i]
    raw_wav = sample["raw_wav"]
    print(f"Sample {i}: raw_wav length = {len(raw_wav)}")

