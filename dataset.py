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
import random

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import soundfile as sf
import numpy as np
from transformers import WhisperFeatureExtractor

from audio_augmentation import AudioAugmentor
class SALMONNDataset(Dataset):
    def __init__(self, ann_path, whisper_path):
        super().__init__()

        self.annotation = json.load(open(ann_path, "r"))["annotation"]

        self.wav_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        samples_spectrogram = [s["spectrogram"] for s in samples]
        cat_spectrogram = torch.stack(samples_spectrogram, dim=0)

        raw_wav = [torch.from_numpy(s["raw_wav"]) for s in samples]
        raw_wav_length = torch.tensor([len(s["raw_wav"]) for s in samples])
        raw_wav = pad_sequence(raw_wav, batch_first=True, padding_value=0)
        paddding_mask = torch.arange(raw_wav.size(1)).unsqueeze(0) >= raw_wav_length.unsqueeze(1)

        text = [s["text"] for s in samples]
        task = [s["task"] for s in samples]
        Q = [s["Q"] for s in samples]
        id = [s["id"] for s in samples]

        return {
            "spectrogram": cat_spectrogram,
            "raw_wav": raw_wav,
            "padding_mask": paddding_mask,
            "text": text,
            "task": task,
            "Q": Q,
            "id": id,
        }

    def __getitem__(self, index):
        ann = self.annotation[index]

        audio, sr = sf.read(ann["path"])
        if len(audio.shape) == 2: # stereo to mono
            audio = audio[:, 0]
        if "expand_wav" in ann:
            for p in ann["expand_wav"]:
                expand_audio, _ = sf.read(p)
                if len(expand_audio.shape) == 2:
                    expand_audio = expand_audio[:, 0]
                sil = np.zeros(1600, dtype=float)
                audio = np.concatenate((audio, sil, expand_audio), axis=0)
        if len(audio) < sr: # pad audio to at least 1s
            sil = np.zeros(sr - len(audio), dtype=float)
            audio = np.concatenate((audio, sil), axis=0)
        audio = audio[: sr * 30] # truncate audio to at most 30s

        spectrogram = self.wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze()
        text = ann["text"]
        task = ann.get("task", "asr")
        Q = ann.get("Q", "")

        return {
            "spectrogram": spectrogram,
            "raw_wav": audio,
            "text": text,
            "task": task,
            "Q": Q,
            "id": ann["path"],
        }


class MultiTaskDataset(Dataset):
    """
    A dataset class that combines multiple task datasets and samples them with balanced distribution.
    For each batch, it randomly selects similar numbers of samples from each task.
    """
    def __init__(self, task_ann_paths, whisper_path, augmentation_config=None):
        """
        Args:
            task_ann_paths: List of tuples (task_name, ann_path) or dict {task_name: ann_path}
                           or OmegaConf DictConfig {task_name: ann_path}
            whisper_path: Path to whisper model
        """
        super().__init__()

        self.wav_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)
        self.datasets = {}
        self.task_names = []
        self.task_indices = {}
        self.total_samples = 0
        
        
        # Initialize augmentation
        self.augmentation_config = augmentation_config or {}
        self.enable_augmentation = self.augmentation_config.get("enable", False)
        self.augmentation_prob = self.augmentation_config.get("prob", 0.5)
        self.augmentation_types = self.augmentation_config.get(
            "types", ["random"]
        )

        if self.enable_augmentation:
            self.augmentor = AudioAugmentor(sr=16000)
        else:
            self.augmentor = None
        # Load all datasets
        # Convert OmegaConf DictConfig or dict to list of tuples
        if not isinstance(task_ann_paths, list):
            # Handle both dict and OmegaConf DictConfig
            task_ann_paths = list(task_ann_paths.items())

        for task_name, ann_path in task_ann_paths:
            annotation = json.load(open(ann_path, "r"))["annotation"]
            self.datasets[task_name] = annotation
            self.task_names.append(task_name)
            self.task_indices[task_name] = list(range(len(annotation)))
            self.total_samples += len(annotation)

        # Create a mapping from global index to (task_name, local_index)
        self._build_index_mapping()

    def _build_index_mapping(self):
        """Build a mapping from global index to (task_name, local_index)"""
        self.index_mapping = []
        for task_name in self.task_names:
            for local_idx in range(len(self.datasets[task_name])):
                self.index_mapping.append((task_name, local_idx))

    def __len__(self):
        return self.total_samples

    def collater(self, samples):
        """Collate function for multi-task dataset"""
        samples_spectrogram = [s["spectrogram"] for s in samples]
        cat_spectrogram = torch.stack(samples_spectrogram, dim=0)

        raw_wav = [torch.from_numpy(s["raw_wav"]) for s in samples]
        raw_wav_length = torch.tensor([len(s["raw_wav"]) for s in samples])
        raw_wav = pad_sequence(raw_wav, batch_first=True, padding_value=0)
        paddding_mask = torch.arange(raw_wav.size(1)).unsqueeze(0) >= raw_wav_length.unsqueeze(1)

        text = [s["text"] for s in samples]
        task = [s["task"] for s in samples]
        Q = [s["Q"] for s in samples]
        id = [s["id"] for s in samples]

        return {
            "spectrogram": cat_spectrogram,
            "raw_wav": raw_wav,
            "padding_mask": paddding_mask,
            "text": text,
            "task": task,
            "Q": Q,
            "id": id,
        }
    
    def __getitem__(self, index):
        task_name, local_idx = self.index_mapping[index]
        ann = self.datasets[task_name][local_idx]

        audio, sr = sf.read(ann["path"])
        if len(audio.shape) == 2:  # stereo to mono
            audio = audio[:, 0]
        if "expand_wav" in ann:
            for p in ann["expand_wav"]:
                expand_audio, _ = sf.read(p)
                if len(expand_audio.shape) == 2:
                    expand_audio = expand_audio[:, 0]
                sil = np.zeros(1600, dtype=float)
                audio = np.concatenate((audio, sil, expand_audio), axis=0)
        if len(audio) < sr:  # pad audio to at least 1s
            sil = np.zeros(sr - len(audio), dtype=float)
            audio = np.concatenate((audio, sil), axis=0)
            
        # # ======== 调试：增强前信息 ========
        # print("\n[DEBUG before augmentation]")
        # print(f"  shape: {audio.shape}, sr: {sr}, dtype: {audio.dtype}")
        # print(f"  min={np.min(audio):.5f}, max={np.max(audio):.5f}, mean={np.mean(audio):.5f}, std={np.std(audio):.5f}")
        # print(f"  has_nan={np.isnan(audio).any()}, has_inf={np.isinf(audio).any()}")

        # Apply augmentation if enabled
        if self.enable_augmentation and self.augmentor is not None:
            aug_type = np.random.choice(self.augmentation_types)
            # print(f"[DEBUG] Applying augmentation: {aug_type}")
            audio = self.augmentor.apply_augmentation(
                audio,
                augmentation_type=aug_type,
                prob=self.augmentation_prob
            )
        # # ======== 调试：增强后信息 ========
        # print("[DEBUG after augmentation]")
        # print(f"  shape: {audio.shape}, dtype: {type(audio)}, type: {type(audio)}")
        # if isinstance(audio, torch.Tensor):
        #     audio_np = audio.detach().cpu().numpy()
        # else:
        #     audio_np = np.array(audio)
        # print(f"  min={np.min(audio_np):.5f}, max={np.max(audio_np):.5f}, mean={np.mean(audio_np):.5f}, std={np.std(audio_np):.5f}")
        # print(f"  has_nan={np.isnan(audio_np).any()}, has_inf={np.isinf(audio_np).any()}")
        # print(f"  abs>1 fraction={(np.abs(audio_np) > 1).mean():.6f}")

        audio = audio[: sr * 30]  # truncate audio to at most 30s

        # # ======== 截断后信息 ========
        # print("[DEBUG after truncation]")
        # print(f"  final_len={len(audio)}, min={np.min(audio_np):.5f}, max={np.max(audio_np):.5f}")

        spectrogram = self.wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze()
        text = ann["text"]
        task = ann.get("task", "asr")
        Q = ann.get("Q", "")

        return {
            "spectrogram": spectrogram,
            "raw_wav": audio,
            "text": text,
            "task": task,
            "Q": Q,
            "id": ann["path"],
        }


class BalancedMultiTaskSampler(torch.utils.data.Sampler):
    """
    A sampler that ensures balanced sampling from multiple tasks in each batch.
    For each batch, it samples approximately equal numbers of samples from each task.
    """
    def __init__(self, dataset, batch_size, num_tasks, shuffle=True, seed=42):
        """
        Args:
            dataset: MultiTaskDataset instance
            batch_size: Batch size
            num_tasks: Number of tasks
            shuffle: Whether to shuffle samples
            seed: Random seed
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_tasks = num_tasks
        self.shuffle = shuffle
        self.seed = seed

        # Calculate samples per task per batch
        self.samples_per_task = batch_size // num_tasks
        if batch_size % num_tasks != 0:
            # If batch_size is not divisible by num_tasks, distribute remainder
            self.samples_per_task_list = [self.samples_per_task + (1 if i < batch_size % num_tasks else 0)
                                          for i in range(num_tasks)]
        else:
            self.samples_per_task_list = [self.samples_per_task] * num_tasks

        # Group indices by task
        self.task_indices = {}
        for task_name in dataset.task_names:
            self.task_indices[task_name] = [i for i, (t, _) in enumerate(dataset.index_mapping) if t == task_name]

        # Calculate number of batches
        self.num_batches = len(dataset) // batch_size
        if len(dataset) % batch_size != 0:
            self.num_batches += 1

    def __iter__(self):
        rng = np.random.RandomState(self.seed)

        # Shuffle indices for each task
        task_indices_shuffled = {}
        for task_name, indices in self.task_indices.items():
            indices_copy = indices.copy()
            if self.shuffle:
                rng.shuffle(indices_copy)
            task_indices_shuffled[task_name] = indices_copy

        # Create batches with balanced sampling
        batch_indices = []
        task_pointers = {task_name: 0 for task_name in self.dataset.task_names}

        for batch_idx in range(self.num_batches):
            batch = []
            for task_idx, task_name in enumerate(self.dataset.task_names):
                samples_needed = self.samples_per_task_list[task_idx]
                indices = task_indices_shuffled[task_name]

                for _ in range(samples_needed):
                    if task_pointers[task_name] >= len(indices):
                        # Reshuffle and restart
                        if self.shuffle:
                            rng.shuffle(indices)
                        task_pointers[task_name] = 0

                    batch.append(indices[task_pointers[task_name]])
                    task_pointers[task_name] += 1

            batch_indices.extend(batch)

        return iter(batch_indices)

    def __len__(self):
        return self.num_batches * self.batch_size