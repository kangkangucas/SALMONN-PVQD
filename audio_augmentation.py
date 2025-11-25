"""
Audio Augmentation Module for Voice Quality Assessment
Implements various augmentation techniques while preserving voice quality characteristics
"""

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.io import wavfile
import librosa
import pyloudnorm


class AudioAugmentor:
    """
    Audio augmentation class that applies various transformations to audio
    while preserving voice quality characteristics (breathiness, roughness, strain)
    """
    
    def __init__(self, sr=16000, seed=None):
        """
        Args:
            sr: Sampling rate (default 16000 Hz)
            seed: Random seed for reproducibility
        """
        self.sr = sr
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize loudness meter for loudness normalization
        self.meter = pyloudnorm.Meter(sr)
    
    def apply_augmentation(self, audio, augmentation_type='random', prob=0.5):
        """
        Apply augmentation to audio

        Args:
            audio: Input audio array
            augmentation_type: Type of augmentation ('random', 'speed', 'loudness', 'crop', 'noise', 'reverb')
            prob: Probability of applying augmentation (0-1)

        Returns:
            Augmented audio array
        """
        
        if np.random.random() > prob:
            # print("No apply_augmentation!!!")
            return audio
        
        # Check for invalid input
        if np.isnan(audio).any() or np.isinf(audio).any():
            print("Warning: Input audio contains NaN or Inf values")
            return audio

        if augmentation_type == 'random':
            # Randomly select an augmentation method
            methods = ['speed', 'loudness', 'crop', 'noise', 'reverb']
            augmentation_type = np.random.choice(methods)
            print(f"Random augmentation type: {augmentation_type}")
        try:
            if augmentation_type == 'speed':
                result = self.time_stretch(audio)
            elif augmentation_type == 'loudness':
                result = self.loudness_augmentation(audio)
            elif augmentation_type == 'crop':
                result = self.random_crop_with_vad(audio)
            elif augmentation_type == 'noise':
                result = self.add_noise(audio)
            elif augmentation_type == 'reverb':
                result = self.add_reverb(audio)
            else:
                result = audio


            if augmentation_type != 'crop':  # 增加crop的概率
                if np.random.random() > 0.5:
                    result = self.random_crop_with_vad(result)
                    

            # Check output for NaN/Inf
            if np.isnan(result).any() or np.isinf(result).any():
                print(f"Warning: Augmentation {augmentation_type} produced NaN/Inf, returning original audio")
                return audio

            # Ensure values are in reasonable range
            result = np.clip(result, -1.0, 1.0)
            return result

        except Exception as e:
            print(f"Error in augmentation {augmentation_type}: {e}")
            return audio
    
    def time_stretch(self, audio, rate_range=(0.9, 1.1)):
        """
        Change speech speed without changing pitch

        Args:
            audio: Input audio array
            rate_range: Tuple of (min_rate, max_rate) for time stretching

        Returns:
            Time-stretched audio
        """
        # Ensure audio is in valid range before stretching
        audio = np.clip(audio, -1.0, 1.0)

        rate = np.random.uniform(rate_range[0], rate_range[1])
        try:
            stretched = librosa.effects.time_stretch(audio, rate=rate)

            # Ensure output is valid
            if stretched is None or len(stretched) == 0:
                print(f"Warning: time_stretch produced empty output, returning original audio")
                return audio

            if np.isnan(stretched).any() or np.isinf(stretched).any():
                print(f"Warning: time_stretch produced NaN/Inf, returning original audio")
                return audio

            # Clip to prevent extreme values
            stretched = np.clip(stretched, -1.0, 1.0)


            # Final validation
            if np.isnan(stretched).any() or np.isinf(stretched).any():
                print(f"Warning: Clipped audio still contains NaN/Inf, returning original")
                return audio

            return stretched
        except Exception as e:
            print(f"Error in time_stretch: {e}")
            return audio
    
    def loudness_augmentation(self, audio, loudness_range=(-5, 5)):
        """
        Change loudness while preserving spectral characteristics

        Args:
            audio: Input audio array
            loudness_range: Tuple of (min_loudness, max_loudness) in LUFS

        Returns:
            Loudness-adjusted audio
        """
        try:
            # Suppress pyloudnorm warnings about clipped samples
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)

                current_loudness = self.meter.integrated_loudness(audio)
                if np.isnan(current_loudness) or np.isinf(current_loudness):
                    # If loudness measurement fails, use simple amplitude scaling
                    return self.amplitude_scaling(audio, loudness_range)

                target_loudness = np.random.uniform(loudness_range[0], loudness_range[1])
                # Fix: Use pyloudnorm.normalize.loudness() with correct API
                normalized = pyloudnorm.normalize.loudness(
                    audio, current_loudness, target_loudness
                )

            # Check for NaN/Inf in normalized audio
            if np.isnan(normalized).any() or np.isinf(normalized).any():
                print("Warning: pyloudnorm.normalize produced NaN/Inf")
                return self.amplitude_scaling(audio, loudness_range)

            # Clip to prevent extreme values (more aggressive clipping)
            normalized = np.clip(normalized, -1.0, 1.0)

            # Final check after clipping
            if np.isnan(normalized).any() or np.isinf(normalized).any():
                print("Warning: Audio contains NaN/Inf after clipping")
                return audio

            return normalized
        except Exception as e:
            print(f"Error in loudness_augmentation: {e}")
            # Fallback to simple amplitude scaling
            return self.amplitude_scaling(audio, loudness_range)
    
    def amplitude_scaling(self, audio, loudness_range=(-5, 5)):
        """
        Simple amplitude scaling as fallback

        Args:
            audio: Input audio array
            loudness_range: Tuple of (min_scale, max_scale) in dB

        Returns:
            Scaled audio
        """
        scale_db = np.random.uniform(loudness_range[0], loudness_range[1])
        scale_linear = 10 ** (scale_db / 20.0)

        # Check for invalid scale
        if np.isnan(scale_linear) or np.isinf(scale_linear):
            print(f"Warning: Invalid scale_linear {scale_linear}, returning original audio")
            return audio

        scaled = audio * scale_linear

        # Check for NaN/Inf
        if np.isnan(scaled).any() or np.isinf(scaled).any():
            print(f"Warning: amplitude_scaling produced NaN/Inf, returning original audio")
            return audio

        # Clip to prevent clipping
        scaled = np.clip(scaled, -1.0, 1.0)
        return scaled
    
    def random_crop_with_vad(self, audio, min_duration=1, max_duration=None):
        """
        Randomly crop audio with Voice Activity Detection (VAD)
        Ensures we don't crop to silence

        Args:
            audio: Input audio array
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds (None = use full audio)

        Returns:
            Cropped audio
        """
        min_samples = int(min_duration * self.sr)

        if max_duration is None:
            max_duration = len(audio) / self.sr
        max_samples = int(max_duration * self.sr)

        if len(audio) <= min_samples:
            return audio

        # Check if audio is all silence
        if np.max(np.abs(audio)) < 1e-6:
            # All silence, return original audio
            return audio

        # Simple VAD: detect non-silent frames
        # Use energy-based VAD
        frame_length = int(0.025 * self.sr)  # 25ms frames
        hop_length = int(0.010 * self.sr)    # 10ms hop

        energy = np.array([
            np.sum(audio[i:i+frame_length]**2)
            for i in range(0, len(audio)-frame_length, hop_length)
        ])

        # Threshold: 40% of max energy
        threshold = 0.4 * np.max(energy)
        active_frames = np.where(energy > threshold)[0]

        if len(active_frames) == 0:
            # If no active frames, just do random crop
            start = np.random.randint(0, len(audio) - min_samples)
            end = min(start + max_samples, len(audio))
            return audio[start:end]

        # Find continuous active regions
        active_start = active_frames[0] * hop_length
        active_end = active_frames[-1] * hop_length + frame_length

        # Ensure we have enough active audio
        active_duration = (active_end - active_start) / self.sr
        if active_duration < min_duration:
            return audio  # Not enough active audio

        # Random crop within active region
        crop_length = np.random.randint(min_samples, min(max_samples, active_end - active_start))
        max_start = active_end - crop_length
        start = np.random.randint(active_start, max(active_start + 1, max_start))
        end = min(start + crop_length, len(audio))

        cropped = audio[start:end]

        # Ensure cropped audio is not all silence
        if np.max(np.abs(cropped)) < 1e-6:
            # If cropped audio is all silence, return original
            return audio

        return cropped
    
    def add_noise(self, audio, snr_range=(5, 20)):
        """
        Add background noise to audio

        Args:
            audio: Input audio array
            snr_range: Tuple of (min_SNR, max_SNR) in dB

        Returns:
            Noisy audio
        """
        snr = np.random.uniform(snr_range[0], snr_range[1])

        # Generate white noise
        noise = np.random.normal(0, 1, len(audio))

        # Calculate signal and noise power
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)

        # Avoid division by zero
        if noise_power < 1e-10:
            print("Warning: noise_power too small, returning original audio")
            return audio

        # Scale noise to achieve target SNR
        snr_linear = 10 ** (snr / 10.0)
        noise_scaled = noise * np.sqrt(signal_power / (snr_linear * noise_power))

        # Check for NaN/Inf
        if np.isnan(noise_scaled).any() or np.isinf(noise_scaled).any():
            print("Warning: add_noise produced NaN/Inf, returning original audio")
            return audio

        noisy = audio + noise_scaled

        # Check for NaN/Inf
        if np.isnan(noisy).any() or np.isinf(noisy).any():
            print("Warning: noisy audio contains NaN/Inf, returning original audio")
            return audio

        noisy = np.clip(noisy, -1.0, 1.0)
        return noisy
    
    def add_reverb(self, audio, room_scale_range=(0.1, 0.5)):
        """
        Add simple reverb effect using convolution

        Args:
            audio: Input audio array
            room_scale_range: Tuple of (min_scale, max_scale) for reverb intensity

        Returns:
            Reverb-added audio
        """
        room_scale = np.random.uniform(room_scale_range[0], room_scale_range[1])
        g0 = np.random.uniform(0.6, 0.9)  # 直达分量（越大越“靠前”）
        # 2) 生成更自然的尾巴：指数衰减 * 随机扩散（正负交替，避免成低通平均）
        ir_len = int(0.5 * self.sr)  # ≈500 ms
        t = np.arange(ir_len, dtype=np.float32) / float(self.sr)
        tau = room_scale * 0.35  # 衰减常数，可微调（单位：秒）
        decay = np.exp(-t / max(tau, 1e-4))

        # 随机扩散：高斯+轻微高通让反射更“亮”，避免纯低通
        rng = np.random.default_rng()
        noise = rng.normal(0.0, 1.0, size=ir_len)
        # 简单一阶高通（去点直流，减少“糊”）
        alpha = 0.95
        for n in range(1, ir_len):
            noise[n] = noise[n] - alpha * noise[n-1]

        tail = decay * noise

        # 3) 直达 + 尾巴（仍是“纯卷积”的 IR）
        ir = np.zeros(ir_len + 1, dtype=np.float32)
        ir[0] = g0                                   # 直达分量
        ir[1:] = tail * (1.0 - g0)                   # 尾巴占 (1 - g0)，防止直达过小

        # # --- 用 L2 归一化（功率增益≈1），而不是按和归一 ---
        eps = 1e-12
        ir_energy = float(np.sum(ir * ir))  # == ||h||_2^2
        if ir_energy < 1e-20 or np.isnan(ir_energy) or np.isinf(ir_energy):
            print("Warning: Invalid IR energy, returning original audio")
            return audio
        ir = ir / np.sqrt(ir_energy + eps)

        # Convolve with audio
        try:
            # === 卷积：full + 中心裁剪，避免边界能量损失 ===
            full = signal.fftconvolve(audio, ir, mode='full')
            start = (len(ir) - 1) // 2
            reverb = full[start:start + len(audio)]

            # 方案B：分位数峰值匹配（推荐，听感更稳）
            p = 99.5  # 你可以试 99, 99.5, 99.9
            in_peak_q  = float(np.percentile(np.abs(audio), p) + eps)
            out_peak_q = float(np.percentile(np.abs(reverb), p) + eps)
            gain = in_peak_q / out_peak_q
            
            # 可选：留一点头间隙，防止刚好顶到 1.0（例如 -0.5 dB）
            headroom = 10 ** (-0.5 / 20.0)  # ≈ 0.944
            reverb = reverb * gain * headroom
            # 最后保证不爆
            maxabs = float(np.max(np.abs(reverb))) if reverb.size else 0.0
            if maxabs > 1.0:
                reverb = reverb / maxabs  # 或者 np.clip(reverb, -1.0, 1.0)
                
            # Check for NaN/Inf
            if np.isnan(reverb).any() or np.isinf(reverb).any():
                print("Warning: add_reverb produced NaN/Inf, returning original audio")
                return audio

            reverb = np.clip(reverb, -1.0, 1.0)
            return reverb
        except Exception as e:
            print(f"Error in add_reverb: {e}")
            return audio


def augment_audio_file(input_path, output_path, augmentation_type='random', prob=0.5, sr=16000):
    """
    Convenience function to augment an audio file
    
    Args:
        input_path: Path to input audio file
        output_path: Path to save augmented audio
        augmentation_type: Type of augmentation
        prob: Probability of applying augmentation
        sr: Sampling rate
    """
    audio, file_sr = sf.read(input_path)
    
    # Convert to mono if stereo
    if len(audio.shape) == 2:
        audio = audio[:, 0]
    
    # Resample if necessary
    if file_sr != sr:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
    
    augmentor = AudioAugmentor(sr=sr)
    augmented = augmentor.apply_augmentation(audio, augmentation_type, prob)
    
    sf.write(output_path, augmented, sr)
    print(f"Augmented audio saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    augmentor = AudioAugmentor(sr=16000)
    
    # Load example audio
    audio, sr = sf.read("example.wav")
    
    # Apply different augmentations
    augmented_speed = augmentor.time_stretch(audio)
    augmented_loudness = augmentor.loudness_augmentation(audio)
    augmented_crop = augmentor.random_crop_with_vad(audio)
    augmented_noise = augmentor.add_noise(audio)
    augmented_reverb = augmentor.add_reverb(audio)
    
    print("Augmentation examples created!")

