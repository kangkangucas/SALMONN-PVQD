#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Usage:
# python compute_nasalization_index.py --input_dir /scratch/project_2011211/fang/data/DH-FaceVid-1K_extracted_wav  --output_csv brightness_nasalization_lables_dh_facevid_1k_all.csv
"""
Compute nasalization- and brightness-related acoustic features from WAV files.

输出（按列）：
    wav_path
    A1P0_dB_med, B1_Hz_med, F1_Hz_med, LF_ratio_dB_med, Tilt_dB_med
    S_A1P0, S_B1, S_LF, S_Tilt, nasal_score_0_100
    S_HF, S_SC, brightness_0_100

说明：
    - 所有 S_* 和 *_0_100 均为 0–100，数值越大表示“越鼻 / 越亮”
    - 不依赖任何特定数据库的经验上界，只基于能量/幅度比和频率范围
"""

import os
import csv
import argparse
from typing import List, Optional, Tuple, Dict

import numpy as np
import soundfile as sf
import parselmouth
from tqdm import tqdm
from scipy.signal import get_window


# =========================================================
# 基础工具
# =========================================================

def load_audio_mono(path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """读取 wav，转单声道并可选重采样到 target_sr。"""
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if target_sr is not None and sr != target_sr:
        import scipy.signal
        ns = int(round(len(y) * float(target_sr) / sr))
        y = scipy.signal.resample(y, ns)
        sr = target_sr
    return y.astype(np.float64), sr


def pre_emphasis(x: np.ndarray, pre: float = 0.97) -> np.ndarray:
    """简单预加重。"""
    if len(x) < 2:
        return x.copy()
    out = np.empty_like(x)
    out[0] = x[0]
    out[1:] = x[1:] - pre * x[:-1]
    return out


def compute_spectrum(frame: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """对一帧做 Hamming + rFFT，返回 (freqs, mag)。"""
    frame = frame * get_window("hamming", len(frame), fftbins=True)
    spec = np.fft.rfft(frame)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(len(frame), d=1.0 / sr)
    return freqs, mag


def band_energy(mag: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> float:
    """频带能量（功率）。"""
    mask = (freqs >= fmin) & (freqs < fmax)
    if not np.any(mask):
        return 0.0
    band = mag[mask]
    return float(np.sum(band ** 2))


def db_ratio(num: float, den: float, eps: float = 1e-12) -> float:
    """10*log10(num/den) 带防零。"""
    num = max(num, eps)
    den = max(den, eps)
    return 10.0 * np.log10(num / den)


# =========================================================
# Formant & 单帧特征
# =========================================================

def compute_formant_object(sound: parselmouth.Sound,
                           time_step: float = 0.01,
                           max_formant: float = 5000.0) -> parselmouth.Formant:
    """使用 Praat Burg 算法计算 formant 轨迹。"""
    formant = sound.to_formant_burg(
        time_step=time_step,
        max_number_of_formants=5,
        maximum_formant=max_formant,
        window_length=0.025,
        pre_emphasis_from=50.0,
    )
    return formant


def get_frame_window(sound: parselmouth.Sound,
                     center_time: float,
                     win_len: float) -> np.ndarray:
    """从 Sound 中提取中心在 center_time 的窗长为 win_len 的帧。"""
    start = max(0.0, center_time - win_len / 2.0)
    end = min(sound.get_total_duration(), center_time + win_len / 2.0)
    sub = sound.extract_part(from_time=start, to_time=end,
                             preserve_times=False)
    return sub.values[0].copy()  # mono


def compute_frame_features(sound: parselmouth.Sound,
                           sr: int,
                           formant: parselmouth.Formant,
                           time: float,
                           win_len: float) -> Optional[Dict[str, float]]:
    """
    对单帧计算：
        A1P0_dB, B1_Hz, F1_Hz, LF_ratio_dB, Tilt_dB
        HF_ratio（高频/低频能量比）、SC_Hz（谱质心）
    若该帧不可用则返回 None。
    """
    # ---- F1 & B1 ----
    f1 = formant.get_value_at_time(1, time)
    b1 = formant.get_bandwidth_at_time(1, time)
    if np.isnan(f1) or np.isnan(b1) or f1 <= 0 or b1 <= 0:
        return None

    # ---- 取帧 & 频谱 ----
    frame = get_frame_window(sound, time, win_len)
    if len(frame) < 16:
        return None
    frame = pre_emphasis(frame)
    freqs, mag = compute_spectrum(frame, sr)
    ny = sr / 2.0

    # ---- A1：在 F1 ± B1/2 区间找最大谱峰 ----
    bw_half = b1 / 2.0
    f_low_a1 = max(0.0, f1 - bw_half)
    f_high_a1 = f1 + bw_half
    mask_a1 = (freqs >= f_low_a1) & (freqs <= f_high_a1)
    if not np.any(mask_a1):
        return None
    A1_mag = float(np.max(mag[mask_a1]))
    A1_dB = 20.0 * np.log10(max(A1_mag, 1e-12))

    # ---- P0：250–450 Hz 区间最大谱峰 ----
    P0_low = 250.0
    P0_high = 450.0
    if P0_low >= ny:
        return None
    P0_high = min(P0_high, ny * 0.9)
    mask_p0 = (freqs >= P0_low) & (freqs <= P0_high)
    if not np.any(mask_p0):
        return None
    P0_mag = float(np.max(mag[mask_p0]))
    P0_dB = 20.0 * np.log10(max(P0_mag, 1e-12))
    A1P0_dB = A1_dB - P0_dB

    # ---- LF ratio：0–320 vs 320–5360 Hz ----
    lf_energy = band_energy(mag, freqs, 0.0, 320.0)
    hf_energy = band_energy(mag, freqs, 320.0, min(5360.0, ny))
    LF_ratio_dB = db_ratio(lf_energy, hf_energy)

    # ---- Tilt：0–1000 vs 1000–4000 Hz ----
    low_1000 = band_energy(mag, freqs, 0.0, min(1000.0, ny))
    high_1000_4000 = band_energy(mag, freqs, 1000.0, min(4000.0, ny))
    Tilt_dB = db_ratio(low_1000, high_1000_4000)

    # ---- Brightness: HF_ratio & SC ----
    f_split = 3000.0
    E_lf = band_energy(mag, freqs, 0.0, min(f_split, ny))
    E_hf = band_energy(mag, freqs, f_split, ny)
    HF_ratio = E_hf / (E_lf + 1e-12)

    # spectral centroid
    num = float(np.sum(freqs * mag))
    den = float(np.sum(mag) + 1e-12)
    SC_Hz = num / den

    return {
        "A1P0_dB": A1P0_dB,
        "B1_Hz": float(b1),
        "F1_Hz": float(f1),
        "LF_ratio_dB": LF_ratio_dB,
        "Tilt_dB": Tilt_dB,
        "HF_ratio": HF_ratio,
        "SC_Hz": SC_Hz,
    }


# =========================================================
# 映射到 0–100 分数
# =========================================================

def feature_to_scores_0_100(A1P0_dB_med: float,
                            B1_Hz_med: float,
                            F1_Hz_med: float,
                            LF_ratio_dB_med: float,
                            Tilt_dB_med: float) -> Dict[str, float]:
    """
    把鼻音相关中位数特征映射成 4 个 0–100 分数：
        S_A1P0, S_B1, S_LF, S_Tilt
    全部：越大越鼻。
    通过幂次 gamma>1 放大中间区域差异，让分布更分散。
    """
    eps = 1e-12

    gamma_a1p0 = 2.5   # A1P0 比较集中，多拉一点
    gamma_other = 1.8  # 其他三个适中

    # --- 1) A1P0：A1/P0 的幅度比 ---
    # A1P0_dB = 20 * log10(A1/P0) => R_ap = A1/P0
    R_ap = 10.0 ** (A1P0_dB_med / 20.0)
    R_ap_pow = max(R_ap, eps) ** gamma_a1p0
    # 口音：A1 >> P0 (R大) => S小； 鼻音：A1 << P0 (R小) => S大
    S_A1P0 = 100.0 * (1.0 / (1.0 + R_ap_pow))

    # --- 2) B1/F1 阻尼比 ---
    R_b = max(B1_Hz_med, eps) / max(F1_Hz_med, eps)
    R_b_pow = R_b ** gamma_other
    S_B1 = 100.0 * (R_b_pow / (1.0 + R_b_pow))

    # --- 3) LF_ratio ---
    R_lf = 10.0 ** (LF_ratio_dB_med / 10.0)
    R_lf_pow = R_lf ** gamma_other
    S_LF = 100.0 * (R_lf_pow / (1.0 + R_lf_pow))

    # --- 4) Tilt ---
    R_tilt = 10.0 ** (Tilt_dB_med / 10.0)
    R_tilt_pow = R_tilt ** gamma_other
    S_Tilt = 100.0 * (R_tilt_pow / (1.0 + R_tilt_pow))

    return {
        "S_A1P0": float(S_A1P0),
        "S_B1": float(S_B1),
        "S_LF": float(S_LF),
        "S_Tilt": float(S_Tilt),
    }


def compute_brightness_scores(HF_list: List[float],
                              SC_list: List[float],
                              gamma_hf: float = 2.0,
                              gamma_sc: float = 1.8,
                              f_min: float = 100.0,
                              f_max: float = 8000.0) -> Tuple[float, float, float]:
    """
    根据每帧的 HF_ratio 和 SC_Hz 计算 0–100 Brightness 指标。

    返回：
        S_HF_med       : 高频能量子分（0–100）
        S_SC_med       : 谱质心子分（0–100）
        brightness_0_100 : 综合亮度（0–100，越大越亮）
    """
    eps = 1e-12

    # ---- 1) 高频能量子分 ----
    HF_arr = np.array(HF_list, dtype=np.float64)
    HF_arr = np.maximum(HF_arr, eps)
    HF_pow = HF_arr ** gamma_hf
    S_HF = 100.0 * HF_pow / (1.0 + HF_pow)
    S_HF_med = float(np.median(S_HF))

    # ---- 2) 谱质心子分 ----
    SC_arr = np.array(SC_list, dtype=np.float64)
    SC_norm = (SC_arr - f_min) / (f_max - f_min)
    SC_norm = np.clip(SC_norm, 0.0, 1.0)
    SC_norm_pow = SC_norm ** gamma_sc
    S_SC = 100.0 * SC_norm_pow
    S_SC_med = float(np.median(S_SC))

    # ---- 3) 综合亮度 ----
    w_hf = 0.6
    w_sc = 0.4
    brightness_0_100 = (
        w_hf * S_HF_med +
        w_sc * S_SC_med
    ) / (w_hf + w_sc)

    return S_HF_med, S_SC_med, float(brightness_0_100)


# =========================================================
# 对单个 WAV 计算所有特征
# =========================================================

def compute_features_for_wav(path: str,
                             target_sr: int = 16000,
                             frame_shift: float = 0.01,
                             win_len: float = 0.025,
                             min_frames: int = 20) -> Optional[Dict[str, float]]:
    """对一个 wav 计算鼻音 & 亮度所有指标。"""
    y, sr = load_audio_mono(path, target_sr=target_sr)
    if len(y) < sr * 0.2:
        return None

    snd = parselmouth.Sound(y, sampling_frequency=sr)
    pitch = snd.to_pitch(time_step=frame_shift)
    formant = compute_formant_object(snd, time_step=frame_shift, max_formant=5000.0)

    duration = snd.get_total_duration()
    times = np.arange(0.0, duration, frame_shift)

    A1P0_list: List[float] = []
    B1_list: List[float] = []
    F1_list: List[float] = []
    LF_list: List[float] = []
    Tilt_list: List[float] = []
    HF_list: List[float] = []
    SC_list: List[float] = []

    for t in times:
        f0 = pitch.get_value_at_time(t)
        if np.isnan(f0) or f0 <= 0:
            continue  # 跳过无声帧

        feat = compute_frame_features(snd, sr, formant, t, win_len)
        if feat is None:
            continue

        A1P0_list.append(feat["A1P0_dB"])
        B1_list.append(feat["B1_Hz"])
        F1_list.append(feat["F1_Hz"])
        LF_list.append(feat["LF_ratio_dB"])
        Tilt_list.append(feat["Tilt_dB"])
        HF_list.append(feat["HF_ratio"])
        SC_list.append(feat["SC_Hz"])

    if len(A1P0_list) < min_frames:
        return None

    A1P0_med = float(np.median(A1P0_list))
    B1_med = float(np.median(B1_list))
    F1_med = float(np.median(F1_list))
    LF_med = float(np.median(LF_list))
    Tilt_med = float(np.median(Tilt_list))

    # ---- 鼻音 0–100 子分 ----
    scores = feature_to_scores_0_100(
        A1P0_dB_med=A1P0_med,
        B1_Hz_med=B1_med,
        F1_Hz_med=F1_med,
        LF_ratio_dB_med=LF_med,
        Tilt_dB_med=Tilt_med,
    )
    S_A1P0 = scores["S_A1P0"]
    S_B1 = scores["S_B1"]
    S_LF = scores["S_LF"]
    S_Tilt = scores["S_Tilt"]

    # ---- 综合 nasal_score_0_100 ----
    # 你主观验证 A1P0 与 B1 最重要
    w_a1 = 0.4
    w_b1 = 0.4
    w_lf = 0.1
    w_ti = 0.1

    nasal_score_0_100 = (
        w_a1 * S_A1P0 +
        w_b1 * S_B1 +
        w_lf * S_LF +
        w_ti * S_Tilt
    ) / (w_a1 + w_b1 + w_lf + w_ti)

    # ---- 亮度 0–100 ----
    S_HF, S_SC, brightness_0_100 = compute_brightness_scores(
        HF_list, SC_list
    )

    return {
        "A1P0_dB_med": A1P0_med,
        "B1_Hz_med": B1_med,
        "F1_Hz_med": F1_med,
        "LF_ratio_dB_med": LF_med,
        "Tilt_dB_med": Tilt_med,
        "S_A1P0": S_A1P0,
        "S_B1": S_B1,
        "S_LF": S_LF,
        "S_Tilt": S_Tilt,
        "nasal_score_0_100": float(nasal_score_0_100),
        "S_HF": S_HF,
        "S_SC": S_SC,
        "brightness_0_100": brightness_0_100,
    }


# =========================================================
# 批处理 & CLI
# =========================================================

def find_wavs_in_dir(root: str,
                     ext: str = ".wav",
                     recursive: bool = True) -> List[str]:
    """递归查找目录下所有 wav。"""
    files: List[str] = []
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(ext.lower()):
                    files.append(os.path.join(dirpath, fn))
    else:
        for fn in os.listdir(root):
            if fn.lower().endswith(ext.lower()):
                files.append(os.path.join(root, fn))
    files.sort()
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Compute nasalization and brightness indices from WAV files."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--wav", type=str, help="Path to a single WAV file.")
    group.add_argument("--input_dir", type=str, help="Directory containing WAV files.")
    parser.add_argument("--ext", type=str, default=".wav", help="WAV file extension.")
    parser.add_argument("--no_recursive", action="store_true",
                        help="Do not recurse into subdirectories.")
    parser.add_argument("--target_sr", type=int, default=16000,
                        help="Target sample rate (default: 16000).")
    parser.add_argument("--min_valid_frames", type=int, default=20,
                        help="Minimum voiced frames to keep a file.")
    parser.add_argument("--output_csv", type=str, default="brightness_nasalization_lables_lrs2_train.csv",
                        help="Output CSV filename for directory mode.")
    args = parser.parse_args()

    # ---- 单文件模式 ----
    if args.wav:
        feats = compute_features_for_wav(
            args.wav,
            target_sr=args.target_sr,
            min_frames=args.min_valid_frames,
        )
        if feats is None:
            print(f"{args.wav}\tNA")
        else:
            # 打印为一行 TSV，方便 quick-check
            fields = [
                "A1P0_dB_med", "B1_Hz_med", "F1_Hz_med",
                "LF_ratio_dB_med", "Tilt_dB_med",
                "S_A1P0", "S_B1", "S_LF", "S_Tilt",
                "nasal_score_0_100",
                "S_HF", "S_SC", "brightness_0_100",
            ]
            values = [feats[k] for k in fields]
            print(args.wav + "\t" + "\t".join(f"{v:.4f}" for v in values))
        return

    # ---- 目录模式 ----
    wav_paths = find_wavs_in_dir(
        args.input_dir,
        ext=args.ext,
        recursive=not args.no_recursive,
    )
    if not wav_paths:
        print("No WAV files found.")
        return

    out_csv = args.output_csv
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        # 表头
        writer.writerow([
            "wav_path",
            "A1P0_dB_med", "B1_Hz_med", "F1_Hz_med",
            "LF_ratio_dB_med", "Tilt_dB_med",
            "S_A1P0", "S_B1", "S_LF", "S_Tilt",
            "nasal_score_0_100",
            "S_HF", "S_SC", "brightness_0_100",
        ])

        for path in tqdm(wav_paths):
            feats = compute_features_for_wav(
                path,
                target_sr=args.target_sr,
                min_frames=args.min_valid_frames,
            )
            if feats is None:
                writer.writerow([path] + ["NA"] * 13)
                continue

            writer.writerow([
                path,
                feats["A1P0_dB_med"],
                feats["B1_Hz_med"],
                feats["F1_Hz_med"],
                feats["LF_ratio_dB_med"],
                feats["Tilt_dB_med"],
                feats["S_A1P0"],
                feats["S_B1"],
                feats["S_LF"],
                feats["S_Tilt"],
                feats["nasal_score_0_100"],
                feats["S_HF"],
                feats["S_SC"],
                feats["brightness_0_100"],
            ])

    print(f"CSV saved to: {out_csv}")


if __name__ == "__main__":
    main()
    
    
    