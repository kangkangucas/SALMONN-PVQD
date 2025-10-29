import os
from pathlib import Path
import librosa
import soundfile as sf
from tqdm import tqdm
import re

def canonical_id(name):
    """
    从文件名中提取前面的字母数字编号部分。
    例如:
      BL01_ENSS -> BL01
      LA1002ENSS -> LA1002
      BL01 ENSS -> BL01
    """
    # 去掉下划线及空格后部分
    name = re.split(r'[_\s]+', name)[0]
    # 再从前面提取字母+数字的模式
    m = re.match(r'([A-Za-z]+[0-9]+)', name)
    if m:
        return m.group(1)
    return name  # fallback
    
def resample_and_rename_wavs(src_dir, dst_dir, target_sr=16000):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    wav_files = list(src_dir.glob("*.wav"))

    for wav_path in tqdm(wav_files, desc="Processing"):
        # 读取音频
        audio, sr = librosa.load(wav_path, sr=None, mono=True)  # 强制读单声道
        # 重采样
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        # 新文件名
        base = wav_path.stem
        new_name = canonical_id(base) + ".wav"
        out_path = dst_dir / new_name

        # 保存
        sf.write(out_path, audio, target_sr)

    print(f"\n✅ Done! Resampled & renamed wavs saved to: {dst_dir}")


# ================= 使用示例 =================
# 修改为你自己的路径：
resample_and_rename_wavs(
    src_dir="/scratch/project_2011211/fang/data/PerceptualVoiceQualitiesDatabase/AudioFiles",
    dst_dir="/scratch/project_2011211/fang/data/PerceptualVoiceQualitiesDatabase/AudioFiles_16k"
)