import json
import re
from pathlib import Path
import pandas as pd

def canonical_key(s: str) -> str:
    """把文件名/ID规范化：只保留字母数字，转大写。"""
    return re.sub(r'[^A-Za-z0-9]+', '', s).upper()

def infer_task_from_excel_name(xlsx_path: Path) -> str:
    name = xlsx_path.name.lower()
    if 'breath' in name:
        return 'breathiness_evaluation_onlyscore'
    if 'rough' in name:
        return 'roughness_evaluation_onlyscore'
    if 'strain' in name:
        return 'strain_evaluation_onlyscore'
    raise ValueError(f"无法从文件名推断任务: {xlsx_path.name}")

def load_score_map(xlsx_path: Path) -> dict:
    """
    读取单个 CAPE-V xlsx，返回 {canon_key -> avg_value(str)}。
    要求至少含列：'File' 与 'Average Values'（区分大小写不严格）。
    """
    df = pd.read_excel(xlsx_path)
    # 容错列名
    cols = {c.lower(): c for c in df.columns}
    file_col = cols.get('file')
    avg_col = cols.get('average values') or cols.get('average_values') or cols.get('average')
    if not file_col or not avg_col:
        raise ValueError(f"{xlsx_path} 缺少 'File' 或 'Average Values' 列，实际列名：{list(df.columns)}")

    score_map = {}
    for _, row in df.iterrows():
        fid = str(row[file_col]).strip()
        if fid == '' or fid.lower() == 'nan':
            continue
        key = canonical_key(fid)
        val = row[avg_col]
        # 只保留数字（表里可能有文本/空）
        if pd.isna(val):
            continue
        try:
            num = float(val)
            score_map[key] = f"{num}"  # 以字符串形式写入，原始尺度（如 0-100）
        except Exception:
            # 忽略不可转数字的项
            continue
    return score_map

def build_annotations_for_task(wav_dir: Path, score_map: dict, task_name: str,
                               include_missing: bool = False, missing_value: str = ""):
    """
    为一个任务构建 annotation 列表。
    include_missing=True 时，没分数的样本也写入，text=missing_value。
    """
    ann = []
    for wav in sorted(wav_dir.rglob("*.wav")):
        base = wav.stem  # 例如 BL01_ENSS
        key = canonical_key(base)  # 例如 BL01ENSS -> BL01ENSS
        # 进一步尝试：去掉常见尾缀（如 ENSS、_ENSS、-ENSS、空格后缀）
        # 同时尝试取前缀（连续的字母数字块开头）
        candidates = {key}
        # 去掉常见的 ENSS / ESS / SNSS 等尾部字母块（最多一次）
        candidates.add(re.sub(r'(ENSS|ESS|SNS|SNSP|NS|WAV)$', '', key))
        # 取最前面的字母数字块作为主ID（如 BL01、LA2004）
        m = re.match(r'([A-Z]+[0-9]+)', key)
        if m:
            candidates.add(m.group(1))

        found_val = None
        for ck in candidates:
            if ck in score_map:
                found_val = score_map[ck]
                break
        if found_val is None:
            if include_missing:
                found_val = missing_value
            else:
                continue  # 没找到就跳过该任务的这个样本

        ann.append({
            "path": str(wav.resolve()),
            "task": task_name,
            "text": f"{found_val}"
        })
    return ann

def make_capev_jsons(wav_dir: str, excel_files: list, out_dir: str,
                     include_missing: bool = False, missing_value: str = ""):
    """
    wav_dir: 含 .wav 的目录
    excel_files: 3 个 xlsx 路径（breathiness/roughness/strain），顺序不限
    out_dir: 导出目录
    include_missing: 是否把没有匹配到的样本也写入（text 为空或指定占位值）
    """
    wav_dir = Path(wav_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for xlsx in map(Path, excel_files):
        task = infer_task_from_excel_name(xlsx)
        score_map = load_score_map(xlsx)
        annotations = build_annotations_for_task(
            wav_dir, score_map, task,
            include_missing=include_missing, missing_value=missing_value
        )
        data = {"annotation": annotations}
        out_path = out_dir / f"{task}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[OK] {xlsx.name} -> {out_path}  共 {len(annotations)} 条")

# ========= 使用示例 =========
make_capev_jsons(
    wav_dir="/Users/kangfang/work/research/Kang/data/PerceptualVoiceQualitiesDatabase/AudioFiles",
    excel_files=[
        "/Users/kangfang/work/research/Kang/data/PerceptualVoiceQualitiesDatabase/Ratings Spreadsheets/cape_v_breathiness_only.xlsx",
        "/Users/kangfang/work/research/Kang/data/PerceptualVoiceQualitiesDatabase/Ratings Spreadsheets/cape_v_roughness_only.xlsx",
        "/Users/kangfang/work/research/Kang/data/PerceptualVoiceQualitiesDatabase/Ratings Spreadsheets/cape_v_strain_only.xlsx",
    ],
    out_dir="./all_pvqd_json",
    include_missing=False  # 若希望每个 wav 都出现，即使没分数，也可改为 True
)