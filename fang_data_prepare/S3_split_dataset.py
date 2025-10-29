import json
import random
import re
from pathlib import Path
from typing import Dict, List

def canonical_id_from_path(p: str) -> str:
    """从路径中提取统一样本ID：取文件名去后缀，再截断到下划线/空格之前，再抽取前缀字母+数字块。"""
    name = Path(p).stem  # e.g., BL01_ENSS
    # 先按下划线或空格截断
    name = re.split(r'[_\s]+', name)[0]
    # 抽取前缀 字母+数字（如 BL01 / LA2004）
    m = re.match(r'([A-Za-z]+[0-9]+)', name)
    return m.group(1) if m else name

def load_json_as_map(json_path: Path) -> Dict[str, dict]:
    """读取一个 json，返回 {sample_id -> annotation_item}，同一ID若有多条保留第一条。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("annotation", [])
    m = {}
    for it in items:
        sid = canonical_id_from_path(it["path"])
        if sid not in m:
            m[sid] = it
    return m

def split_ids(common_ids: List[str], train_ratio=0.8, val_ratio=0.1, seed=42):
    """按 8:1:1 划分 ID，余数给 train。"""
    ids = list(common_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]
    return train_ids, val_ids, test_ids

def dump_split(task_map: Dict[str, dict], id_list: List[str], out_path: Path):
    anns = [task_map[i] for i in id_list if i in task_map]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"annotation": anns}, f, ensure_ascii=False, indent=2)
    print(f"[OK] {out_path.name}: {len(anns)} items")

def consistent_split(
    breath_json: str,
    rough_json: str,
    strain_json: str,
    out_dir: str,
    seed: int = 42,
):
    out_dir = Path(out_dir)
    # 读三个任务
    bmap = load_json_as_map(Path(breath_json))
    rmap = load_json_as_map(Path(rough_json))
    smap = load_json_as_map(Path(strain_json))

    # 取三者共同的样本ID，确保一致划分
    common_ids = sorted(set(bmap) & set(rmap) & set(smap))
    print(f"breath={len(bmap)}, rough={len(rmap)}, strain={len(smap)}, common={len(common_ids)}")

    # 划分 8:1:1
    train_ids, val_ids, test_ids = split_ids(common_ids, 0.8, 0.1, seed=seed)
    print(f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # 为三个任务分别写出一致的 split
    dump_split(bmap, train_ids, out_dir / "breathiness_train.json")
    dump_split(bmap, val_ids,   out_dir / "breathiness_val.json")
    dump_split(bmap, test_ids,  out_dir / "breathiness_test.json")

    dump_split(rmap, train_ids, out_dir / "roughness_train.json")
    dump_split(rmap, val_ids,   out_dir / "roughness_val.json")
    dump_split(rmap, test_ids,  out_dir / "roughness_test.json")

    dump_split(smap, train_ids, out_dir / "strain_train.json")
    dump_split(smap, val_ids,   out_dir / "strain_val.json")
    dump_split(smap, test_ids,  out_dir / "strain_test.json")

# ============== 使用示例 ==============
consistent_split(
    breath_json="/scratch/project_2014832/fang/code/SALMONN/fang_data_prepare/all_pvqd_json/breathiness_evaluation_onlyscore.json",
    rough_json="/scratch/project_2014832/fang/code/SALMONN/fang_data_prepare/all_pvqd_json/roughness_evaluation_onlyscore.json",
    strain_json="/scratch/project_2014832/fang/code/SALMONN/fang_data_prepare/all_pvqd_json/strain_evaluation_onlyscore.json",
    out_dir="/scratch/project_2014832/fang/code/SALMONN/fang_data_prepare/splits_pvqd",
    seed=2025,   # 可改，保证可复现
)