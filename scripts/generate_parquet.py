# -*- coding: utf-8 -*-
"""
解析数据并生成 parquet。
1. 支持为每个数据集设定不同的最大采样数量（max_samples）。
2. 支持通过重复采样(Oversampling)强制达到目标异常比例。
"""

import pandas as pd
from pathlib import Path
import random
from typing import List, Optional, Dict

# ==========================================================
# 硬编码配置 1：每个数据集的最小异常样本比例 (Minimum Ratio)
# 注意：这是最低比例，实际采样时异常比例可以高于此值，但不能低于此值
# ==========================================================
ANOMALY_RATIO_CONFIG = {
    "YAHOO": 0.0,
    "ECG": 0.0,
    "IOPS": 0.0,
    "SVDB": 0.0,
    "TODS": 0.0,
    "WSD": 0.0,
    "DEFAULT": 0.0
}

# ==========================================================
# 硬编码配置 2：每个数据集的最大采样数量 (Max Samples)
# ==========================================================
DOMAIN_MAX_SAMPLES = {
    "YAHOO": 1200,
    "IOPS": 1000,
    "TODS": 10000,
    "WSD": 100000,
    "DEFAULT": 0 
}

def is_anomaly_folder(folder: Path) -> bool:
    """check the csv"""
    gt_path = folder / "ground_truth.csv"
    if not gt_path.exists():
        return False
    try:
        df_gt = pd.read_csv(gt_path)
        return len(df_gt) > 0
    except Exception:
        return False

def parse_single_domain(
    input_dir: Path, 
    pattern: str = "*_seg*",
    domain_name: Optional[str] = None,
    max_samples: int = 500,
    min_ratio: float = 0.3,
    seed: Optional[int] = None
) -> List[dict]:
    if seed is not None:
        random.seed(seed)
    
    domain_name = domain_name or input_dir.name
    print(f"\n扫描域: {domain_name} | 采样上限: {max_samples} | 最小异常比例: {min_ratio:.1%}")
    
    all_folders = sorted([f for f in input_dir.iterdir() if f.is_dir() and f.match(pattern)])
    
    anomaly_pool = []
    normal_pool = []

    for folder in all_folders:
        if not (folder / "segment_data.csv").exists() or not (folder / "ground_truth.csv").exists():
            continue
        if is_anomaly_folder(folder):
            anomaly_pool.append(folder)
        else:
            normal_pool.append(folder)

    if not anomaly_pool and not normal_pool:
        print(f"  ⚠️ {domain_name} 未发现有效数据")
        return []

    # 1. 确定最终要采样的总数（取目录实际总量与硬编码上限的最小值）
    total_to_sample = min(len(all_folders), max_samples)
    
    # 2. 计算最小异常数（确保异常比例至少达到 min_ratio）
    min_anomaly_needed = int(total_to_sample * min_ratio)
    
    sampled_anomaly = []
    
    # --- 采样异常样本（确保至少达到最小比例，但可以使用更多）---
    if len(anomaly_pool) > 0:
        if len(anomaly_pool) >= min_anomaly_needed:
            # 异常池足够，使用所有可用的异常样本（在 total_to_sample 范围内）
            # 这样可以充分利用数据，异常比例可能高于最小比例，这是被允许的
            num_anomaly_to_sample = min(len(anomaly_pool), total_to_sample)
            sampled_anomaly = random.sample(anomaly_pool, num_anomaly_to_sample)
        else:
            # 异常池不足，需要重复采样以达到最小比例要求
            sampled_anomaly = list(anomaly_pool)
            shortage = min_anomaly_needed - len(anomaly_pool)
            sampled_anomaly.extend(random.choices(anomaly_pool, k=shortage))
            print(f"  💡 {domain_name}: 异常不足, 重复采样补齐至 {min_anomaly_needed}（最小比例要求）")
    else:
        print(f"  ❌ 警告: {domain_name} 无异常样本，无法满足最小异常比例要求")

    # 3. 填充正常样本（剩余的空位）
    remaining_slots = max(0, total_to_sample - len(sampled_anomaly))
    if len(normal_pool) >= remaining_slots:
        sampled_normal = random.sample(normal_pool, remaining_slots)
    else:
        if len(normal_pool) > 0:
            sampled_normal = random.choices(normal_pool, k=remaining_slots)
        else:
            sampled_normal = []

    final_list = sampled_anomaly + sampled_normal
    random.shuffle(final_list)
    
    # 计算实际异常比例
    actual_ratio = len(sampled_anomaly) / len(final_list) if len(final_list) > 0 else 0.0
    print(f"  采样结果: 总计={len(final_list)} (异常={len(sampled_anomaly)}, 正常={len(sampled_normal)}, 实际异常比例={actual_ratio:.1%})")

    return [{
        "segment_folder": str(folder.resolve()),
        "domain": domain_name,
        "has_anomaly": folder in anomaly_pool
    } for folder in final_list]

def parse_all_preprocessed_data(
    input_dirs: List[Path],
    output_parquet_path: Path,
    seed: Optional[int] = None
):
    if seed is not None:
        random.seed(seed)
    
    print("=" * 80)
    print("跨域数据解析 (动态上限 + 最小异常比例保证)")
    print("=" * 80)
    
    all_data = []

    for input_dir in input_dirs:
        if not input_dir.exists(): continue
        
        domain_name = input_dir.name
        
        # 获取该域特定的配置，如果没有则使用 DEFAULT
        min_ratio = ANOMALY_RATIO_CONFIG.get(domain_name, ANOMALY_RATIO_CONFIG["DEFAULT"])
        limit = DOMAIN_MAX_SAMPLES.get(domain_name, DOMAIN_MAX_SAMPLES["DEFAULT"])
        
        domain_data = parse_single_domain(
            input_dir=input_dir,
            domain_name=domain_name,
            max_samples=limit,
            min_ratio=min_ratio,
            seed=seed
        )
        
        if domain_data:
            all_data.extend(domain_data)

    if not all_data: return

    df = pd.DataFrame(all_data)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet_path, index=False, engine='pyarrow')

    print("\n" + "=" * 80)
    # 统计最终结果
    final_stats = df.groupby('domain')['has_anomaly'].agg(['count', 'sum'])
    final_stats.columns = ['Total_Sampled', 'Original_Anomaly_Count']
    final_stats['Final_Ratio'] = (final_stats['Original_Anomaly_Count'] / final_stats['Total_Sampled'] * 100).map('{:.1f}%'.format)
    print(final_stats)
    print(f"\n成功生成: {output_parquet_path} | 总样本: {len(df)}")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="./unsample.parquet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # 自动搜索 ./TrainData
    train_path = Path("./TrainData")
    dirs = [d for d in train_path.iterdir() if d.is_dir()] if train_path.exists() else []

    parse_all_preprocessed_data(dirs, Path(args.output), seed=args.seed)