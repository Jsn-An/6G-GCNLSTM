"""
数据预处理脚本：将原始 5G KPI 数据集转换为可用于 GNN 的特征工程版本。

处理步骤：
1. 时间特征提取（hour_of_day, day_of_week, is_peak_hour）
2. 派生特征计算（throughput_per_user, spectral_efficiency）
3. 类别编码（cell_type_enc, slice_type_enc）
4. 数值特征 Min-Max 归一化（*_norm 列）

输入: datasets/dataset1_5g_network_kpi.csv
输出: datasets/ds1_processed.csv
"""
import os
import pandas as pd
import numpy as np

# ============================================================================
# 配置
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RAW_CSV = os.path.join(PROJECT_ROOT, "datasets", "dataset1_5g_network_kpi.csv")
OUT_CSV = os.path.join(PROJECT_ROOT, "datasets", "ds1_processed.csv")


def load_data(path: str) -> pd.DataFrame:
    """加载原始 CSV 并解析时间戳。"""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    print(f"加载数据: {path}")
    print(f"  维度: {df.shape}")
    print(f"  小区数: {df['cell_id'].nunique()}")
    print(f"  时间范围: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """从时间戳提取时间特征。

    - hour_of_day: 0-23，一天中的小时
    - day_of_week: 0-6，周一=0
    - is_peak_hour: 是否高峰时段（8-10, 18-20）
    """
    df = df.copy()
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    # 典型高峰时段：上午 8-10，下午/晚上 18-20
    peak_hours = set(range(8, 11)) | set(range(18, 21))
    df["is_peak_hour"] = df["hour_of_day"].isin(peak_hours).astype(int)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算派生特征。

    - throughput_per_user: 每用户吞吐量 = throughput / max(active_users, 1)
    - spectral_efficiency: 频谱效率 = throughput / max(prb_utilization, 1)
    """
    df = df.copy()
    df["throughput_per_user"] = df["throughput_mbps"] / df["active_users"].clip(lower=1)
    df["spectral_efficiency"] = df["throughput_mbps"] / df["prb_utilization_pct"].clip(lower=1)
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """对类别特征进行标签编码。"""
    df = df.copy()

    cell_type_map = {"macro": 0, "micro": 1, "pico": 2}
    df["cell_type_enc"] = df["cell_type"].map(cell_type_map)

    slice_type_map = {"HC": 0, "URLLC": 1, "eMBB": 2, "mMTC": 3}
    df["slice_type_enc"] = df["slice_type"].map(slice_type_map)

    return df


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """对数值特征进行 Min-Max 归一化。

    归一化列：所有参与模型训练的数值特征。
    公式：x_norm = (x - x_min) / (x_max - x_min)
    """
    df = df.copy()

    columns_to_normalize = [
        "throughput_mbps",
        "latency_ms",
        "packet_loss_pct",
        "handover_count",
        "rsrp_dbm",
        "rsrq_db",
        "prb_utilization_pct",
        "active_users",
        "throughput_per_user",
        "spectral_efficiency",
    ]

    for col in columns_to_normalize:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max > col_min:
            df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min)
        else:
            df[f"{col}_norm"] = 0.0

    return df


def main():
    print("=" * 55)
    print("  5G KPI 数据预处理")
    print("=" * 55)

    df = load_data(RAW_CSV)

    print("\n[1/4] 添加时间特征...")
    df = add_time_features(df)

    print("[2/4] 添加派生特征...")
    df = add_derived_features(df)

    print("[3/4] 类别编码...")
    df = encode_categorical(df)

    print("[4/4] 数值归一化...")
    df = normalize_features(df)

    # 保存
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"\n预处理完成！")
    print(f"  输出: {OUT_CSV}")
    print(f"  维度: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"  新增列: hour_of_day, day_of_week, is_peak_hour, "
          f"throughput_per_user, spectral_efficiency")
    print(f"  编码列: cell_type_enc, slice_type_enc")
    print(f"  归一化列: 10 个 *_norm 列")


if __name__ == "__main__":
    main()
