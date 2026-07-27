"""
Dataset 1: 5G Network KPI Dataset (v2 — 带空间相关性)
=============================================================================

改进点（相比 v1）：
1. 每个小区有固定的 cell_type 和主要 slice_type，不再是每行随机
2. 引入全局时间趋势（日周期 + 小时峰值），模拟真实的潮汐效应
3. 通过空间权重矩阵引入跨小区相关性：
   - 小区索引相近的（如 CELL_001 vs CELL_002）有更相似的趋势
   - macro 站之间共享骨干网层面的流量波动
4. 每个小区仍有独立的噪声项，保证数据的真实感

生成结果将覆盖：
  datasets/dataset1_5g_network_kpi.csv  （原始数据）
  datasets/ds1_processed.csv            （预处理后，由外部脚本生成）
=============================================================================

Columns: timestamp, cell_id, cell_type, slice_type(主),
         throughput_mbps, latency_ms, packet_loss_pct, handover_count,
         rsrp_dbm, rsrq_db, prb_utilization_pct, active_users, sla_compliant
"""
import os
import pandas as pd
import numpy as np

np.random.seed(42)

# ============================================================================
# 配置参数
# ============================================================================
NUM_CELLS = 20               # 基站数量
TIMESTEPS = 250              # 每个基站的时间步数
FREQ_MIN = 60                # 采样间隔（分钟），60 = 每小时 1 条

# 空间相关性强度 (0~1)，越大则相邻小区模式越相似
SPATIAL_CORR_STRENGTH = 0.6
# 全局趋势强度 (0~1)，越大则所有小区共享的潮汐模式越强
GLOBAL_TREND_STRENGTH = 0.4

# ============================================================================
# Step 1: 定义每个小区的固定属性
# ============================================================================

# 20 个小区：按索引分配 cell_type 和主 slice_type
cell_ids = np.array([f"CELL_{i:03d}" for i in range(1, NUM_CELLS + 1)])

# cell_type: 前 10 个 macro，中间 6 个 micro，后 4 个 pico
cell_types = np.array(
    ["macro"] * 10 + ["micro"] * 6 + ["pico"] * 4
)

# 主 slice_type: 均匀分配
primary_slices = np.array(
    ["eMBB"] * 5 + ["URLLC"] * 5 + ["mMTC"] * 5 + ["HC"] * 5
)

print(f"基站配置: {NUM_CELLS} 个小区")
print(f"  macro: {(cell_types == 'macro').sum()}, "
      f"micro: {(cell_types == 'micro').sum()}, "
      f"pico: {(cell_types == 'pico').sum()}")


# ============================================================================
# Step 2: 构建空间权重矩阵（控制相关性强度）
# ============================================================================

# 方法：指数衰减核，小区 i 和 j 越近（索引差越小），权重越大
def build_spatial_weight_matrix(n_cells: int, strength: float) -> np.ndarray:
    """构建 (n_cells, n_cells) 的空间权重矩阵。

    小区索引差越大，相关性越低。类似真实网络中地理邻近基站
    共享相似流量模式的现象。
    """
    dist = np.abs(np.subtract.outer(np.arange(n_cells), np.arange(n_cells)))
    # 指数衰减：距离为 0 时权重=1，距离越大越接近 0
    W = np.exp(-dist / (n_cells * 0.15)) * strength
    # 对角线（自身）= 1
    np.fill_diagonal(W, 1.0)
    return W


W = build_spatial_weight_matrix(NUM_CELLS, SPATIAL_CORR_STRENGTH)
print(f"\n空间权重矩阵形状: {W.shape}")
print(f"  平均空间相关性(非对角线): {W[~np.eye(NUM_CELLS, dtype=bool)].mean():.3f}")


# ============================================================================
# Step 3: 生成全局时间趋势（共享的潮汐模式）
# ============================================================================

def generate_global_trend(n_timesteps: int, strength: float) -> np.ndarray:
    """生成全局时间趋势信号。

    模拟真实网络中的流量潮汐效应：
    - 日周期（24 小时正弦波）
    - 加上多谐波使模式更丰富
    - 归一化到 [0, strength] 范围
    """
    t = np.arange(n_timesteps) / (24 * 60 / FREQ_MIN) * 2 * np.pi  # 日周期
    trend = (
        0.6 * np.sin(t)                    # 主日周期
        + 0.25 * np.sin(2 * t + 1.5)       # 二次谐波（早晚高峰）
        + 0.15 * np.sin(0.5 * t + 3.0)     # 低频慢变（工作日模式）
    )
    # 归一化
    trend = (trend - trend.min()) / (trend.max() - trend.min() + 1e-8)
    return trend * strength


global_trend = generate_global_trend(TIMESTEPS, GLOBAL_TREND_STRENGTH)


# ============================================================================
# Step 4: 生成每个小区的 KPI 时间序列（带相关性）
# ============================================================================

def get_kpi_params(slice_type: str) -> dict:
    """根据切片类型返回 KPI 均值和标准差。"""
    params = {
        "eMBB": {
            "throughput": (820, 120), "latency": (8.5, 2.1),
            "pkt_loss": (0.5,), "active_users": (45,),
            "prb_util": (72, 12),
        },
        "URLLC": {
            "throughput": (250, 60), "latency": (1.8, 0.4),
            "pkt_loss": (0.05,), "active_users": (12,),
            "prb_util": (55, 8),
        },
        "mMTC": {
            "throughput": (45, 15), "latency": (25, 8),
            "pkt_loss": (1.2,), "active_users": (200,),
            "prb_util": (35, 10),
        },
        "HC": {
            "throughput": (1200, 200), "latency": (4.5, 1.0),
            "pkt_loss": (0.2,), "active_users": (8,),
            "prb_util": (80, 10),
        },
    }
    return params[slice_type]


# 生成所有小区的 KPI 时间序列
rows = []
for i in range(NUM_CELLS):
    cid = cell_ids[i]
    ct = cell_types[i]
    st = primary_slices[i]
    params = get_kpi_params(st)

    # 该小区从全局趋势中获得的份额 = 全局趋势 + 空间加权邻居贡献
    cell_trend = global_trend.copy()
    for j in range(NUM_CELLS):
        if i != j and W[i, j] > 0.01:
            # 邻居 j 的小幅随机波动也影响到 i
            neighbor_noise = np.random.normal(0, W[i, j] * 0.3, TIMESTEPS)
            cell_trend += neighbor_noise

    # 归一化趋势到 [0, 1]
    cell_trend = (cell_trend - cell_trend.min()) / (cell_trend.max() - cell_trend.min() + 1e-8)

    # 生成时间序列
    timestamps = pd.date_range(
        "2024-01-01", periods=TIMESTEPS, freq=f"{FREQ_MIN}min"
    )
    start_ts = timestamps[0]

    for t_idx, ts in enumerate(timestamps):
        # 趋势驱动的 KPI 波动（高峰期吞吐量和 PRB 利用率上升）
        trend_factor = 1.0 + (cell_trend[t_idx] - 0.5) * 0.8

        tp_mean, tp_std = params["throughput"]
        throughput = np.clip(
            np.random.normal(tp_mean * trend_factor, tp_std), 5, 2000
        )

        lat_mean, lat_std = params["latency"]
        latency = np.clip(
            np.random.normal(lat_mean * (2 - trend_factor), lat_std), 0.3, 100
        )

        pkt_loss_scale = params["pkt_loss"][0]
        pkt_loss = np.clip(np.random.exponential(pkt_loss_scale), 0, 10)

        ho_count = int(np.random.poisson(3))

        rsrp_val = np.clip(np.random.normal(-95, 15), -120, -70)
        rsrq_val = np.clip(np.random.normal(-12, 5), -20, -3)

        prb_mean, prb_std = params["prb_util"]
        prb_util = np.clip(
            np.random.normal(prb_mean * trend_factor, prb_std), 5, 100
        )

        au_lambda = params["active_users"][0]
        active_users = int(np.clip(
            np.random.poisson(au_lambda * trend_factor), 1, 300
        ))

        # SLA 合规判定
        if st == "eMBB":
            sla_ok = throughput >= 300 and latency <= 20 and pkt_loss <= 2.0
        elif st == "URLLC":
            sla_ok = throughput >= 100 and latency <= 3.0 and pkt_loss <= 0.1
        elif st == "mMTC":
            sla_ok = latency <= 50 and pkt_loss <= 5.0
        else:  # HC
            sla_ok = throughput >= 500 and latency <= 10 and pkt_loss <= 0.5

        rows.append({
            "timestamp": ts,
            "cell_id": cid,
            "cell_type": ct,
            "slice_type": st,
            "throughput_mbps": round(throughput, 2),
            "latency_ms": round(latency, 3),
            "packet_loss_pct": round(pkt_loss, 4),
            "handover_count": ho_count,
            "rsrp_dbm": round(rsrp_val, 1),
            "rsrq_db": round(rsrq_val, 2),
            "prb_utilization_pct": round(prb_util, 1),
            "active_users": active_users,
            "sla_compliant": int(sla_ok),
        })

# ============================================================================
# Step 5: 保存
# ============================================================================

df = pd.DataFrame(rows)

# 输出到 datasets/ 目录（与脚本同目录）
script_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(script_dir, "dataset1_5g_network_kpi.csv")
df.to_csv(out_path, index=False)

print(f"\n数据集已保存: {out_path}")
print(f"  维度: {df.shape[0]} rows × {df.shape[1]} cols")
print(f"  小区数: {df['cell_id'].nunique()}")
print(f"  每小区时间步: {TIMESTEPS}")
print(f"  时间范围: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
print(f"\n前 3 行预览:")
print(df.head(3).to_string())
print(f"\nSLA 合规率（按切片类型）:")
print(df.groupby("slice_type")["sla_compliant"].mean().round(3))

# ============================================================================
# Step 6: 验证相关性（快速检查）
# ============================================================================
print(f"\n========== 相关性验证 ==========")
prb_pivot = df.pivot_table(
    values="prb_utilization_pct", index="timestamp", columns="cell_id"
)
corr_mat = prb_pivot.corr().values
upper = corr_mat[np.triu_indices_from(corr_mat, k=1)]
print(f"PRB 利用率的小区间平均 |相关性|: {np.abs(upper).mean():.4f}")
print(f"  (v1 约为 0.05，v2 应显著提高)")
print(f"=================================")
