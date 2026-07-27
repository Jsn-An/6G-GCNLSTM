"""
GCN+LSTM 模型训练脚本

流程：
1. 加载预处理后的 KPI 时序数据 (ds1_processed.csv)
2. 加载预构建的图结构 (graph_data.pt)
3. 构造全图滑动窗口数据集（每个窗口包含所有节点的连续时间步快照）
4. 训练 GCN+LSTM 模型
5. 保存模型检查点和训练历史

数据流：
  (num_cells, total_timesteps, F) ──滑动窗口──▶ (num_windows, num_cells, T, F)
  ↓
  GCN 每时间步提取空间特征 → LSTM 建模时序 → FC 输出预测
"""

import os
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 将项目根目录加入 Python Path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.gcn_lstm import GCNLSTM


# ============================================================================
# 配置
# ============================================================================

class Config:
    """训练配置。"""

    # ---- 路径 ----
    processed_csv = os.path.join(PROJECT_ROOT, "datasets", "ds1_processed.csv")
    graph_pt = os.path.join(PROJECT_ROOT, "data", "processed", "graph_data.pt")
    model_dir = os.path.join(PROJECT_ROOT, "results", "models")
    log_dir = os.path.join(PROJECT_ROOT, "results", "metrics")

    # ---- 数据 ----
    # 输入特征列（使用归一化后的特征）
    feature_cols = [
        "throughput_mbps_norm",
        "latency_ms_norm",
        "packet_loss_pct_norm",
        "handover_count_norm",
        "prb_utilization_pct_norm",
        "active_users_norm",
        "throughput_per_user_norm",
        "spectral_efficiency_norm",
    ]
    # 时间特征（需要归一化到 [0,1]）
    time_cols = ["hour_of_day", "day_of_week", "is_peak_hour"]

    target_col = "prb_utilization_pct_norm"   # 预测目标：PRB 利用率（拥塞指标）
    input_window = 12   # 用过去 12 个时间步预测下一时刻
    train_ratio = 0.70
    val_ratio = 0.15

    # ---- 模型 ----
    gcn_hidden = 64
    lstm_hidden = 128
    lstm_layers = 2
    dropout = 0.3

    # ---- 训练 ----
    batch_size = 8     # 每个 batch 包含多个全图时间窗口
    epochs = 100
    lr = 1e-3
    weight_decay = 1e-5
    patience = 15
    grad_clip = 1.0

    # ---- 设备 ----
    device = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()


# ============================================================================
# 数据集：全图时间窗口
# ============================================================================

class GraphWindowDataset(Dataset):
    """全图时间窗口数据集。

    每个样本是连续 input_window 步的**全图**节点特征快照。
    该设计确保 GCN 能够在每个时间步利用完整的图拓扑信息。

    Returns:
        x: (num_nodes, input_window, num_features)  torch.float32
        y: (num_nodes,)                              torch.float32
    """

    def __init__(
        self,
        data: np.ndarray,      # (num_nodes, total_timesteps, num_features)
        target: np.ndarray,    # (num_nodes, total_timesteps)
        input_window: int,
    ):
        self.num_nodes, self.total_ts, self.num_features = data.shape
        self.input_window = input_window
        self.num_windows = self.total_ts - input_window

        self.data = torch.tensor(data, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        x = self.data[:, idx : idx + self.input_window, :]    # (N, T, F)
        y = self.target[:, idx + self.input_window]            # (N,)
        return x, y


# ============================================================================
# 数据加载
# ============================================================================

def load_data():
    """加载预处理数据并构造训练/验证/测试集。"""
    print("=" * 60)
    print("加载数据...")
    print("=" * 60)

    df = pd.read_csv(cfg.processed_csv, parse_dates=["timestamp"])
    df = df.sort_values(["cell_id", "timestamp"]).reset_index(drop=True)

    cell_ids = sorted(df["cell_id"].unique())
    num_cells = len(cell_ids)

    # 对齐所有小区的时间步
    ts_per_cell = df.groupby("cell_id").size()
    min_ts = ts_per_cell.min()
    print(f"  小区数: {num_cells}  |  每小区时间步: {min_ts}")

    # 特征矩阵: (num_cells, min_ts, num_features)
    df["hour_norm"] = df["hour_of_day"] / 23.0
    df["dow_norm"] = df["day_of_week"] / 6.0
    actual_cols = cfg.feature_cols + ["hour_norm", "dow_norm", "is_peak_hour"]
    num_features = len(actual_cols)

    data_arr = np.zeros((num_cells, min_ts, num_features), dtype=np.float32)
    for i, cid in enumerate(cell_ids):
        cell_df = df[df["cell_id"] == cid].head(min_ts)
        data_arr[i] = cell_df[actual_cols].values

    # 目标
    target_idx = actual_cols.index(cfg.target_col)
    target_arr = data_arr[:, :, target_idx].copy()

    print(f"  特征维度: {num_features}  |  目标: {cfg.target_col}")

    # 按时间顺序划分（训练集不接触未来，测试集在时间上最远）
    train_end = int(min_ts * cfg.train_ratio)
    val_end = int(min_ts * (cfg.train_ratio + cfg.val_ratio))

    train_ds = GraphWindowDataset(data_arr[:, :val_end, :], target_arr[:, :val_end], cfg.input_window)
    val_ds = GraphWindowDataset(data_arr[:, :val_end, :], target_arr[:, :val_end], cfg.input_window)
    test_ds = GraphWindowDataset(data_arr[:, train_end:, :], target_arr[:, train_end:], cfg.input_window)

    print(f"  训练窗口: {len(train_ds)}  |  验证窗口: {len(val_ds)}  |  测试窗口: {len(test_ds)}")

    # 图结构
    g = torch.load(cfg.graph_pt, map_location="cpu", weights_only=False)

    # ---- 图节点数与数据节点数对齐 ----
    # 图数据可能是用不同规模的数据集生成的，需要裁剪边索引
    if g.num_nodes != num_cells:
        print(f"\n  ⚠ 图节点数 ({g.num_nodes}) ≠ 数据小区数 ({num_cells})，自动裁剪边索引...")
        mask = (g.edge_index[0] < num_cells) & (g.edge_index[1] < num_cells)
        g.edge_index = g.edge_index[:, mask]
        if g.edge_attr is not None:
            g.edge_attr = g.edge_attr[mask]
        print(f"  裁剪后: {g.edge_index.shape[1]} 边")

    print(f"  图结构: {g.num_nodes} 节点, {g.edge_index.shape[1]} 边\n")

    return train_ds, val_ds, test_ds, g.edge_index, g.edge_attr, num_cells, num_features


# ============================================================================
# 训练 / 评估
# ============================================================================

def run_epoch(model, loader, edge_index, edge_weight, optimizer, criterion, device, training):
    """一个 epoch 的训练或评估。"""
    model.train() if training else model.eval()

    total_loss, n = 0.0, 0
    for x, y in loader:
        # x: (batch, N, T, F)   y: (batch, N)
        x, y = x.to(device), y.to(device)

        if training:
            optimizer.zero_grad()

        pred = model(x, edge_index, edge_weight).squeeze(-1)  # (batch, N)
        loss = criterion(pred, y)

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        n += x.size(0)

    return total_loss / n


# ============================================================================
# 主训练流程
# ============================================================================

def train():
    """主训练流程。"""
    print("=" * 60)
    print("GCN+LSTM 拥塞预测 — 训练")
    print("=" * 60)

    # ---- 数据 ----
    (train_ds, val_ds, test_ds,
     edge_index, edge_weight,
     num_cells, num_features) = load_data()

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    # ---- 模型 ----
    model = GCNLSTM(
        in_features=num_features,
        gcn_hidden=cfg.gcn_hidden,
        lstm_hidden=cfg.lstm_hidden,
        lstm_layers=cfg.lstm_layers,
        output_dim=1,
        dropout=cfg.dropout,
    ).to(cfg.device)

    edge_index = edge_index.to(cfg.device)
    if edge_weight is not None:
        edge_weight = edge_weight.to(cfg.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型: GCN({num_features}→{cfg.gcn_hidden}) "
          f"+ LSTM({cfg.gcn_hidden}→{cfg.lstm_hidden}×{cfg.lstm_layers}) + FC")
    print(f"参数: {n_params:,}  |  设备: {cfg.device}\n")

    # ---- 优化器 ----
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7)
    criterion = nn.MSELoss()

    # ---- 训练 ----
    best_val, patience_cnt = float("inf"), 0
    history = {"train": [], "val": []}

    header = f"{'Epoch':>5}  {'Train':>10}  {'Val':>10}  {'LR':>8}  Status"
    print(header + "\n" + "-" * len(header))

    for epoch in range(1, cfg.epochs + 1):
        tr = run_epoch(model, train_loader, edge_index, edge_weight, optimizer, criterion, cfg.device, True)
        va = run_epoch(model, val_loader, edge_index, edge_weight, None, criterion, cfg.device, False)

        history["train"].append(tr)
        history["val"].append(va)
        scheduler.step(va)

        lr_now = optimizer.param_groups[0]["lr"]
        status = ""

        if va < best_val:
            best_val = va
            patience_cnt = 0
            status = "✓"
            os.makedirs(cfg.model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(cfg.model_dir, "gcn_lstm_best.pt"))
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.patience:
                print(f"{epoch:>5d}  {tr:>10.6f}  {va:>10.6f}  {lr_now:>8.2e}  早停")
                break

        if epoch <= 5 or epoch % 10 == 0:
            print(f"{epoch:>5d}  {tr:>10.6f}  {va:>10.6f}  {lr_now:>8.2e}  {status}")

    # ---- 测试 ----
    print(f"\n{'='*60}\n测试集评估\n{'='*60}")
    model.load_state_dict(torch.load(
        os.path.join(cfg.model_dir, "gcn_lstm_best.pt"),
        map_location=cfg.device, weights_only=True,
    ))
    te = run_epoch(model, test_loader, edge_index, edge_weight, None, criterion, cfg.device, False)
    print(f"  MSE : {te:.6f}")
    print(f"  RMSE: {np.sqrt(te):.6f}")
    print(f"  Best Val: {best_val:.6f}")

    # ---- 保存历史 ----
    os.makedirs(cfg.log_dir, exist_ok=True)
    np.savez(os.path.join(cfg.log_dir, "training_history.npz"), train=history["train"], val=history["val"])
    print(f"\n历史 → {cfg.log_dir}/training_history.npz")
    print(f"模型 → {cfg.model_dir}/gcn_lstm_best.pt")

    return model


if __name__ == "__main__":
    train()
