"""
GCN+LSTM 模型评估脚本

功能：
1. 加载训练好的模型和测试数据
2. 计算回归评估指标（MSE, MAE, RMSE, R², MAPE）
3. 生成预测 vs 真实值可视化
4. 逐节点误差分析
5. 保存预测结果和评估报告
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.gcn_lstm import GCNLSTM


# ============================================================================
# 配置
# ============================================================================

class EvalConfig:
    processed_csv = os.path.join(PROJECT_ROOT, "datasets", "ds1_processed.csv")
    graph_pt = os.path.join(PROJECT_ROOT, "data", "processed", "graph_data.pt")
    model_path = os.path.join(PROJECT_ROOT, "results", "models", "gcn_lstm_best.pt")
    pred_dir = os.path.join(PROJECT_ROOT, "results", "predictions")
    figure_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
    metrics_dir = os.path.join(PROJECT_ROOT, "results", "metrics")

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
    target_col = "prb_utilization_pct_norm"
    input_window = 12
    train_ratio = 0.70
    val_ratio = 0.15

    gcn_hidden = 64
    lstm_hidden = 128
    lstm_layers = 2
    dropout = 0.3

    device = "cpu"


cfg = EvalConfig()


# ============================================================================
# 数据集（与训练一致）
# ============================================================================

class GraphWindowDataset(torch.utils.data.Dataset):
    def __init__(self, data, target, input_window):
        self.num_windows = data.shape[1] - input_window
        self.data = torch.tensor(data, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)
        self.input_window = input_window

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        x = self.data[:, idx : idx + self.input_window, :]
        y = self.target[:, idx + self.input_window]
        return x, y


# ============================================================================
# 指标计算
# ============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算回归评估指标。"""
    # 去除 NaN
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)

    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # MAPE (Mean Absolute Percentage Error)
    nonzero_mask = np.abs(y_true) > 1e-6
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
    else:
        mape = 0.0

    # 皮尔逊相关系数
    corr = np.corrcoef(y_true, y_pred)[0, 1]

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE(%)": mape,
        "Correlation": corr,
    }


# ============================================================================
# 可视化
# ============================================================================

def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cell_ids: list,
    save_path: str,
    num_cells_to_plot: int = 8,
):
    """绘制部分小区的预测 vs 真实值曲线。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    axes = axes.flatten()

    for i in range(min(num_cells_to_plot, len(cell_ids))):
        ax = axes[i]
        t = np.arange(len(y_true[i]))
        ax.plot(t, y_true[i], "b-", linewidth=1.2, alpha=0.7, label="真实值")
        ax.plot(t, y_pred[i], "r--", linewidth=1.2, alpha=0.7, label="预测值")
        ax.set_title(f"{cell_ids[i]}", fontsize=10, fontweight="bold")
        ax.set_xlabel("时间步")
        ax.set_ylabel("PRB 利用率 (归一化)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("GCN+LSTM 拥塞预测 — 预测 vs 真实值（测试集）", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  预测曲线图: {save_path}")


def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    """绘制预测误差分布直方图。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    errors = (y_pred - y_true).flatten()
    errors = errors[~np.isnan(errors)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 误差分布直方图
    axes[0].hist(errors, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5, label="零误差线")
    axes[0].axvline(np.mean(errors), color="orange", linestyle="-", linewidth=1.5,
                    label=f"均值={np.mean(errors):.4f}")
    axes[0].set_xlabel("预测误差 (y_pred - y_true)")
    axes[0].set_ylabel("频数")
    axes[0].set_title("预测误差分布")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 散点图：真实值 vs 预测值
    axes[1].scatter(y_true.flatten(), y_pred.flatten(), alpha=0.3, s=5, color="steelblue")
    lims = [0, 1]
    axes[1].plot(lims, lims, "r--", linewidth=1.5, label="y=x (完美预测)")
    axes[1].set_xlabel("真实值")
    axes[1].set_ylabel("预测值")
    axes[1].set_title(f"预测 vs 真实 (R²={np.corrcoef(y_true.flatten(), y_pred.flatten())[0,1]**2:.4f})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  误差分布图: {save_path}")


def plot_per_node_metrics(metrics_per_node: dict, cell_ids: list, save_path: str):
    """绘制每个小区的 RMSE 条形图。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    rmse_values = [metrics_per_node[cid]["RMSE"] for cid in cell_ids]
    colors = ["#2196F3" if v < np.median(rmse_values) else "#FF5722" for v in rmse_values]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(cell_ids)), rmse_values, color=colors, edgecolor="white")
    ax.axhline(np.mean(rmse_values), color="green", linestyle="--", linewidth=1.5,
               label=f"平均 RMSE={np.mean(rmse_values):.4f}")
    ax.set_xticks(range(len(cell_ids)))
    ax.set_xticklabels(cell_ids, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("RMSE")
    ax.set_title("各小区 PRB 利用率预测 RMSE")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  逐节点误差图: {save_path}")


# ============================================================================
# 主评估流程
# ============================================================================

def evaluate():
    """主评估流程。"""
    print("=" * 60)
    print("GCN+LSTM 拥塞预测 — 评估")
    print("=" * 60)

    # ---- 加载数据 ----
    df = pd.read_csv(cfg.processed_csv, parse_dates=["timestamp"])
    df = df.sort_values(["cell_id", "timestamp"]).reset_index(drop=True)
    cell_ids = sorted(df["cell_id"].unique())
    num_cells = len(cell_ids)

    ts_per_cell = df.groupby("cell_id").size()
    min_ts = int(ts_per_cell.min())

    df["hour_norm"] = df["hour_of_day"] / 23.0
    df["dow_norm"] = df["day_of_week"] / 6.0
    actual_cols = cfg.feature_cols + ["hour_norm", "dow_norm", "is_peak_hour"]
    num_features = len(actual_cols)

    data_arr = np.zeros((num_cells, min_ts, num_features), dtype=np.float32)
    for i, cid in enumerate(cell_ids):
        cell_df = df[df["cell_id"] == cid].head(min_ts)
        data_arr[i] = cell_df[actual_cols].values

    target_idx = actual_cols.index(cfg.target_col)
    target_arr = data_arr[:, :, target_idx].copy()

    train_end = int(min_ts * cfg.train_ratio)
    test_data = data_arr[:, train_end:, :]
    test_target = target_arr[:, train_end:]

    print(f"  小区数: {num_cells}")
    print(f"  测试时间步: {test_data.shape[1]} (从 t={train_end} 开始)")

    # 图结构
    g = torch.load(cfg.graph_pt, map_location="cpu", weights_only=False)
    edge_index = g.edge_index.to(cfg.device)
    edge_weight = g.edge_attr
    if edge_weight is not None:
        edge_weight = edge_weight.to(cfg.device)

    # ---- 数据集 ----
    test_ds = GraphWindowDataset(test_data, test_target, cfg.input_window)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    print(f"  测试窗口: {len(test_ds)}")

    # ---- 模型 ----
    model = GCNLSTM(
        in_features=num_features,
        gcn_hidden=cfg.gcn_hidden,
        lstm_hidden=cfg.lstm_hidden,
        lstm_layers=cfg.lstm_layers,
        output_dim=1,
        dropout=cfg.dropout,
    ).to(cfg.device)

    checkpoint = torch.load(cfg.model_path, map_location=cfg.device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"  模型已加载: {cfg.model_path}")

    # ---- 生成预测 ----
    print("\n生成预测...")
    all_preds, all_trues = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(cfg.device)          # (1, N, T, F)
            y = y.to(cfg.device)          # (1, N)
            pred = model(x, edge_index, edge_weight).squeeze(0).squeeze(-1)  # (N,)
            all_preds.append(pred.cpu().numpy())
            all_trues.append(y.squeeze(0).cpu().numpy())

    # 拼接: (num_windows, num_cells) → 转置为 (num_cells, num_windows)
    y_pred = np.array(all_preds).T      # (num_cells, num_windows)
    y_true = np.array(all_trues).T      # (num_cells, num_windows)

    print(f"  预测矩阵: {y_pred.shape}")

    # ---- 整体指标 ----
    print(f"\n{'='*60}\n整体评估指标\n{'='*60}")
    metrics = compute_metrics(y_true.flatten(), y_pred.flatten())
    for k, v in metrics.items():
        print(f"  {k:>12s}: {v:.6f}")

    # ---- 逐节点指标 ----
    metrics_per_node = {}
    for i, cid in enumerate(cell_ids):
        metrics_per_node[cid] = compute_metrics(y_true[i], y_pred[i])

    avg_rmse = np.mean([m["RMSE"] for m in metrics_per_node.values()])
    avg_mae = np.mean([m["MAE"] for m in metrics_per_node.values()])
    avg_r2 = np.mean([m["R2"] for m in metrics_per_node.values()])
    print(f"\n逐节点平均指标:")
    print(f"  Avg RMSE: {avg_rmse:.6f}")
    print(f"  Avg MAE:  {avg_mae:.6f}")
    print(f"  Avg R²:   {avg_r2:.6f}")

    # ---- 保存预测结果 ----
    os.makedirs(cfg.pred_dir, exist_ok=True)
    np.savez(
        os.path.join(cfg.pred_dir, "test_predictions.npz"),
        y_true=y_true,
        y_pred=y_pred,
        cell_ids=cell_ids,
    )

    # 保存逐节点指标
    metrics_df = pd.DataFrame(metrics_per_node).T
    metrics_df.index.name = "cell_id"
    os.makedirs(cfg.metrics_dir, exist_ok=True)
    metrics_df.to_csv(os.path.join(cfg.metrics_dir, "per_node_metrics.csv"))
    print(f"\n预测结果 → {cfg.pred_dir}/test_predictions.npz")
    print(f"逐节点指标 → {cfg.metrics_dir}/per_node_metrics.csv")

    # ---- 可视化 ----
    print(f"\n{'='*60}\n生成可视化\n{'='*60}")
    plot_predictions(
        y_true, y_pred, cell_ids,
        os.path.join(cfg.figure_dir, "gcn_lstm_predictions.png"),
    )
    plot_error_distribution(
        y_true, y_pred,
        os.path.join(cfg.figure_dir, "gcn_lstm_error_dist.png"),
    )
    plot_per_node_metrics(
        metrics_per_node, cell_ids,
        os.path.join(cfg.figure_dir, "gcn_lstm_per_node_rmse.png"),
    )

    print(f"\n{'='*60}")
    print("评估完成！")
    print(f"模型路径: {cfg.model_path}")
    print(f"图表路径: {cfg.figure_dir}/")
    print("=" * 60)

    return metrics, metrics_per_node


if __name__ == "__main__":
    evaluate()
