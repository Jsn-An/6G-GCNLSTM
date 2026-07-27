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

    device = "cuda" if torch.cuda.is_available() else "cpu"


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
    cell_type_map: dict,
    save_path: str,
    num_cells_to_plot: int = 8,
):
    """Plot predicted vs true PRB utilization for a subset of cells.

    Args:
        y_true: (num_cells, num_windows) 真实值
        y_pred: (num_cells, num_windows) 预测值
        cell_ids: 小区 ID 列表
        cell_type_map: {cell_id: cell_type} 字典，用于标题标注基站类型
        save_path: 保存路径
        num_cells_to_plot: 画几个子图
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 按 cell_type 顺序排：macro → micro → pico，同类内部按 RMSE 升序
    type_order = {"macro": 0, "micro": 1, "pico": 2}
    rmse_per_cell = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))
    scored = [(type_order.get(cell_type_map.get(cid, ""), 99), float(rmse_per_cell[i]), i)
              for i, cid in enumerate(cell_ids)]
    scored.sort()
    plot_indices = [s[2] for s in scored[:num_cells_to_plot]]

    n = min(num_cells_to_plot, len(cell_ids))
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = axes.flatten() if n > 1 else [axes]

    for idx, i in enumerate(plot_indices):
        ax = axes[idx]
        t = np.arange(len(y_true[i]))
        ax.plot(t, y_true[i], "b-", linewidth=1.2, alpha=0.7, label="Ground Truth")
        ax.plot(t, y_pred[i], "r--", linewidth=1.2, alpha=0.7, label="Prediction")

        cid = cell_ids[i]
        ct = cell_type_map.get(cid, "?")
        mse_val = np.mean((y_true[i] - y_pred[i]) ** 2)
        ax.set_title(f"{cid} ({ct})  |  MSE={mse_val:.4f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("PRB Util (norm)")
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    # 只放一个共享图例
    if n > 0:
        axes[0].legend(fontsize=8)

    fig.suptitle("GCN+LSTM Congestion Prediction — Predictions vs Ground Truth (Test Set)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Predictions plot: {save_path}")


def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    """Plot prediction error histogram and scatter plot."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    errors = (y_pred - y_true).flatten()
    errors = errors[~np.isnan(errors)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Error distribution histogram
    axes[0].hist(errors, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error")
    axes[0].axvline(np.mean(errors), color="orange", linestyle="-", linewidth=1.5,
                    label=f"Mean={np.mean(errors):.4f}")
    axes[0].set_xlabel("Prediction Error (y_pred - y_true)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Prediction Error Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: hexbin density plot — 颜色越亮=点越密集，自然集中在 y=x 线附近
    yt = y_true.flatten()
    yp = y_pred.flatten()
    ax2 = axes[1]
    hb = ax2.hexbin(yt, yp, gridsize=50, cmap="YlOrRd", mincnt=1)
    plt.colorbar(hb, ax=ax2, label="Count")

    lims = [0, 1]
    ax2.plot(lims, lims, "b--", linewidth=1.8, label="y=x (Perfect)")
    ax2.set_xlabel("Ground Truth")
    ax2.set_ylabel("Prediction")
    r2 = np.corrcoef(yt, yp)[0, 1] ** 2
    ax2.set_title(f"Prediction vs Ground Truth  (R²={r2:.4f})")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Error distribution plot: {save_path}")


def plot_per_node_metrics(
    metrics_per_node: dict,
    cell_ids: list,
    cell_type_map: dict,
    save_path: str,
):
    """Plot per-cell RMSE bar chart, colored by cell_type.

    Args:
        metrics_per_node: {cell_id: {"RMSE": float, ...}}
        cell_ids: 小区 ID 列表
        cell_type_map: {cell_id: cell_type} 字典
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 按 cell_type 排序再按 RMSE 排序
    type_order = {"macro": 0, "micro": 1, "pico": 2}
    rmse_values = [metrics_per_node[cid]["RMSE"] for cid in cell_ids]
    types = [cell_type_map.get(cid, "unknown") for cid in cell_ids]
    # 用 Python sorted 排序，避免 np.argsort 对元组的兼容问题
    scored = [(type_order.get(types[i], 99), rmse_values[i], i) for i in range(len(cell_ids))]
    scored.sort()
    order = [s[2] for s in scored]
    sorted_ids = [cell_ids[i] for i in order]
    sorted_rmse = np.array([rmse_values[i] for i in order])
    sorted_types = [types[i] for i in order]

    type_colors = {"macro": "#E74C3C", "micro": "#3498DB", "pico": "#2ECC71"}
    bar_colors = [type_colors.get(ct, "#999999") for ct in sorted_types]

    fig, ax = plt.subplots(figsize=(14, 5.5))
    bars = ax.bar(range(len(sorted_ids)), sorted_rmse, color=bar_colors, edgecolor="white")

    mean_rmse = np.mean(sorted_rmse)
    ax.axhline(mean_rmse, color="gray", linestyle="--", linewidth=1.5,
               label=f"Mean RMSE = {mean_rmse:.4f}")

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#E74C3C", label=f"macro ({sorted_types.count('macro')})"),
        Patch(facecolor="#3498DB", label=f"micro ({sorted_types.count('micro')})"),
        Patch(facecolor="#2ECC71", label=f"pico ({sorted_types.count('pico')})"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper left")

    short_labels = [cid.replace("CELL_", "") for cid in sorted_ids]
    ax.set_xticks(range(len(sorted_ids)))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("RMSE", fontsize=11)
    ax.set_title(
        f"Per-Cell PRB Utilization Prediction RMSE  "
        f"(mean={mean_rmse:.4f}, best={sorted_rmse[0]:.4f}, worst={sorted_rmse[-1]:.4f})",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_ylim(0, sorted_rmse[-1] * 1.25)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Per-node RMSE plot: {save_path}")


# ============================================================================
# 主评估流程
# ============================================================================

def evaluate():
    """Main evaluation pipeline."""
    print("=" * 60)
    print("GCN+LSTM Congestion Prediction — Evaluation")
    print("=" * 60)

    # ---- Load data ----
    df = pd.read_csv(cfg.processed_csv, parse_dates=["timestamp"])
    df = df.sort_values(["cell_id", "timestamp"]).reset_index(drop=True)
    cell_ids = sorted(df["cell_id"].unique())
    num_cells = len(cell_ids)

    # ---- 获取每个基站的 cell_type（用于可视化标注） ----
    cell_type_map = df.groupby("cell_id")["cell_type"].first().to_dict()

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

    print(f"  Cells: {num_cells}")
    print(f"  Test timesteps: {test_data.shape[1]} (from t={train_end})")

    # Graph
    g = torch.load(cfg.graph_pt, map_location="cpu", weights_only=False)
    edge_index = g.edge_index.to(cfg.device)
    edge_weight = g.edge_attr
    if edge_weight is not None:
        edge_weight = edge_weight.to(cfg.device)

    # ---- Dataset ----
    test_ds = GraphWindowDataset(test_data, test_target, cfg.input_window)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    print(f"  Test windows: {len(test_ds)}")

    # ---- Model ----
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
    print(f"  Model loaded: {cfg.model_path}")

    # ---- Generate predictions ----
    print("\nGenerating predictions...")
    all_preds, all_trues = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(cfg.device)          # (1, N, T, F)
            y = y.to(cfg.device)          # (1, N)
            pred = model(x, edge_index, edge_weight).squeeze(0).squeeze(-1)  # (N,)
            all_preds.append(pred.cpu().numpy())
            all_trues.append(y.squeeze(0).cpu().numpy())

    y_pred = np.array(all_preds).T      # (num_cells, num_windows)
    y_true = np.array(all_trues).T

    print(f"  Prediction matrix: {y_pred.shape}")

    # ---- Overall metrics ----
    print(f"\n{'='*60}\nOverall Metrics\n{'='*60}")
    metrics = compute_metrics(y_true.flatten(), y_pred.flatten())
    for k, v in metrics.items():
        print(f"  {k:>12s}: {v:.6f}")

    # ---- Per-node metrics ----
    metrics_per_node = {}
    for i, cid in enumerate(cell_ids):
        metrics_per_node[cid] = compute_metrics(y_true[i], y_pred[i])

    avg_rmse = np.mean([m["RMSE"] for m in metrics_per_node.values()])
    avg_mae = np.mean([m["MAE"] for m in metrics_per_node.values()])
    avg_r2 = np.mean([m["R2"] for m in metrics_per_node.values()])
    print(f"\nPer-Node Average Metrics:")
    print(f"  Avg RMSE: {avg_rmse:.6f}")
    print(f"  Avg MAE:  {avg_mae:.6f}")
    print(f"  Avg R²:   {avg_r2:.6f}")

    # ---- Save predictions ----
    os.makedirs(cfg.pred_dir, exist_ok=True)
    np.savez(
        os.path.join(cfg.pred_dir, "test_predictions.npz"),
        y_true=y_true,
        y_pred=y_pred,
        cell_ids=cell_ids,
    )

    metrics_df = pd.DataFrame(metrics_per_node).T
    metrics_df.index.name = "cell_id"
    os.makedirs(cfg.metrics_dir, exist_ok=True)
    metrics_df.to_csv(os.path.join(cfg.metrics_dir, "per_node_metrics.csv"))
    print(f"\nPredictions → {cfg.pred_dir}/test_predictions.npz")
    print(f"Per-node metrics → {cfg.metrics_dir}/per_node_metrics.csv")

    # ---- Visualization ----
    print(f"\n{'='*60}\nGenerating Visualizations\n{'='*60}")
    plot_predictions(
        y_true, y_pred, cell_ids, cell_type_map,
        os.path.join(cfg.figure_dir, "gcn_lstm_predictions.png"),
    )
    plot_error_distribution(
        y_true, y_pred,
        os.path.join(cfg.figure_dir, "gcn_lstm_error_dist.png"),
    )
    plot_per_node_metrics(
        metrics_per_node, cell_ids, cell_type_map,
        os.path.join(cfg.figure_dir, "gcn_lstm_per_node_rmse.png"),
    )

    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"Model: {cfg.model_path}")
    print(f"Figures: {cfg.figure_dir}/")
    print("=" * 60)

    return metrics, metrics_per_node


if __name__ == "__main__":
    evaluate()
