"""
任务卸载与智能调度 — 运行脚本 (Pipeline)

流程：
1. 加载 GCN+LSTM 模型预测结果
2. 加载图结构
3. 执行任务卸载优化（多种策略对比）
4. 执行智能调度仿真（多种策略对比）
5. 生成可视化报告
6. 保存结果

用法：
    python scripts/run_offloading.py
    python scripts/run_offloading.py --strategy hybrid
    python scripts/run_offloading.py --timesteps 50 --verbose
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.optimization.offloading import (
    OffloadingConfig,
    OffloadingStrategy,
    OffloadingResult,
    run_offloading,
    compare_strategies,
    detect_congestion,
)
from src.optimization.scheduler import (
    SchedulingConfig,
    SchedulingPolicy,
    SchedulingResult,
    run_scheduler,
    compare_scheduling_policies,
    generate_tasks,
)


# ============================================================================
# 配置
# ============================================================================

class PipelineConfig:
    """Pipeline 配置。"""
    # 路径
    predictions_npz = os.path.join(PROJECT_ROOT, "results", "predictions", "test_predictions.npz")
    graph_pt = os.path.join(PROJECT_ROOT, "data", "processed", "graph_data.pt")
    per_node_csv = os.path.join(PROJECT_ROOT, "results", "metrics", "per_node_metrics.csv")
    output_dir = os.path.join(PROJECT_ROOT, "results", "metrics")
    figure_dir = os.path.join(PROJECT_ROOT, "reports", "figures")

    # 卸载配置
    congestion_threshold = 0.70
    max_offload_ratio = 0.50
    max_targets = 3
    max_hop = 2

    # 调度配置
    num_sim_timesteps = 30

    # 默认策略
    default_strategy = "hybrid"


cfg = PipelineConfig()


# ============================================================================
# 数据加载
# ============================================================================

def load_data():
    """加载预测结果和图结构。"""
    print("=" * 60)
    print("加载数据...")
    print("=" * 60)

    # 预测结果
    if os.path.exists(cfg.predictions_npz):
        data = np.load(cfg.predictions_npz)
        y_true = data["y_true"]          # (num_cells, num_windows)
        y_pred = data["y_pred"]
        cell_ids = data["cell_ids"]
        print(f"  Predictions: {y_pred.shape} (from {cfg.predictions_npz})")
    else:
        print(f"  ⚠ 预测文件不存在: {cfg.predictions_npz}")
        # 使用模拟数据继续演示
        print("  使用模拟数据继续演示...")
        num_cells = 20
        num_windows = 50
        y_true = np.random.default_rng(42).beta(2, 2, (num_cells, num_windows)).astype(np.float32)
        y_pred = y_true + np.random.default_rng(43).normal(0, 0.05, y_true.shape).astype(np.float32)
        cell_ids = np.array([f"CELL_{i:03d}" for i in range(1, num_cells + 1)])

    # 图结构
    if os.path.exists(cfg.graph_pt):
        g = torch.load(cfg.graph_pt, map_location="cpu", weights_only=False)
        edge_index = g.edge_index.numpy()
        edge_weight = g.edge_attr.numpy() if g.edge_attr is not None else None
        print(f"  Graph: {edge_index.shape[1]} edges (from {cfg.graph_pt})")
    else:
        print(f"  ⚠ 图文件不存在: {cfg.graph_pt}")
        # 生成随机图
        num_nodes = y_pred.shape[0]
        edge_index, edge_weight = _generate_random_graph(num_nodes)
        print(f"  使用随机图: {edge_index.shape[1]} edges")

    return y_true, y_pred, cell_ids, edge_index, edge_weight


def _generate_random_graph(num_nodes: int, avg_degree: int = 4):
    """生成随机图结构（备用）。"""
    rng = np.random.default_rng(42)
    edges = []
    weights = []
    for i in range(num_nodes):
        # 每个节点连接 avg_degree 个邻居
        targets = set()
        for _ in range(avg_degree):
            j = rng.integers(0, num_nodes)
            if j != i:
                targets.add(j)
        for j in targets:
            edges.append([i, j])
            edges.append([j, i])
            w = 0.3 + 0.7 * rng.random()
            weights.extend([w, w])
    return np.array(edges).T, np.array(weights)


# ============================================================================
# 可视化
# ============================================================================

def plot_offloading_comparison(
    load_before: np.ndarray,
    load_after: np.ndarray,
    cell_ids: np.ndarray,
    strategy_name: str,
    save_path: str,
):
    """绘制卸载前后负载对比图（只保留负载分布，去掉拥塞状态矩阵）。

    Args:
        load_before: 卸载前负载数组
        load_after: 卸载后负载数组
        cell_ids: 小区 ID
        strategy_name: 策略名称
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n = len(load_before)
    x = np.arange(n)

    # 按卸载前负载降序排列，只看有变化的
    delta = np.abs(load_before - load_after)
    changed = delta > 0.001
    order = np.argsort(load_before * changed.astype(float))[::-1]
    # 取前 15 个（有变化的优先）
    top_n = min(15, n)
    plot_idx = order[:top_n]

    fig, ax = plt.subplots(figsize=(14, 5.5))

    xi = np.arange(top_n)
    width = 0.35
    before_vals = load_before[plot_idx]
    after_vals = load_after[plot_idx]
    labels = [cell_ids[i].replace("CELL_", "") for i in plot_idx]

    bars1 = ax.bar(xi - width/2, before_vals, width, label="Before Offloading",
                   color="#FF5722", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(xi + width/2, after_vals, width, label="After Offloading",
                   color="#4CAF50", alpha=0.85, edgecolor="white")

    # 标注变化量
    for i, (b, a) in enumerate(zip(before_vals, after_vals)):
        if abs(b - a) > 0.005:
            ax.annotate(f"-{b-a:.2f}", (xi[i], max(b, a) + 0.02),
                       ha="center", fontsize=8, color="#4CAF50", fontweight="bold")

    ax.axhline(y=np.mean(load_before), color="red", linestyle="--", linewidth=1.5,
               alpha=0.7, label=f"Mean Before={np.mean(load_before):.3f}")
    ax.axhline(y=np.mean(load_after), color="green", linestyle="--", linewidth=1.5,
               alpha=0.7, label=f"Mean After={np.mean(load_after):.3f}")
    ax.set_xticks(xi)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("PRB Utilization (normalized)", fontsize=11)
    ax.set_title(f"Load Distribution Before vs After Offloading — {strategy_name}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Task Offloading Optimization — Before vs After",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_strategy_comparison(df: pd.DataFrame, save_path: str):
    """绘制策略对比图。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    strategies = df["strategy"].tolist()

    # 拥塞节点减少
    ax = axes[0, 0]
    ax.bar(strategies, df["congestion_reduction_pct"], color=["#FF5722", "#2196F3", "#4CAF50", "#FF9800"])
    ax.set_title("Congestion Reduction (%)")
    ax.set_ylabel("Reduction %")
    ax.grid(True, alpha=0.3, axis="y")

    # Jain 公平指数
    ax = axes[0, 1]
    x = np.arange(len(strategies))
    width = 0.35
    ax.bar(x - width/2, df["jain_fairness_before"], width, label="Before", color="#FF5722", alpha=0.7)
    ax.bar(x + width/2, df["jain_fairness_after"], width, label="After", color="#4CAF50", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=8)
    ax.set_title("Jain's Fairness Index")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # 最大负载
    ax = axes[1, 0]
    ax.bar(strategies, df["max_load_after"], color="#2196F3")
    ax.axhline(y=df["max_load_before"].iloc[0], color="red", linestyle="--", label="Max Before")
    ax.set_title("Maximum Load After Offloading")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # 卸载任务数
    ax = axes[1, 1]
    ax.bar(strategies, df["total_decisions"], color="#FF9800")
    ax.set_title("Number of Offloading Decisions")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Offloading Strategy Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_scheduling_timeline(
    result: SchedulingResult,
    save_path: str,
):
    """绘制调度时间线图（只保留任务到达与完成，去掉无变化的队列长度）。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    timeline = result.timeline
    timesteps = [t["timestep"] for t in timeline]
    completed = [t["completed_this_step"] for t in timeline]
    new_tasks = [t["new_tasks"] for t in timeline]
    pending = [t["pending_count"] for t in timeline]

    fig, ax = plt.subplots(figsize=(14, 5))

    x = np.arange(len(timesteps))
    width = 0.35
    ax.bar(x - width/2, new_tasks, width, alpha=0.85, color="#2196F3",
           label="New Tasks", edgecolor="white")
    ax.bar(x + width/2, completed, width, alpha=0.85, color="#4CAF50",
           label="Completed", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(timesteps, fontsize=8)
    ax.set_xlabel("Timestep", fontsize=11)
    ax.set_ylabel("Tasks", fontsize=11)
    ax.set_title("Task Arrival & Completion per Timestep", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis="y")

    # 左上角放统计摘要
    peak_queue = max(pending)
    ax.text(0.02, 0.97,
            f"Total: {result.total_tasks}  |  Completed: {result.completed_tasks}  |  "
            f"Offloaded: {result.offloaded_tasks}  |  Max Queue: {peak_queue}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle(f"Scheduling Timeline — {result.completed_tasks}/{result.total_tasks} completed, "
                 f"{result.offloaded_tasks} offloaded",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_scheduling_policy_comparison(df: pd.DataFrame, save_path: str):
    """绘制调度策略对比图。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    policies = df["policy"].tolist()

    # 完成率
    ax = axes[0]
    bars = ax.bar(policies, df["completion_rate"] * 100, color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"])
    ax.set_title("Task Completion Rate (%)")
    ax.set_ylabel("Completion %")
    ax.set_ylim(0, 105)
    for bar, val in zip(bars, df["completion_rate"] * 100):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # 平均完成时间
    ax = axes[1]
    ax.bar(policies, df["avg_completion_time"], color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"])
    ax.set_title("Avg Completion Time (steps)")
    ax.set_ylabel("Steps")
    ax.grid(True, alpha=0.3, axis="y")

    # SLA 违规
    ax = axes[2]
    ax.bar(policies, df["sla_violations"], color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"])
    ax.set_title("SLA Violations")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Scheduling Policy Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================================
# 主流程
# ============================================================================

def run_pipeline(strategy_name: str = "hybrid", timesteps: int = 30, verbose: bool = True):
    """执行完整的任务卸载与调度 pipeline。"""
    print("\n" + "=" * 70)
    print("  任务卸载与智能调度 Pipeline (Task Offloading & Scheduling)")
    print("=" * 70)

    # ---- 1. 加载数据 ----
    y_true, y_pred, cell_ids, edge_index, edge_weight = load_data()
    num_cells = y_pred.shape[0]

    # 取最后一个时间步的预测作为当前时刻
    predicted_loads = y_pred[:, -1]  # (N,)
    predicted_loads = np.clip(predicted_loads, 0.0, 1.0)

    print(f"\n  Nodes: {num_cells}")
    print(f"  Predicted load range: [{predicted_loads.min():.3f}, {predicted_loads.max():.3f}]")
    print(f"  Mean predicted load: {predicted_loads.mean():.3f}")

    # ---- 2. 策略对比：任务卸载 ----
    print(f"\n{'='*60}")
    print("Step 1: 任务卸载策略对比")
    print("=" * 60)

    base_cfg = OffloadingConfig(
        congestion_threshold=cfg.congestion_threshold,
        max_offload_ratio=cfg.max_offload_ratio,
        max_targets_per_source=cfg.max_targets,
        max_hop_distance=cfg.max_hop,
    )

    cmp_df = compare_strategies(predicted_loads, edge_index, edge_weight, base_cfg)
    print("\n策略对比结果:")
    print(cmp_df.to_string(index=False))

    cmp_csv = os.path.join(cfg.output_dir, "offloading_strategy_comparison.csv")
    cmp_df.to_csv(cmp_csv, index=False)
    print(f"\n  Saved: {cmp_csv}")

    plot_strategy_comparison(
        cmp_df,
        os.path.join(cfg.figure_dir, "offloading_strategy_comparison.png"),
    )

    # ---- 3. 使用选定策略执行详细卸载 ----
    print(f"\n{'='*60}")
    print(f"Step 2: 使用 {strategy_name} 策略执行详细卸载")
    print("=" * 60)

    strategy_map = {
        "greedy": OffloadingStrategy.GREEDY,
        "load_balance": OffloadingStrategy.LOAD_BALANCE,
        "latency_aware": OffloadingStrategy.LATENCY_AWARE,
        "hybrid": OffloadingStrategy.HYBRID,
    }
    selected = strategy_map.get(strategy_name, OffloadingStrategy.HYBRID)
    base_cfg.strategy = selected
    result = run_offloading(predicted_loads, edge_index, edge_weight, base_cfg)

    print("\n" + result.summary())

    # 可视化
    plot_offloading_comparison(
        result.load_before, result.load_after,
        cell_ids, strategy_name,
        os.path.join(cfg.figure_dir, f"offloading_{strategy_name}_comparison.png"),
    )

    # 保存详细决策
    decisions_df = pd.DataFrame([
        {
            "source": d.source_node,
            "target": d.target_node,
            "amount": d.offload_amount,
            "strategy": d.strategy_used,
        }
        for d in result.decisions
    ])
    dec_csv = os.path.join(cfg.output_dir, f"offloading_decisions_{strategy_name}.csv")
    decisions_df.to_csv(dec_csv, index=False)
    print(f"  Saved: {dec_csv}")

    # ---- 4. 智能调度仿真 ----
    print(f"\n{'='*60}")
    print("Step 3: 智能调度仿真")
    print("=" * 60)

    # 扩展预测负载到时间序列
    num_sim_ts = min(timesteps, y_pred.shape[1])
    predicted_seq = y_pred[:, :num_sim_ts].T  # (T, N)

    # 生成任务流
    mean_loads = predicted_seq.mean(axis=0)
    task_stream = generate_tasks(num_cells, mean_loads, num_sim_ts)

    print(f"\n  生成 {sum(len(ts) for ts in task_stream)} 个任务 "
          f"({num_sim_ts} 时间步)")

    # 调度策略对比
    print(f"\n{'='*60}")
    print("Step 4: 调度策略对比")
    print("=" * 60)

    sched_cmp = compare_scheduling_policies(
        predicted_seq, edge_index, edge_weight, task_stream
    )
    print("\n调度策略对比结果:")
    print(sched_cmp.to_string(index=False))

    sched_csv = os.path.join(cfg.output_dir, "scheduling_policy_comparison.csv")
    sched_cmp.to_csv(sched_csv, index=False)
    print(f"\n  Saved: {sched_csv}")

    plot_scheduling_policy_comparison(
        sched_cmp,
        os.path.join(cfg.figure_dir, "scheduling_policy_comparison.png"),
    )

    # ---- 5. 使用自适应策略执行详细调度 ----
    print(f"\n{'='*60}")
    print("Step 5: 自适应调度详细执行")
    print("=" * 60)

    sched_cfg = SchedulingConfig(policy=SchedulingPolicy.ADAPTIVE)
    # 使用独立副本，避免被前面 compare_scheduling_policies 污染任务状态
    import copy
    cloned_stream = [[copy.deepcopy(t) for t in ts] for ts in task_stream]
    sched_result = run_scheduler(
        predicted_seq, edge_index, edge_weight,
        offload_config=base_cfg, sched_config=sched_cfg,
        task_stream=cloned_stream, verbose=verbose,
    )

    print("\n" + sched_result.summary())

    plot_scheduling_timeline(
        sched_result,
        os.path.join(cfg.figure_dir, "scheduling_timeline.png"),
    )

    # ---- 6. 最终总结 ----
    print(f"\n{'='*70}")
    print("  Pipeline 执行完毕!")
    print("=" * 70)
    _print_final_summary(cmp_df, sched_cmp, result, sched_result)


def _print_final_summary(
    offload_cmp: pd.DataFrame,
    sched_cmp: pd.DataFrame,
    offload_result: OffloadingResult,
    sched_result: SchedulingResult,
):
    """打印最终总结报告。"""
    # 最佳卸载策略：拥塞减少最多
    best_offload = offload_cmp.sort_values("congestion_reduction_pct", ascending=False).iloc[0]
    # 最佳调度策略：完成率最高，SLA 违规最少为 tiebreaker
    best_sched = sched_cmp.sort_values(
        ["completion_rate", "sla_violations"], ascending=[False, True]
    ).iloc[0]

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    最终总结 (Final Summary)                         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  📊 任务卸载优化 (Task Offloading)                                   ║
║  ─────────────────────────────────                                  ║
║  最佳策略: {best_offload['strategy']:<20s} (拥塞减少 {best_offload['congestion_reduction_pct']:.1f}%)                   ║
║  公平指数: {best_offload['jain_fairness_before']:.4f} → {best_offload['jain_fairness_after']:.4f}                         ║
║  最大负载: {best_offload['max_load_before']:.3f} → {best_offload['max_load_after']:.3f}                               ║
║                                                                      ║
║  📋 智能调度 (Intelligent Scheduling)                                ║
║  ────────────────────────────────                                   ║
║  最佳策略: {best_sched['policy']:<20s} (完成率 {best_sched['completion_rate']*100:.1f}%)                  ║
║  平均延迟: {best_sched['avg_completion_time']:.2f} steps  |  SLA 违规: {best_sched['sla_violations']}                               ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")


# ============================================================================
# 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="任务卸载与智能调度 Pipeline"
    )
    parser.add_argument(
        "--strategy", type=str, default="hybrid",
        choices=["greedy", "load_balance", "latency_aware", "hybrid"],
        help="卸载策略 (默认: hybrid)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=30,
        help="调度仿真时间步数 (默认: 30)",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="详细输出",
    )
    args = parser.parse_args()

    run_pipeline(
        strategy_name=args.strategy,
        timesteps=args.timesteps,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
