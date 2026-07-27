"""
任务卸载优化模块 (Task Offloading Optimization)

基于 GCN+LSTM 拥塞预测结果的智能任务卸载与资源调度。

功能：
1. 拥塞检测：利用模型预测的 PRB 利用率识别过载节点
2. 卸载决策：选择卸载目标节点与卸载比例
3. 多种卸载策略：贪心、负载均衡、时延感知
4. 性能对比：卸载前后的负载分布、拥塞率等指标

策略说明：
- Greedy：将任务卸载到预测负载最低的邻居节点
- LoadBalance：将负载均匀分摊到所有可达低负载节点
- LatencyAware：考虑边权（通信质量）与时延，选择综合代价最小的目标
- Hybrid：综合以上策略的加权混合

作者：HKU Project
日期：2026-07-27
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# 配置与数据结构
# ============================================================================

class OffloadingStrategy(Enum):
    """卸载策略枚举。"""
    GREEDY = "greedy"
    LOAD_BALANCE = "load_balance"
    LATENCY_AWARE = "latency_aware"
    HYBRID = "hybrid"


@dataclass
class OffloadingConfig:
    """任务卸载配置。"""

    # ---- 拥塞检测 ----
    congestion_threshold: float = 0.70        # PRB 利用率 > 该阈值视为拥塞
    severe_threshold: float = 0.85            # 严重拥塞阈值

    # ---- 卸载约束 ----
    max_offload_ratio: float = 0.50           # 单节点最大卸载比例
    min_remaining_load: float = 0.20          # 卸载后源节点最低保留负载

    # ---- 目标选择 ----
    max_targets_per_source: int = 3           # 每个源节点最多选择的卸载目标数
    max_hop_distance: int = 2                 # 最大跳数（基于图拓扑）
    target_load_threshold: float = 0.50       # 仅当目标预测负载低于此值时才接收任务

    # ---- 策略参数 ----
    strategy: OffloadingStrategy = OffloadingStrategy.HYBRID
    greedy_weight: float = 0.4                # 贪心权重 (HYBRID)
    balance_weight: float = 0.3              # 负载均衡权重 (HYBRID)
    latency_weight: float = 0.3               # 时延感知权重 (HYBRID)

    # ---- 仿真参数 ----
    task_load_unit: float = 0.05              # 每个任务单元的负载（归一化）


@dataclass
class OffloadingDecision:
    """单条卸载决策记录。"""
    source_node: int
    target_node: int
    offload_amount: float                     # 卸载负载量（归一化）
    strategy_used: str
    cost: float = 0.0


@dataclass
class OffloadingResult:
    """卸载结果汇总。"""
    decisions: list[OffloadingDecision] = field(default_factory=list)
    load_before: np.ndarray = field(default_factory=lambda: np.array([]))
    load_after: np.ndarray = field(default_factory=lambda: np.array([]))
    congested_before: np.ndarray = field(default_factory=lambda: np.array([]))
    congested_after: np.ndarray = field(default_factory=lambda: np.array([]))
    metrics: dict = field(default_factory=dict)

    def summary(self) -> str:
        """生成可读的汇总报告。"""
        lines = [
            "=" * 60,
            "    任务卸载优化结果 (Task Offloading Result)",
            "=" * 60,
        ]
        for k, v in self.metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k:>30s}: {v:.4f}")
            else:
                lines.append(f"  {k:>30s}: {v}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================================
# 图拓扑工具
# ============================================================================

def compute_k_hop_neighbors(
    edge_index: np.ndarray,
    num_nodes: int,
    max_hop: int = 2,
) -> list[set[int]]:
    """计算每个节点的 k-hop 邻居集合。

    使用 BFS 从 (2, E) 的 edge_index 构建邻接表，
    然后对每个节点计算 k-hop 可达集合。

    Args:
        edge_index: (2, E) 边索引数组
        num_nodes: 图中节点总数
        max_hop: 最大跳数

    Returns:
        neighbors[i]: 节点 i 在 max_hop 内的邻居集合（不含自身）
    """
    # 构建邻接表
    adj = [[] for _ in range(num_nodes)]
    for u, v in edge_index.T:
        u, v = int(u), int(v)
        if u < num_nodes and v < num_nodes and u != v:
            adj[u].append(v)
            adj[v].append(u)  # 无向图

    def bfs_k_hop(start: int, k: int) -> set[int]:
        visited: set[int] = set()
        frontier = {start}
        for _ in range(k):
            next_frontier: set[int] = set()
            for node in frontier:
                for nb in adj[node]:
                    if nb != start and nb not in visited:
                        next_frontier.add(nb)
                        visited.add(nb)
            frontier = next_frontier
        return visited

    return [bfs_k_hop(i, max_hop) for i in range(num_nodes)]


def compute_shortest_path_distances(
    edge_index: np.ndarray,
    num_nodes: int,
    edge_weight: np.ndarray | None = None,
) -> np.ndarray:
    """计算节点间的最短路径距离（基于 BFS 跳数）。

    Args:
        edge_index: (2, E) 边索引
        edge_weight: (E,) 边权重（越高表示连接越强，距离越短）

    Returns:
        dist: (N, N) 最短路径距离矩阵，不可达为 inf
    """
    adj = [[] for _ in range(num_nodes)]
    for idx, (u, v) in enumerate(edge_index.T):
        u, v = int(u), int(v)
        if u < num_nodes and v < num_nodes and u != v:
            w = 1.0 / max(edge_weight[idx], 1e-6) if edge_weight is not None else 1.0
            adj[u].append((v, w))
            adj[v].append((u, w))

    INF = float("inf")
    dist = np.full((num_nodes, num_nodes), INF)

    for src in range(num_nodes):
        visited = [False] * num_nodes
        cur_dist = [INF] * num_nodes
        cur_dist[src] = 0.0

        for _ in range(num_nodes):
            # 选最小距离未访问节点
            min_d = INF
            u = -1
            for i in range(num_nodes):
                if not visited[i] and cur_dist[i] < min_d:
                    min_d = cur_dist[i]
                    u = i
            if u == -1:
                break
            visited[u] = True
            for v, w in adj[u]:
                if not visited[v]:
                    new_d = cur_dist[u] + w
                    if new_d < cur_dist[v]:
                        cur_dist[v] = new_d
        dist[src] = np.array(cur_dist)

    return dist


# ============================================================================
# 拥塞检测
# ============================================================================

def detect_congestion(
    predicted_loads: np.ndarray,       # (num_nodes,) — 预测的 PRB 利用率
    threshold: float = 0.70,
    severe_threshold: float = 0.85,
) -> tuple[np.ndarray, np.ndarray]:
    """检测拥塞节点。

    Args:
        predicted_loads: 每个节点的预测 PRB 利用率
        threshold: 一般拥塞阈值
        severe_threshold: 严重拥塞阈值

    Returns:
        congested_mask: (N,) bool 数组，True=该节点拥塞
        severity: (N,) int 数组，0=正常, 1=拥塞, 2=严重拥塞
    """
    congested = predicted_loads > threshold
    severity = np.zeros_like(predicted_loads, dtype=np.int32)
    severity[predicted_loads > threshold] = 1
    severity[predicted_loads > severe_threshold] = 2
    return congested, severity


# ============================================================================
# 卸载策略实现
# ============================================================================

def _find_candidate_targets(
    source: int,
    predicted_loads: np.ndarray,
    k_hop_neighbors: list[set[int]],
    cfg: OffloadingConfig,
) -> list[int]:
    """为源节点查找候选卸载目标。

    条件: 在 k-hop 范围内 且 预测负载 < target_load_threshold
    """
    candidates = []
    for nb in sorted(k_hop_neighbors[source]):
        if predicted_loads[nb] < cfg.target_load_threshold:
            candidates.append(nb)
    return candidates[: cfg.max_targets_per_source]


def _greedy_offload(
    source: int,
    predicted_loads: np.ndarray,
    k_hop_neighbors: list[set[int]],
    cfg: OffloadingConfig,
) -> list[OffloadingDecision]:
    """贪心策略：卸载到预测负载最低的候选目标。"""
    candidates = _find_candidate_targets(source, predicted_loads, k_hop_neighbors, cfg)
    if not candidates:
        return []

    # 按预测负载升序
    candidates_sorted = sorted(candidates, key=lambda c: predicted_loads[c])

    decisions = []
    src_load = predicted_loads[source]
    excess = max(0.0, src_load - cfg.congestion_threshold)
    offload_remaining = min(excess, src_load * cfg.max_offload_ratio)

    for target in candidates_sorted:
        if offload_remaining <= 0:
            break
        available = cfg.target_load_threshold - predicted_loads[target]
        amount = min(cfg.task_load_unit, offload_remaining, available)
        if amount <= 0:
            continue

        decisions.append(OffloadingDecision(
            source_node=source,
            target_node=target,
            offload_amount=float(amount),
            strategy_used="greedy",
        ))
        offload_remaining -= amount

    return decisions


def _load_balance_offload(
    source: int,
    predicted_loads: np.ndarray,
    k_hop_neighbors: list[set[int]],
    cfg: OffloadingConfig,
) -> list[OffloadingDecision]:
    """负载均衡策略：将超额负载均匀分摊到所有候选目标。"""
    candidates = _find_candidate_targets(source, predicted_loads, k_hop_neighbors, cfg)
    if not candidates:
        return []

    src_load = predicted_loads[source]
    excess = max(0.0, src_load - cfg.congestion_threshold)
    offload_total = min(excess, src_load * cfg.max_offload_ratio)

    decisions = []
    per_target = offload_total / len(candidates)

    for target in candidates:
        available = cfg.target_load_threshold - predicted_loads[target]
        amount = min(per_target, available)
        # 至少一个任务单元
        amount = max(amount, cfg.task_load_unit) if amount >= cfg.task_load_unit else 0.0
        if amount <= 0:
            continue
        decisions.append(OffloadingDecision(
            source_node=source,
            target_node=target,
            offload_amount=float(amount),
            strategy_used="load_balance",
        ))

    return decisions


def _latency_aware_offload(
    source: int,
    predicted_loads: np.ndarray,
    k_hop_neighbors: list[set[int]],
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None,
    cfg: OffloadingConfig,
) -> list[OffloadingDecision]:
    """时延感知策略：综合通信代价和预测负载选择卸载目标。

    代价函数: cost = α * normalized_distance + (1-α) * target_load
    """
    candidates = _find_candidate_targets(source, predicted_loads, k_hop_neighbors, cfg)
    if not candidates:
        return []

    # 计算距离
    dist = compute_shortest_path_distances(edge_index, len(predicted_loads), edge_weight)
    max_dist = float(dist[dist < float("inf")].max()) if dist.any() else 1.0

    # 对每个候选计算综合代价
    candidate_costs = []
    for target in candidates:
        d = dist[source, target]
        norm_dist = d / max_dist if max_dist > 0 else 0.0
        norm_load = float(predicted_loads[target])
        cost = 0.5 * norm_dist + 0.5 * norm_load
        candidate_costs.append((target, cost))

    candidate_costs.sort(key=lambda x: x[1])

    src_load = predicted_loads[source]
    excess = max(0.0, src_load - cfg.congestion_threshold)
    offload_remaining = min(excess, src_load * cfg.max_offload_ratio)

    decisions = []
    for target, cost in candidate_costs:
        if offload_remaining <= 0:
            break
        available = cfg.target_load_threshold - predicted_loads[target]
        amount = min(cfg.task_load_unit, offload_remaining, available)
        if amount <= 0:
            continue
        decisions.append(OffloadingDecision(
            source_node=source,
            target_node=target,
            offload_amount=float(amount),
            strategy_used="latency_aware",
            cost=cost,
        ))
        offload_remaining -= amount

    return decisions


def _hybrid_offload(
    source: int,
    predicted_loads: np.ndarray,
    k_hop_neighbors: list[set[int]],
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None,
    cfg: OffloadingConfig,
) -> list[OffloadingDecision]:
    """混合策略：对多种策略加权打分后选择最优目标。"""
    candidates = _find_candidate_targets(source, predicted_loads, k_hop_neighbors, cfg)
    if not candidates:
        return []

    num_nodes = len(predicted_loads)
    dist = compute_shortest_path_distances(edge_index, num_nodes, edge_weight)
    max_dist = float(dist[dist < float("inf")].max()) if dist.any() else 1.0

    scores = {}
    for target in candidates:
        # 贪心分数：负载越低越好
        greedy_score = 1.0 - float(predicted_loads[target])

        # 负载均衡分数：候选的平均负载与全局平均的偏差越小越好
        candidate_loads = [predicted_loads[c] for c in candidates]
        avg_candidate_load = np.mean(candidate_loads)
        balance_score = 1.0 - abs(float(predicted_loads[target]) - avg_candidate_load)

        # 时延分数：距离越近越好
        d = dist[source, target]
        latency_score = 1.0 - (d / max_dist if max_dist > 0 else 0.0)
        latency_score = max(latency_score, 0.0)

        total = (
            cfg.greedy_weight * greedy_score
            + cfg.balance_weight * balance_score
            + cfg.latency_weight * latency_score
        )
        scores[target] = total

    # 按综合分数降序
    sorted_targets = sorted(scores, key=scores.get, reverse=True)

    src_load = predicted_loads[source]
    excess = max(0.0, src_load - cfg.congestion_threshold)
    offload_remaining = min(excess, src_load * cfg.max_offload_ratio)

    decisions = []
    for target in sorted_targets:
        if offload_remaining <= 0:
            break
        available = cfg.target_load_threshold - predicted_loads[target]
        amount = min(cfg.task_load_unit, offload_remaining, available)
        if amount <= 0:
            continue
        decisions.append(OffloadingDecision(
            source_node=source,
            target_node=target,
            offload_amount=float(amount),
            strategy_used="hybrid",
            cost=1.0 - scores[target],
        ))
        offload_remaining -= amount

    return decisions


# ============================================================================
# 主卸载流程
# ============================================================================

def run_offloading(
    predicted_loads: np.ndarray,
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None = None,
    config: OffloadingConfig | None = None,
) -> OffloadingResult:
    """执行任务卸载优化。

    Args:
        predicted_loads: (N,) 各节点的预测 PRB 利用率
        edge_index: (2, E) 图边索引
        edge_weight: (E,) 边权重
        config: 卸载配置

    Returns:
        OffloadingResult 包含所有卸载决策和前后对比指标
    """
    if config is None:
        config = OffloadingConfig()

    num_nodes = len(predicted_loads)
    load_before = predicted_loads.copy()
    load_after = predicted_loads.copy()

    # 1. 拥塞检测
    congested_before, severity = detect_congestion(
        predicted_loads, config.congestion_threshold, config.severe_threshold
    )
    congested_nodes = np.where(congested_before)[0]

    # 2. 计算 k-hop 邻域
    k_hop_neighbors = compute_k_hop_neighbors(
        edge_index, num_nodes, config.max_hop_distance
    )

    # 3. 对每个拥塞节点执行卸载
    all_decisions: list[OffloadingDecision] = []

    for src in sorted(congested_nodes, key=lambda i: predicted_loads[i], reverse=True):
        strategy = config.strategy

        if strategy == OffloadingStrategy.GREEDY:
            decisions = _greedy_offload(src, load_after, k_hop_neighbors, config)
        elif strategy == OffloadingStrategy.LOAD_BALANCE:
            decisions = _load_balance_offload(src, load_after, k_hop_neighbors, config)
        elif strategy == OffloadingStrategy.LATENCY_AWARE:
            decisions = _latency_aware_offload(
                src, load_after, k_hop_neighbors, edge_index, edge_weight, config
            )
        elif strategy == OffloadingStrategy.HYBRID:
            decisions = _hybrid_offload(
                src, load_after, k_hop_neighbors, edge_index, edge_weight, config
            )
        else:
            decisions = []

        # 更新负载状态（按优先级处理，更拥塞的先处理）
        for d in decisions:
            load_after[d.source_node] -= d.offload_amount
            load_after[d.target_node] += d.offload_amount
        all_decisions.extend(decisions)

    # 4. 卸载后拥塞检测
    congested_after, _ = detect_congestion(
        load_after, config.congestion_threshold, config.severe_threshold
    )

    # 5. 计算评估指标
    metrics = _compute_offloading_metrics(
        load_before, load_after, congested_before, congested_after, all_decisions
    )

    return OffloadingResult(
        decisions=all_decisions,
        load_before=load_before,
        load_after=load_after,
        congested_before=congested_before,
        congested_after=congested_after,
        metrics=metrics,
    )


def _compute_offloading_metrics(
    load_before: np.ndarray,
    load_after: np.ndarray,
    congested_before: np.ndarray,
    congested_after: np.ndarray,
    decisions: list[OffloadingDecision],
) -> dict:
    """计算卸载前后的评估指标。"""
    n = len(load_before)

    num_congested_before = int(congested_before.sum())
    num_congested_after = int(congested_after.sum())

    congestion_reduction = (
        (num_congested_before - num_congested_after) / max(num_congested_before, 1) * 100
    )

    mean_load_before = float(np.mean(load_before))
    mean_load_after = float(np.mean(load_after))
    max_load_before = float(np.max(load_before))
    max_load_after = float(np.max(load_after))
    std_before = float(np.std(load_before))
    std_after = float(np.std(load_after))

    # Jain's Fairness Index (值越高越公平)
    jfi_before = float(np.sum(load_before) ** 2 / (n * np.sum(load_before ** 2) + 1e-10))
    jfi_after = float(np.sum(load_after) ** 2 / (n * np.sum(load_after ** 2) + 1e-10))

    total_offloaded = float(sum(d.offload_amount for d in decisions))
    num_offloaded_sources = len(set(d.source_node for d in decisions))
    num_offloaded_targets = len(set(d.target_node for d in decisions))

    return {
        "num_nodes": n,
        "strategy": decisions[0].strategy_used if decisions else "none",
        "total_decisions": len(decisions),
        "num_offloaded_sources": num_offloaded_sources,
        "num_offloaded_targets": num_offloaded_targets,
        "total_offloaded_load": total_offloaded,
        "congested_before": num_congested_before,
        "congested_after": num_congested_after,
        "congestion_reduction_pct": congestion_reduction,
        "mean_load_before": mean_load_before,
        "mean_load_after": mean_load_after,
        "max_load_before": max_load_before,
        "max_load_after": max_load_after,
        "std_before": std_before,
        "std_after": std_after,
        "jain_fairness_before": jfi_before,
        "jain_fairness_after": jfi_after,
    }


# ============================================================================
# 策略对比
# ============================================================================

def compare_strategies(
    predicted_loads: np.ndarray,
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None = None,
    base_config: OffloadingConfig | None = None,
) -> pd.DataFrame:
    """对比所有卸载策略的性能。

    Returns:
        DataFrame，每行一个策略，列为各项指标
    """
    if base_config is None:
        base_config = OffloadingConfig()

    strategies = [
        OffloadingStrategy.GREEDY,
        OffloadingStrategy.LOAD_BALANCE,
        OffloadingStrategy.LATENCY_AWARE,
        OffloadingStrategy.HYBRID,
    ]

    rows = []
    for strat in strategies:
        cfg = OffloadingConfig(
            congestion_threshold=base_config.congestion_threshold,
            severe_threshold=base_config.severe_threshold,
            max_offload_ratio=base_config.max_offload_ratio,
            max_targets_per_source=base_config.max_targets_per_source,
            max_hop_distance=base_config.max_hop_distance,
            target_load_threshold=base_config.target_load_threshold,
            strategy=strat,
            greedy_weight=base_config.greedy_weight,
            balance_weight=base_config.balance_weight,
            latency_weight=base_config.latency_weight,
        )
        result = run_offloading(predicted_loads, edge_index, edge_weight, cfg)
        row = {"strategy": strat.value}
        row.update(result.metrics)
        rows.append(row)

    return pd.DataFrame(rows)
