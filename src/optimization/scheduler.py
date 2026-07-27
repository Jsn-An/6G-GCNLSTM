"""
智能调度模块 (Intelligent Task Scheduler)

在任务卸载基础上，实现考虑时间序列动态的智能调度：
1. 基于滑动窗口的连续时刻调度
2. 任务优先级排队
3. 资源预留与 SLA 保障

调度策略包括：
- 先到先服务 (FCFS)
- 最短任务优先 (SJF)
- 优先级调度 (Priority-based)
- 自适应阈值动态调度

作者：HKU Project
日期：2026-07-27
"""

import os
import sys
import copy
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.optimization.offloading import (
    OffloadingConfig,
    OffloadingStrategy,
    OffloadingDecision,
    OffloadingResult,
    run_offloading,
    detect_congestion,
    compute_k_hop_neighbors,
)


# ============================================================================
# 数据结构
# ============================================================================

class SchedulingPolicy(Enum):
    """调度策略枚举。"""
    FCFS = "fcfs"               # 先到先服务
    SJF = "sjf"                  # 最短任务优先
    PRIORITY = "priority"       # 优先级调度
    ADAPTIVE = "adaptive"       # 自适应动态调度


class TaskPriority(Enum):
    """任务优先级。"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Task:
    """单个任务的数据结构。"""
    task_id: int
    source_node: int
    load: float                          # 所需计算负载（归一化）
    priority: TaskPriority = TaskPriority.NORMAL
    latency_req_ms: float = 100.0        # 时延要求 (ms)
    arrival_time: int = 0                # 到达时间步
    deadline: int | None = None          # 截止时间步
    slice_type: str = "eMBB"            # 网络切片类型
    assigned_node: int | None = None    # 分配到的节点
    completed_time: int | None = None   # 完成时间步
    status: str = "pending"             # pending/assigned/completed/dropped


@dataclass
class NodeState:
    """节点运行时状态。"""
    node_id: int
    current_load: float = 0.0
    max_capacity: float = 1.0
    task_queue: deque[Task] = field(default_factory=deque)
    processing_rate: float = 0.08         # 每时间步处理能力（降低以制造资源竞争）
    reserved_load: float = 0.0           # 已预留负载
    sla_violations: int = 0              # SLA 违规计数


@dataclass
class SchedulingConfig:
    """调度配置。"""
    policy: SchedulingPolicy = SchedulingPolicy.ADAPTIVE
    lookahead_window: int = 5            # 前瞻窗口大小
    reservation_ratio: float = 0.1       # 预留容量比例
    sla_latency_margin: float = 1.2      # SLA 时延安全系数
    max_queue_length: int = 20           # 最大排队长度
    dynamic_threshold_alpha: float = 0.3 # 动态阈值平滑系数


# ============================================================================
# 任务生成器
# ============================================================================

def generate_tasks(
    num_nodes: int,
    predicted_loads: np.ndarray,
    num_timesteps: int,
    task_rate: float = 0.5,              # 每时间步每节点生成任务的概率（提高以制造资源竞争）
    base_load: float = 0.06,             # 基础任务负载
    seed: int = 42,
) -> list[list[Task]]:
    """生成时间序列任务流。

    任务生成概率与节点预测负载成正比，模拟高负载节点产生更多任务。

    Args:
        num_nodes: 节点数量
        predicted_loads: (N,) 各节点预测负载
        num_timesteps: 模拟时间步数
        task_rate: 基础任务生成率
        base_load: 基础任务负载
        seed: 随机种子

    Returns:
        task_stream[t][*]: 时间步 t 到达的所有任务列表
    """
    rng = np.random.default_rng(seed)
    slice_types = ["eMBB", "URLLC", "mMTC", "HC"]
    slice_priorities = {
        "URLLC": TaskPriority.CRITICAL,
        "HC": TaskPriority.HIGH,
        "eMBB": TaskPriority.NORMAL,
        "mMTC": TaskPriority.LOW,
    }
    slice_latency = {
        "URLLC": 10.0,
        "HC": 50.0,
        "eMBB": 100.0,
        "mMTC": 500.0,
    }

    task_id = 0
    task_stream: list[list[Task]] = []

    for t in range(num_timesteps):
        timestep_tasks: list[Task] = []
        for node in range(num_nodes):
            # 生成概率与负载正相关
            prob = task_rate * predicted_loads[node]
            if rng.random() < prob:
                num_new = rng.integers(1, 4)  # 每次 1-3 个任务
                for _ in range(num_new):
                    stype = rng.choice(slice_types)
                    # 任务负载与节点当前预测负载相关
                    load = base_load * (1 + rng.random())
                    # 截止时间（高优先级任务 deadline 更紧）
                    if stype in ("URLLC", "HC"):
                        deadline = t + rng.integers(1, 4)   # 1-3 步内必须完成
                    else:
                        deadline = t + rng.integers(2, 8)   # 2-7 步内完成
                    task = Task(
                        task_id=task_id,
                        source_node=node,
                        load=float(load),
                        priority=slice_priorities[stype],
                        latency_req_ms=float(slice_latency[stype]),
                        arrival_time=t,
                        deadline=deadline,
                        slice_type=stype,
                    )
                    timestep_tasks.append(task)
                    task_id += 1
        task_stream.append(timestep_tasks)

    return task_stream


# ============================================================================
# 调度策略
# ============================================================================

def _fcfs_schedule(
    pending_tasks: list[Task],
    node_states: list[NodeState],
    predicted_loads: np.ndarray,
    config: SchedulingConfig,
) -> list[Task]:
    """先到先服务：按到达时间排序。"""
    return sorted(pending_tasks, key=lambda t: t.arrival_time)


def _sjf_schedule(
    pending_tasks: list[Task],
    node_states: list[NodeState],
    predicted_loads: np.ndarray,
    config: SchedulingConfig,
) -> list[Task]:
    """最短任务优先：按任务负载升序。"""
    return sorted(pending_tasks, key=lambda t: t.load)


def _priority_schedule(
    pending_tasks: list[Task],
    node_states: list[NodeState],
    predicted_loads: np.ndarray,
    config: SchedulingConfig,
) -> list[Task]:
    """优先级调度：优先级高 > 截止时间近 > 到达早。"""
    return sorted(
        pending_tasks,
        key=lambda t: (-t.priority.value, t.deadline or 9999, t.arrival_time),
    )


def _adaptive_schedule(
    pending_tasks: list[Task],
    node_states: list[NodeState],
    predicted_loads: np.ndarray,
    config: SchedulingConfig,
) -> list[Task]:
    """自适应调度：动态计算每个任务的紧急度分数。

    分数 = α * urgency + β * priority + γ * efficiency
    - urgency: 基于该任务自身 deadline 窗口的紧迫程度
    - priority: 任务优先级
    - efficiency: 任务负载越小越高效
    """

    def score(task: Task) -> float:
        # 紧急度：基于任务自身的 deadline - arrival 时间窗口
        total_window = (task.deadline - task.arrival_time) if task.deadline is not None else 20
        if total_window > 0:
            urgency = 1.0 / (total_window + 1.0)
        else:
            urgency = 0.5

        # 优先级归一化
        pri_norm = task.priority.value / 3.0

        # 效率（短任务更高效）
        efficiency = 1.0 / (task.load + 0.1)

        return 0.4 * urgency + 0.35 * pri_norm + 0.25 * efficiency

    return sorted(pending_tasks, key=score, reverse=True)


# ============================================================================
# 节点分配
# ============================================================================

def find_best_node_for_task(
    task: Task,
    node_states: list[NodeState],
    k_hop_neighbors: list[set[int]],
    predicted_loads: np.ndarray,
    config: SchedulingConfig,
) -> int | None:
    """为任务寻找最佳执行节点。

    优先考虑:
    1. 源节点自身（如果负载允许）
    2. k-hop 邻居中负载最低的
    3. 考虑 SLA 时延约束
    """
    src = task.source_node
    num_nodes = len(node_states)

    # 检查源节点
    if (node_states[src].reserved_load + task.load
            < node_states[src].max_capacity * (1 - config.reservation_ratio)):
        return src

    # 搜索邻居
    candidates = list(k_hop_neighbors[src]) + [src]
    # 扩展到更远节点
    if len(candidates) <= 1:
        all_nodes = list(range(num_nodes))
        candidates = sorted(all_nodes, key=lambda n: abs(predicted_loads[n] - 0.3))

    best_node = None
    best_score = -float("inf")

    for node in candidates:
        ns = node_states[node]
        available = ns.max_capacity * (1 - config.reservation_ratio) - ns.reserved_load
        if available < task.load:
            continue

        # 综合评分: 容量越大越好，负载越小越好
        score = available / ns.max_capacity - predicted_loads[node] * 0.3
        if score > best_score:
            best_score = score
            best_node = node

    return best_node


# ============================================================================
# 主调度流程
# ============================================================================

@dataclass
class SchedulingResult:
    """调度结果汇总。"""
    total_tasks: int = 0
    completed_tasks: int = 0
    offloaded_tasks: int = 0
    dropped_tasks: int = 0
    sla_violations: int = 0
    avg_completion_time: float = 0.0
    avg_waiting_time: float = 0.0
    load_history: np.ndarray = field(default_factory=lambda: np.array([]))
    timeline: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        """生成调度汇总报告。"""
        lines = [
            "=" * 60,
            "    智能调度结果 (Scheduling Result)",
            "=" * 60,
            f"  总任务数:        {self.total_tasks}",
            f"  已完成:          {self.completed_tasks}",
            f"  已卸载:          {self.offloaded_tasks}",
            f"  丢弃:            {self.dropped_tasks}",
            f"  SLA 违规:        {self.sla_violations}",
            f"  完成率:          {self.completed_tasks / max(self.total_tasks, 1) * 100:.2f}%",
            f"  平均完成时间:     {self.avg_completion_time:.2f} steps",
            f"  平均等待时间:     {self.avg_waiting_time:.2f} steps",
            "=" * 60,
        ]
        return "\n".join(lines)


def run_scheduler(
    predicted_loads_seq: np.ndarray,          # (T, N) 预测负载时间序列
    edge_index: np.ndarray,                    # (2, E)
    edge_weight: np.ndarray | None = None,
    offload_config: OffloadingConfig | None = None,
    sched_config: SchedulingConfig | None = None,
    task_stream: list[list[Task]] | None = None,
    verbose: bool = True,
) -> SchedulingResult:
    """运行智能调度仿真。

    Args:
        predicted_loads_seq: (T, N) 每个时间步各节点的预测 PRB 利用率
        edge_index: 图边索引
        edge_weight: 边权重
        offload_config: 卸载配置
        sched_config: 调度配置
        task_stream: 预生成的任务流，若为 None 则自动生成
        verbose: 是否打印进度

    Returns:
        SchedulingResult
    """
    if offload_config is None:
        offload_config = OffloadingConfig()
    if sched_config is None:
        sched_config = SchedulingConfig()

    num_timesteps, num_nodes = predicted_loads_seq.shape

    # 初始化节点状态
    node_states = [NodeState(node_id=i) for i in range(num_nodes)]

    # 计算 k-hop 邻域
    ei = edge_index if isinstance(edge_index, np.ndarray) else edge_index.numpy()
    k_hop = compute_k_hop_neighbors(ei, num_nodes, offload_config.max_hop_distance)

    # 生成任务流（如果未提供）
    if task_stream is None:
        mean_loads = predicted_loads_seq.mean(axis=0)
        task_stream = generate_tasks(num_nodes, mean_loads, num_timesteps)

    # 调度策略函数
    policy_map = {
        SchedulingPolicy.FCFS: _fcfs_schedule,
        SchedulingPolicy.SJF: _sjf_schedule,
        SchedulingPolicy.PRIORITY: _priority_schedule,
        SchedulingPolicy.ADAPTIVE: _adaptive_schedule,
    }
    sort_func = policy_map.get(sched_config.policy, _adaptive_schedule)

    # 统计
    total_tasks = 0
    completed_tasks = 0
    offloaded_count = 0
    dropped_count = 0
    sla_violations = 0
    completion_times: list[float] = []
    waiting_times: list[float] = []
    load_history: list[np.ndarray] = []
    timeline: list[dict] = []

    # 全局待处理队列
    global_pending: list[Task] = []

    for t in range(num_timesteps):
        pred_t = predicted_loads_seq[t]

        # ---- Step 1: 新任务到达 ----
        new_tasks = task_stream[t] if t < len(task_stream) else []
        global_pending.extend(new_tasks)
        total_tasks += len(new_tasks)

        # ---- Step 2: 任务卸载决策 ----
        # 基于当前预测负载执行卸载，将拥塞源节点的 pending 任务
        # 优先分配至卸载目标节点（而非留在源节点排队）
        offload_result = run_offloading(pred_t, ei, edge_weight, offload_config)
        offload_targets: dict[int, list[int]] = {}
        for d in offload_result.decisions:
            offload_targets.setdefault(d.source_node, []).append(d.target_node)

        for task in global_pending:
            if task.status != "pending" or task.source_node not in offload_targets:
                continue
            assigned = False
            for target in offload_targets[task.source_node]:
                ns = node_states[target]
                if ns.reserved_load + task.load < ns.max_capacity * (1 - sched_config.reservation_ratio):
                    task.assigned_node = target
                    task.status = "assigned"
                    ns.reserved_load += task.load
                    ns.task_queue.append(task)
                    offloaded_count += 1
                    assigned = True
                    break
            if not assigned:
                # 无法卸载，留到 Step 4 正常分配
                pass

        # ---- Step 3: 任务排序 ----
        pending = [t for t in global_pending if t.status == "pending"]
        sorted_tasks = sort_func(pending, node_states, pred_t, sched_config)

        # ---- Step 4: 任务分配 ----
        for task in sorted_tasks:
            if task.status != "pending":
                continue
            # 检查队列长度
            if task.source_node < num_nodes:
                ns = node_states[task.source_node]
                if len(ns.task_queue) >= sched_config.max_queue_length:
                    task.status = "dropped"
                    dropped_count += 1
                    continue

            best_node = find_best_node_for_task(
                task, node_states, k_hop, pred_t, sched_config
            )
            if best_node is not None:
                task.assigned_node = best_node
                task.status = "assigned"
                node_states[best_node].reserved_load += task.load
                node_states[best_node].task_queue.append(task)
                if best_node != task.source_node:
                    offloaded_count += 1
            else:
                # 无法分配，根据优先级决定是否丢弃
                if task.priority.value >= TaskPriority.HIGH.value:
                    # 高优先级任务强制放入源节点队列
                    ns = node_states[task.source_node]
                    task.assigned_node = task.source_node
                    task.status = "assigned"
                    ns.task_queue.append(task)
                    ns.reserved_load += task.load
                else:
                    task.status = "dropped"
                    dropped_count += 1

        # ---- Step 5: 处理节点队列 ----
        for ns in node_states:
            processed = 0.0
            max_process = ns.processing_rate
            while ns.task_queue and processed < max_process:
                task = ns.task_queue.popleft()
                remaining = task.load
                # 处理任务
                task.completed_time = t
                task.status = "completed"
                completed_tasks += 1
                completion_times.append(t - task.arrival_time)
                waiting_times.append(t - task.arrival_time - task.load)
                ns.reserved_load = max(0.0, ns.reserved_load - task.load)
                processed += remaining

                # SLA 检查
                if task.deadline is not None and t > task.deadline:
                    ns.sla_violations += 1
                    sla_violations += 1

            # 当前负载 = 剩余未处理任务的负载总和（不重复加）
            ns.current_load = ns.reserved_load

        # 记录负载
        load_history.append(np.array([ns.current_load for ns in node_states]))

        # 时间线记录
        timeline.append({
            "timestep": t,
            "pending_count": len([x for x in global_pending if x.status == "pending"]),
            "new_tasks": len(new_tasks),
            "completed_this_step": sum(
                1 for task in global_pending if task.completed_time == t
            ),
        })

        if verbose and t % max(1, num_timesteps // 10) == 0:
            print(f"  Timestep {t:4d}/{num_timesteps}: "
                  f"pending={len(pending)}, completed={completed_tasks}, "
                  f"offloaded={offloaded_count}, dropped={dropped_count}")

    return SchedulingResult(
        total_tasks=total_tasks,
        completed_tasks=completed_tasks,
        offloaded_tasks=offloaded_count,
        dropped_tasks=dropped_count,
        sla_violations=sla_violations,
        avg_completion_time=float(np.mean(completion_times)) if completion_times else 0.0,
        avg_waiting_time=float(np.mean(waiting_times)) if waiting_times else 0.0,
        load_history=np.array(load_history),
        timeline=timeline,
    )


# ============================================================================
# 调度对比分析
# ============================================================================

def compare_scheduling_policies(
    predicted_loads_seq: np.ndarray,
    edge_index: np.ndarray,
    edge_weight: np.ndarray | None = None,
    task_stream: list[list[Task]] | None = None,
) -> pd.DataFrame:
    """对比不同调度策略的性能。"""
    policies = [
        SchedulingPolicy.FCFS,
        SchedulingPolicy.SJF,
        SchedulingPolicy.PRIORITY,
        SchedulingPolicy.ADAPTIVE,
    ]

    rows = []
    for policy in policies:
        sched_cfg = SchedulingConfig(policy=policy)
        # 每个策略使用独立的任务副本，避免前一个策略修改任务状态
        cloned = [[copy.deepcopy(t) for t in ts] for ts in task_stream] if task_stream else None
        result = run_scheduler(
            predicted_loads_seq, edge_index, edge_weight,
            sched_config=sched_cfg, task_stream=cloned,
            verbose=False,
        )
        rows.append({
            "policy": policy.value,
            "total_tasks": result.total_tasks,
            "completed": result.completed_tasks,
            "offloaded": result.offloaded_tasks,
            "dropped": result.dropped_tasks,
            "sla_violations": result.sla_violations,
            "completion_rate": result.completed_tasks / max(result.total_tasks, 1),
            "avg_completion_time": result.avg_completion_time,
            "avg_waiting_time": result.avg_waiting_time,
        })

    return pd.DataFrame(rows)
