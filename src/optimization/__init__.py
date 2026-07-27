"""
优化模块：任务卸载与智能调度

提供基于 GCN+LSTM 拥塞预测的智能任务卸载和调度功能。

子模块:
- offloading: 任务卸载优化（贪心 / 负载均衡 / 时延感知 / 混合策略）
- scheduler: 智能任务调度（FCFS / SJF / 优先级 / 自适应）

用法:
    from src.optimization.offloading import run_offloading, compare_strategies
    from src.optimization.scheduler import run_scheduler, compare_scheduling_policies
"""

from src.optimization.offloading import (
    OffloadingConfig,
    OffloadingStrategy,
    OffloadingDecision,
    OffloadingResult,
    run_offloading,
    compare_strategies,
    detect_congestion,
)

from src.optimization.scheduler import (
    SchedulingConfig,
    SchedulingPolicy,
    Task,
    TaskPriority,
    SchedulingResult,
    run_scheduler,
    compare_scheduling_policies,
    generate_tasks,
)

__all__ = [
    # offloading
    "OffloadingConfig",
    "OffloadingStrategy",
    "OffloadingDecision",
    "OffloadingResult",
    "run_offloading",
    "compare_strategies",
    "detect_congestion",
    # scheduler
    "SchedulingConfig",
    "SchedulingPolicy",
    "Task",
    "TaskPriority",
    "SchedulingResult",
    "run_scheduler",
    "compare_scheduling_policies",
    "generate_tasks",
]