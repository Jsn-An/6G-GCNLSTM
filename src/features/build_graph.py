"""
图构建模块：基于 KPI 时间序列相关性的 5G 基站网络图构建

本模块实现了以下功能：
1. 从预处理后的数据集中提取每个基站小区的 KPI 时间序列
2. 计算小区之间的多特征时间序列相关性（皮尔逊相关系数）
3. 应用自适应阈值进行边筛选（保留 top 30% 高相关边）
4. 融入 cell_type 和 slice_type 领域先验知识增强边权
5. 输出邻接矩阵（NumPy）和 PyTorch Geometric Data 对象
6. 生成相关性矩阵热力图和网络拓扑可视化

作者：HKU Project
日期：2026-07-27
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，用于服务器环境
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================================
# 配置参数
# ============================================================================

# 项目路径
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# 输入文件
PROCESSED_CSV = os.path.join(PROJECT_ROOT, "datasets", "ds1_processed.csv")

# 输出路径
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")

# 用于计算相关性的 KPI 特征列（使用原始值，非归一化）
KPI_COLUMNS = [
    "throughput_mbps",       # 吞吐量
    "latency_ms",            # 时延
    "packet_loss_pct",       # 丢包率
    "handover_count",        # 切换次数
    "prb_utilization_pct",   # PRB 利用率（关键拥塞指标）
    "active_users",          # 活跃用户数
]

# 图构建参数
CORRELATION_PERCENTILE = 70        # 保留 top 30% 的相关边（百分位数阈值）
CELL_TYPE_PRIOR_WEIGHT = 0.15      # 同 cell_type 的边权偏置
SLICE_TYPE_PRIOR_WEIGHT = 0.10     # 同 slice_type 的边权偏置（取每小区的主切片）
SELF_LOOP = True                   # 是否添加自环


# ============================================================================
# 辅助函数
# ============================================================================

def ensure_dir(path: str) -> None:
    """确保目录存在，不存在则创建。"""
    os.makedirs(path, exist_ok=True)


def load_and_sort_data(csv_path: str) -> pd.DataFrame:
    """加载预处理后的 CSV，按小区和时间排序。

    Args:
        csv_path: CSV 文件路径。

    Returns:
        排序后的 DataFrame。
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values(["cell_id", "timestamp"]).reset_index(drop=True)
    return df


def build_cell_kpi_timeseries(
    df: pd.DataFrame, kpi_columns: list[str]
) -> tuple[dict, np.ndarray, np.ndarray]:
    """从 DataFrame 中提取每个小区的 KPI 时间序列矩阵。

    返回两个关键结构：
    - cell_ts_dict: {cell_id -> DataFrame of time series}
    - ts_matrix: (num_cells, timesteps, num_kpis) 三维数组
    - cell_ids: 排序后的小区 ID 列表

    Args:
        df: 排序后的数据。
        kpi_columns: 要提取的 KPI 列名列表。

    Returns:
        (cell_ts_dict, ts_matrix, cell_ids) 三元组。
    """
    cell_ids = sorted(df["cell_id"].unique())
    num_cells = len(cell_ids)
    num_kpis = len(kpi_columns)

    # 获取时间步数（取所有小区中的最小长度以对齐）
    min_len = min(
        len(df[df["cell_id"] == cid]) for cid in cell_ids
    )

    # 构建 (num_cells, timesteps, num_kpis) 的 3D 数组
    ts_matrix = np.zeros((num_cells, min_len, num_kpis))

    cell_ts_dict = {}
    for i, cid in enumerate(cell_ids):
        cell_df = df[df["cell_id"] == cid].head(min_len)
        cell_ts_dict[cid] = cell_df
        for j, col in enumerate(kpi_columns):
            ts_matrix[i, :, j] = cell_df[col].values

    return cell_ts_dict, ts_matrix, np.array(cell_ids)


def compute_correlation_matrix(
    ts_matrix: np.ndarray, kpi_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """计算所有小区对之间的多特征平均相关系数矩阵。

    对于每对小区 (i, j)，分别计算每个 KPI 的时间序列皮尔逊相关系数，
    然后取绝对值后求平均，得到综合的相关性分数。

    Args:
        ts_matrix: (num_cells, timesteps, num_kpis) 三维数组。
        kpi_names: KPI 名称列表，仅用于打印信息。

    Returns:
        (corr_matrix, per_kpi_corr)：
        - corr_matrix: (num_cells, num_cells) 综合相关性矩阵
        - per_kpi_corr: (num_kpis, num_cells, num_cells) 每个 KPI 的相关性矩阵
    """
    num_cells, _, num_kpis = ts_matrix.shape
    corr_matrix = np.zeros((num_cells, num_cells))
    per_kpi_corr = np.zeros((num_kpis, num_cells, num_cells))

    for k in range(num_kpis):
        # 提取第 k 个 KPI 的 (num_cells, timesteps) 矩阵
        kpi_data = ts_matrix[:, :, k]  # shape: (num_cells, timesteps)

        # 计算相关系数矩阵（向量化，比双重循环快很多）
        # 使用 np.corrcoef，每行是一个变量的观测值
        kpi_corr = np.abs(np.corrcoef(kpi_data))  # (num_cells, num_cells)
        # 处理可能的 NaN（方差为零的情况）
        kpi_corr = np.nan_to_num(kpi_corr, nan=0.0)
        per_kpi_corr[k] = kpi_corr
        corr_matrix += kpi_corr

    # 取平均
    corr_matrix /= num_kpis

    # 打印每个 KPI 的平均相关性
    print("\n各 KPI 的平均小区间相关性：")
    for k, name in enumerate(kpi_names):
        # 取上三角（排除对角线）
        upper = per_kpi_corr[k][np.triu_indices(num_cells, k=1)]
        print(f"  {name:25s}: mean={upper.mean():.4f}, std={upper.std():.4f}")

    return corr_matrix, per_kpi_corr


def apply_threshold(
    corr_matrix: np.ndarray, percentile: float
) -> np.ndarray:
    """应用自适应百分位数阈值，只保留高相关边。

    阈值 = 相关性矩阵上三角部分的第 percentile 百分位数。
    低于阈值的边权重置为零。

    Args:
        corr_matrix: 原始相关性矩阵。
        percentile: 百分位数（如 70 表示保留 top 30%）。

    Returns:
        二值化或加权的邻接矩阵。
    """
    # 获取上三角（不含对角线）的值来计算阈值
    upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    threshold = np.percentile(upper_tri, percentile)

    print(f"\n相关性阈值 (P{percentile}): {threshold:.4f}")
    print(f"  阈值以上边数: {(upper_tri > threshold).sum()} / {len(upper_tri)}")
    print(f"  保留比例: {(upper_tri > threshold).sum() / len(upper_tri) * 100:.1f}%")

    # 阈值化：保留高相关边，低相关置零
    adj_matrix = corr_matrix.copy()
    adj_matrix[corr_matrix < threshold] = 0.0

    # 对角线：便于后续 GCN 使用（加上自环时每个节点包含自己的信息）
    if SELF_LOOP:
        np.fill_diagonal(adj_matrix, 1.0)
    else:
        np.fill_diagonal(adj_matrix, 0.0)

    return adj_matrix


def add_prior_knowledge(
    adj_matrix: np.ndarray,
    cell_ids: np.ndarray,
    df: pd.DataFrame,
    cell_type_weight: float,
    slice_type_weight: float,
) -> np.ndarray:
    """融入领域先验知识，为同类型/同切片的基站之间的边增加权重偏置。

    先验 1：如果两个小区属于相同 cell_type（macro/micro/pico），
           边权增加 cell_type_weight。
    先验 2：如果两个小区的主要 slice_type 相同，
           边权增加 slice_type_weight。

    这些偏置反映了通信网络的层次化部署特征：
    - macro 站之间有骨干连接
    - 同切片类型的基站承载相似业务

    Args:
        adj_matrix: 当前的邻接矩阵。
        cell_ids: 小区 ID 数组。
        df: 原始数据，用于提取 cell_type 和主 slice_type。
        cell_type_weight: 同基站类型偏置权重。
        slice_type_weight: 同切片类型偏置权重。

    Returns:
        增强后的邻接矩阵。
    """
    # 获取每个小区的 cell_type 编码和主 slice_type 编码
    cell_info = df.groupby("cell_id").agg(
        cell_type_enc=("cell_type_enc", "first"),
        # 取该小区出现最多的 slice_type 编码
        slice_type_enc=("slice_type_enc", lambda x: x.mode().iloc[0]),
    )

    num_cells = len(cell_ids)
    id_to_idx = {cid: i for i, cid in enumerate(cell_ids)}

    ct_bias_added = 0
    st_bias_added = 0

    for i, cid_i in enumerate(cell_ids):
        ct_i = cell_info.loc[cid_i, "cell_type_enc"]
        st_i = cell_info.loc[cid_i, "slice_type_enc"]
        for j, cid_j in enumerate(cell_ids):
            if i >= j or adj_matrix[i, j] == 0:
                continue

            ct_j = cell_info.loc[cid_j, "cell_type_enc"]
            st_j = cell_info.loc[cid_j, "slice_type_enc"]

            if ct_i == ct_j:
                adj_matrix[i, j] += cell_type_weight
                adj_matrix[j, i] += cell_type_weight
                ct_bias_added += 1

            if st_i == st_j:
                adj_matrix[i, j] += slice_type_weight
                adj_matrix[j, i] += slice_type_weight
                st_bias_added += 1

    print(f"\n领域先验增强：")
    print(f"  同 cell_type 偏置应用: {ct_bias_added} 次 (~{cell_type_weight})")
    print(f"  同 slice_type 偏置应用: {st_bias_added} 次 (~{slice_type_weight})")

    # 对边权进行 min-max 归一化，保持在合理范围
    non_diag = adj_matrix[~np.eye(num_cells, dtype=bool)]
    if non_diag.max() > non_diag.min():
        max_val = non_diag.max()
        min_val = non_diag.min()
        # 只归一化非对角线元素
        mask = ~np.eye(num_cells, dtype=bool)
        adj_matrix[mask] = (adj_matrix[mask] - min_val) / (max_val - min_val)

    return adj_matrix


def save_artifacts(
    adj_matrix: np.ndarray,
    cell_ids: np.ndarray,
    output_dir: str,
) -> None:
    """保存邻接矩阵和小区 ID 列表到文件。

    Args:
        adj_matrix: 最终邻接矩阵。
        cell_ids: 小区 ID 数组。
        output_dir: 输出目录。
    """
    ensure_dir(output_dir)

    adj_path = os.path.join(output_dir, "adjacency_matrix.npy")
    cell_path = os.path.join(output_dir, "cell_ids.npy")

    np.save(adj_path, adj_matrix)
    np.save(cell_path, cell_ids)

    print(f"\n图结构已保存：")
    print(f"  邻接矩阵: {adj_path}  ({adj_matrix.shape})")
    print(f"  小区 ID:  {cell_path}  ({len(cell_ids)} 个节点)")
    print(f"  边数:      {(adj_matrix > 0).sum() - (np.diag(adj_matrix) > 0).sum()}")


def build_pyg_data(
    adj_matrix: np.ndarray,
    cell_ids: np.ndarray,
    df: pd.DataFrame,
) -> "tuple[object, dict]":
    """构建 PyTorch Geometric Data 对象，用于后续 GNN 模型训练。

    PyG Data 对象包含：
    - x: 节点特征矩阵 (num_nodes, num_features)
    - edge_index: COO 格式的边索引 (2, num_edges)
    - edge_weight: 边权重 (num_edges,)
    - y: 节点标签（此处暂不设置，训练时根据时间窗口动态构建）

    同时返回一个 node_feature_stats dict 记录每个节点的静态特征。

    Args:
        adj_matrix: 邻接矩阵。
        cell_ids: 小区 ID 数组。
        df: 原始数据。

    Returns:
        (pyg_data, metadata) 元组。
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError:
        print(
            "\n⚠ PyTorch Geometric 未安装，跳过 PyG Data 构建。"
            "\n  安装命令: pip install torch-geometric"
        )
        return None, {}

    # 构建 edge_index（COO 格式）
    num_nodes = len(cell_ids)
    edge_list = []
    edge_weights = []

    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i, j] > 0:
                edge_list.append([i, j])
                edge_weights.append(adj_matrix[i, j])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32)

    # 构建节点静态特征（每个小区取所有时间步的平均值）
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

    node_features = []
    for cid in cell_ids:
        cell_df = df[df["cell_id"] == cid]
        mean_feats = cell_df[feature_cols].mean().values
        node_features.append(mean_feats)

    x = torch.tensor(np.array(node_features), dtype=torch.float32)

    # 构建 Data 对象
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        num_nodes=num_nodes,
    )

    # 保存 PyG Data 对象
    pyg_path = os.path.join(OUTPUT_DIR, "graph_data.pt")
    torch.save(data, pyg_path)

    metadata = {
        "num_nodes": num_nodes,
        "num_edges": edge_index.shape[1],
        "num_node_features": x.shape[1],
        "cell_ids": cell_ids,
        "feature_names": feature_cols,
    }

    print(f"  PyG Data 对象: {pyg_path}")
    print(f"    节点数: {num_nodes}, 特征维数: {x.shape[1]}")
    print(f"    边数: {edge_index.shape[1]}")
    print(f"    平均度: {edge_index.shape[1] / num_nodes:.1f}")

    return data, metadata


# ============================================================================
# 可视化函数
# ============================================================================

def plot_correlation_heatmap(
    corr_matrix: np.ndarray, cell_ids: np.ndarray, output_dir: str
) -> None:
    """绘制相关性矩阵热力图。

    Args:
        corr_matrix: 相关性矩阵。
        cell_ids: 小区 ID 标签。
        output_dir: 图片输出目录。
    """
    ensure_dir(output_dir)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")

    # 设置刻度
    n = len(cell_ids)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(cell_ids, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cell_ids, fontsize=8)

    # 颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Average |Correlation|", fontsize=11)

    ax.set_title(
        "5G Cell KPI Correlation Matrix\n"
        f"({n} cells × {len(KPI_COLUMNS)} KPIs averaged)",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()
    path = os.path.join(output_dir, "correlation_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  相关性热力图: {path}")


def plot_graph_topology(
    adj_matrix: np.ndarray,
    cell_ids: np.ndarray,
    df: pd.DataFrame,
    output_dir: str,
) -> None:
    """绘制网络拓扑图，节点颜色按 cell_type 区分。

    Args:
        adj_matrix: 邻接矩阵。
        cell_ids: 小区 ID 数组。
        df: 原始数据，用于提取节点属性。
        output_dir: 图片输出目录。
    """
    ensure_dir(output_dir)

    # 构建 NetworkX 图
    G = nx.Graph()
    n = len(cell_ids)

    # 获取节点属性
    cell_type_map = (
        df.groupby("cell_id")["cell_type"].first().to_dict()
    )

    # 颜色映射
    type_colors = {
        "macro": "#E74C3C",   # 红
        "micro": "#3498DB",   # 蓝
        "pico":  "#2ECC71",   # 绿
    }

    for i, cid in enumerate(cell_ids):
        ct = cell_type_map.get(cid, "unknown")
        G.add_node(i, label=cid, cell_type=ct)

    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i, j] > 0:
                G.add_edge(i, j, weight=adj_matrix[i, j])

    # 布局
    pos = nx.spring_layout(G, seed=42, k=2, iterations=100)

    fig, ax = plt.subplots(figsize=(12, 10))

    # 按 cell_type 分组绘制节点
    for ct, color in type_colors.items():
        node_list = [
            i for i, cid in enumerate(cell_ids)
            if cell_type_map.get(cid) == ct
        ]
        if node_list:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=node_list,
                node_color=color,
                node_size=600,
                alpha=0.9,
                label=f"{ct} ({len(node_list)})",
                ax=ax,
            )

    # 绘制边（粗细按权重）
    edges = list(G.edges(data=True))
    if edges:
        weights = [d["weight"] * 2.5 for (_, _, d) in edges]
        nx.draw_networkx_edges(
            G, pos,
            width=weights,
            alpha=0.4,
            edge_color="#555555",
            ax=ax,
        )

    # 标签
    labels = {i: cid.replace("CELL_", "") for i, cid in enumerate(cell_ids)}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

    ax.set_title(
        "5G Base Station Network Topology\n"
        f"({n} nodes, {G.number_of_edges()} edges, "
        f"avg degree: {2*G.number_of_edges()/n:.1f})",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "network_topology.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  网络拓扑图: {path}")


def plot_degree_distribution(adj_matrix: np.ndarray, output_dir: str) -> None:
    """绘制节点度分布直方图。

    Args:
        adj_matrix: 邻接矩阵。
        output_dir: 图片输出目录。
    """
    ensure_dir(output_dir)

    # 计算每个节点的度（非对角线非零元素个数）
    degrees = (adj_matrix > 0).sum(axis=1) - (np.diag(adj_matrix) > 0).astype(int)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(0, degrees.max() + 2) - 0.5
    ax.hist(degrees, bins=bins, color="#5DADE2", edgecolor="#2C3E50", alpha=0.85)

    ax.set_xlabel("Degree", fontsize=11)
    ax.set_ylabel("Number of Nodes", fontsize=11)
    ax.set_title(
        f"Node Degree Distribution\n"
        f"(mean={degrees.mean():.1f}, min={degrees.min()}, max={degrees.max()})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(range(0, int(degrees.max()) + 1))

    plt.tight_layout()
    path = os.path.join(output_dir, "degree_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  度分布图: {path}")


# ============================================================================
# 主函数
# ============================================================================

def main() -> None:
    """主流程：加载数据 → 计算相关性 → 阈值筛选 → 先验增强 → 保存输出。"""
    print("=" * 65)
    print("  5G 基站网络图构建 (KPI 相关性方法)")
    print("=" * 65)

    # Step 1: 加载数据
    print(f"\n[Step 1] 加载数据: {PROCESSED_CSV}")
    df = load_and_sort_data(PROCESSED_CSV)
    print(f"  总记录数: {len(df)}, 小区数: {df['cell_id'].nunique()}")
    print(f"  时间范围: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

    # Step 2: 提取每个小区的 KPI 时间序列
    print(f"\n[Step 2] 提取 KPI 时间序列")
    print(f"  使用的 KPI: {KPI_COLUMNS}")
    cell_ts_dict, ts_matrix, cell_ids = build_cell_kpi_timeseries(df, KPI_COLUMNS)
    num_cells, timesteps, num_kpis = ts_matrix.shape
    print(f"  时间序列形状: ({num_cells} 个小区, {timesteps} 个时间步, {num_kpis} 个 KPI)")

    # Step 3: 计算多特征相关性矩阵
    print(f"\n[Step 3] 计算相关性矩阵")
    corr_matrix, per_kpi_corr = compute_correlation_matrix(ts_matrix, KPI_COLUMNS)
    print(f"  综合相关性范围: [{corr_matrix.min():.4f}, {corr_matrix.max():.4f}]")

    # Step 4: 自适应阈值筛选
    print(f"\n[Step 4] 自适应阈值筛选 (P{CORRELATION_PERCENTILE})")
    adj_matrix = apply_threshold(corr_matrix, CORRELATION_PERCENTILE)

    # Step 5: 融入领域先验知识
    print(f"\n[Step 5] 融入领域先验知识")
    adj_matrix = add_prior_knowledge(
        adj_matrix, cell_ids, df,
        CELL_TYPE_PRIOR_WEIGHT, SLICE_TYPE_PRIOR_WEIGHT,
    )

    # Step 6: 保存图结构
    print(f"\n[Step 6] 保存图结构")
    save_artifacts(adj_matrix, cell_ids, OUTPUT_DIR)

    # Step 7: 构建 PyG Data 对象（如果环境支持）
    print(f"\n[Step 7] 构建 PyTorch Geometric Data 对象")
    build_pyg_data(adj_matrix, cell_ids, df)

    # Step 8: 可视化
    print(f"\n[Step 8] 生成可视化")
    plot_correlation_heatmap(corr_matrix, cell_ids, FIGURES_DIR)
    plot_graph_topology(adj_matrix, cell_ids, df, FIGURES_DIR)
    plot_degree_distribution(adj_matrix, FIGURES_DIR)

    print("\n" + "=" * 65)
    print("  图构建完成！")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  可视化目录: {FIGURES_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
