"""
GCN+LSTM 时空预测模型

架构：
- 每个时间步使用 GCN 提取小区之间的空间依赖
- 将 GCN 输出的时间序列输入 LSTM 捕获时序演化
- 全连接层输出下一时刻的拥塞预测

GCN 层使用纯 PyTorch 算子实现（torch.sparse + scatter_add_），
不依赖 PyG 的 GCNConv，彻底避开其内部 CUDA scatter 兼容性问题。

输入: 过去 T 个时间步的节点特征 (batch, seq_len, num_nodes, in_features)
输出: 未来时刻各节点的拥塞预测 (batch, num_nodes, output_dim)
"""

import torch
import torch.nn as nn


# ============================================================================
# 自定义 GCN 层（纯 PyTorch 实现，完全绕过 PyG）
# ============================================================================

class GCNLayer(nn.Module):
    """单层图卷积，使用纯 PyTorch 稀疏矩阵运算实现。

    公式：H' = σ(D^{-1/2} Â D^{-1/2} H W)

    其中 Â = A + I（已由 build_graph.py 在 edge_index 中预加自环）。
    本层不再额外添加自环。

    Args:
        in_dim:  输入特征维度
        out_dim: 输出特征维度
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,          # (N, in_dim)
        edge_index: torch.Tensor, # (2, E)
        edge_weight: torch.Tensor | None = None,  # (E,) or (E, 1)
    ) -> torch.Tensor:
        num_nodes = x.size(0)
        device = x.device

        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=device, dtype=x.dtype)
        else:
            # 防御：确保 edge_weight 是 1D，兼容 PyG Data 存储为 (E, 1) 的情况
            edge_weight = edge_weight.view(-1)

        # 计算节点度 D[i] = Σ_j A[i,j]
        # 使用 scatter_add_ 而非 PyG 的 degree()，避免 PyG 内部 CUDA 问题
        row = edge_index[0]
        deg = torch.zeros(num_nodes, device=device, dtype=x.dtype).scatter_add_(0, row, edge_weight)

        # D^{-1/2}，处理孤立节点（度为 0 时设置 0）
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

        # 对称归一化边权重：w_norm[i,j] = d_i^{-1/2} · w[i,j] · d_j^{-1/2}
        col = edge_index[1]
        norm_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        # 构建稀疏归一化邻接矩阵 Â_norm
        A_norm = torch.sparse_coo_tensor(
            edge_index, norm_weight, (num_nodes, num_nodes)
        ).coalesce()

        # 稀疏矩阵乘法：Â_norm @ (X @ W)
        out = torch.sparse.mm(A_norm, self.W(x))
        return out


# ============================================================================
# GCN + LSTM 模型
# ============================================================================

class GCNLSTM(nn.Module):
    """GCN+LSTM 时空预测模型。

    使用自定义 GCNLayer（纯 PyTorch 稀疏运算）替代 PyG 的 GCNConv，
    彻底绕过 PyG 内部的 CUDA scatter 兼容性问题。

    Args:
        in_features:       每个节点的输入特征维度
        gcn_hidden:        GCN 隐层维度
        lstm_hidden:       LSTM 隐层维度
        lstm_layers:       LSTM 层数
        output_dim:        输出维度（默认 1）
        dropout:           Dropout 比例
    """

    def __init__(
        self,
        in_features: int,
        gcn_hidden: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.in_features = in_features
        self.gcn_hidden = gcn_hidden
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.output_dim = output_dim

        # ---- 空间编码 (GCN) ----
        self.gcn1 = GCNLayer(in_features, gcn_hidden)
        self.gcn2 = GCNLayer(gcn_hidden, gcn_hidden)
        self.gcn_dropout = nn.Dropout(dropout)

        # ---- 时序编码 (LSTM) ----
        self.lstm = nn.LSTM(
            input_size=gcn_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # ---- 输出层 ----
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,               # (batch, seq_len, num_nodes, F)
        edge_index: torch.Tensor,       # (2, E)
        edge_weight: torch.Tensor | None = None,  # (E,)
    ) -> torch.Tensor:
        """前向传播。

        对 batch 中每个样本逐时间步独立做 GCN（不拼接图），
        然后在 LSTM 层正常做 batch。
        """
        batch_size, seq_len, num_nodes, _ = x.shape

        all_samples = []
        for b in range(batch_size):
            x_b = x[b]  # (seq_len, num_nodes, F)
            gcn_per_step = []
            for t in range(seq_len):
                x_t = x_b[t]  # (num_nodes, F)
                h = self.gcn1(x_t, edge_index, edge_weight)
                h = torch.relu(h)
                h = self.gcn_dropout(h)
                h = self.gcn2(h, edge_index, edge_weight)
                h = torch.relu(h)
                gcn_per_step.append(h)
            gcn_b = torch.stack(gcn_per_step, dim=0)  # (seq_len, num_nodes, H)
            all_samples.append(gcn_b)

        gcn_seq = torch.stack(all_samples, dim=0)  # (batch, seq_len, N, H)

        # 将 (batch, N) 合并: (batch*N, seq_len, H)
        gcn_seq = gcn_seq.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, seq_len, self.gcn_hidden)

        lstm_out, _ = self.lstm(gcn_seq)
        last_out = lstm_out[:, -1, :]  # (batch*N, lstm_hidden)

        out = self.fc(last_out)  # (batch*N, output_dim)
        return out.view(batch_size, num_nodes, self.output_dim)
