"""
GCN+LSTM 时空预测模型

架构：
- 每个时间步使用 GCNConv (PyG) 提取小区之间的空间依赖
- 将 GCN 输出的时间序列输入 LSTM 捕获时序演化
- 全连接层输出下一时刻的拥塞预测

需要 PyTorch >= 2.0 + PyG >= 2.3，旧版 PyTorch 存在 CUDA scatter 内核 bug。

输入: 过去 T 个时间步的节点特征 (batch, seq_len, num_nodes, in_features)
输出: 未来时刻各节点的拥塞预测 (batch, num_nodes, output_dim)
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNLSTM(nn.Module):
    """GCN+LSTM 时空预测模型。

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

        # ---- 空间编码 (GCN) ----
        # add_self_loops=False：build_graph.py 已手动加自环到 edge_index 中
        self.gcn1 = GCNConv(in_features, gcn_hidden, add_self_loops=False)
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden, add_self_loops=False)
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

        对 batch 中每个样本逐时间步独立做 GCN（不拼接图，避免 PyG 内部
        add_remaining_self_loops 的 CUDA 问题），然后在 LSTM 层正常做 batch。
        """
        batch_size, num_nodes, seq_len, _ = x.shape

        all_samples = []
        for b in range(batch_size):
            x_b = x[b]  # (num_nodes, seq_len, F)
            gcn_per_step = []
            for t in range(seq_len):
                x_t = x_b[:, t, :]  # (num_nodes, F)
                h = self.gcn1(x_t, edge_index, edge_weight)
                h = torch.relu(h)
                h = self.gcn_dropout(h)
                h = self.gcn2(h, edge_index, edge_weight)
                h = torch.relu(h)
                gcn_per_step.append(h)
            gcn_b = torch.stack(gcn_per_step, dim=0)
            all_samples.append(gcn_b)

        gcn_seq = torch.stack(all_samples, dim=0)

        # (batch*N, seq_len, H)
        gcn_seq = gcn_seq.permute(0, 2, 1, 3).reshape(
            batch_size * num_nodes, seq_len, self.gcn_hidden
        )

        lstm_out, _ = self.lstm(gcn_seq)
        last_out = lstm_out[:, -1, :]

        out = self.fc(last_out)
        return out.view(batch_size, num_nodes, self.output_dim)
