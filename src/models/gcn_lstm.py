"""
GCN+LSTM 时空预测模型

架构：
- 每个时间步使用 GCNConv 提取小区之间的空间依赖
- 将 GCN 输出的时间序列输入 LSTM 捕获时序演化
- 全连接层输出下一时刻的拥塞预测

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
        output_dim:        输出维度（默认 1，如预测 PRB 利用率）
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
        # add_self_loops=False：因为 build_graph.py 中已经手动加了自环，
        # 若让 PyG 再加一次，在 mini-batch 手动拼接场景下会触发 CUDA device-side assert
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
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:            节点特征序列 (batch_size, seq_len, num_nodes, in_features)
            edge_index:   图边索引 (2, num_edges) — 原始图，不拼接
            edge_weight:  边权重 (num_edges,)，可选

        Returns:
            y_pred:  预测值 (batch_size, num_nodes, output_dim)

        策略：不在 GCN 层做 mini-batch 拼接（避免 PyG 内部 CUDA scatter 越界），
        而是对 batch 中每个样本独立跑 GCN，再在 LSTM 层正常做 batch。
        """
        batch_size, seq_len, num_nodes, _ = x.shape

        # ---- Step 1: 对 batch 中每个样本，逐时间步独立跑 GCN ----
        all_samples = []  # 收集每个样本的结果

        for b in range(batch_size):
            x_b = x[b]  # (seq_len, num_nodes, in_features)

            gcn_per_step = []
            for t in range(seq_len):
                x_t = x_b[t]  # (num_nodes, in_features)

                h = self.gcn1(x_t, edge_index, edge_weight)
                h = torch.relu(h)
                h = self.gcn_dropout(h)
                h = self.gcn2(h, edge_index, edge_weight)
                h = torch.relu(h)

                gcn_per_step.append(h)  # (num_nodes, gcn_hidden)

            # (seq_len, num_nodes, gcn_hidden)
            gcn_b = torch.stack(gcn_per_step, dim=0)
            all_samples.append(gcn_b)

        # 堆叠所有样本: (batch_size, seq_len, num_nodes, gcn_hidden)
        gcn_seq = torch.stack(all_samples, dim=0)

        # ---- Step 2: LSTM 时序建模 ----
        # 将 (batch, num_nodes) 合并: (batch_size * num_nodes, seq_len, gcn_hidden)
        gcn_seq = gcn_seq.permute(0, 2, 1, 3).reshape(
            batch_size * num_nodes, seq_len, self.gcn_hidden
        )

        lstm_out, _ = self.lstm(gcn_seq)
        last_out = lstm_out[:, -1, :]  # (batch_size * num_nodes, lstm_hidden)

        # ---- Step 3: 输出预测 ----
        out = self.fc(last_out)  # (batch_size * num_nodes, output_dim)
        out = out.view(batch_size, num_nodes, self.output_dim)

        return out



class GCNLSTMSimple(nn.Module):
    """简化版 GCN+LSTM：单节点预测，不使用 mini-batch 图 batching。

    适用于单图场景：每个批次是一个节点的时间窗口。
    该方法用 GCN 聚合邻居特征，再对单节点序列做 LSTM。

    Args:
        in_features:       每个节点的输入特征维度
        gcn_hidden:        GCN 隐层维度
        lstm_hidden:       LSTM 隐层维度
        lstm_layers:       LSTM 层数
        output_dim:        输出维度
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

        # ---- 空间编码 ----
        self.gcn1 = GCNConv(in_features, gcn_hidden, add_self_loops=False)
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden, add_self_loops=False)
        self.gcn_dropout = nn.Dropout(dropout)

        # ---- 时序编码 ----
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
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:            节点特征序列 (batch_size, seq_len, in_features)
                          或 (num_nodes, seq_len, in_features)
            edge_index:   图边索引 (2, num_edges)
            edge_weight:  边权重 (num_edges,)，可选

        Returns:
            y_pred:  预测值 (batch_size, output_dim) 或 (num_nodes, output_dim)
        """
        batch_size, seq_len, _ = x.shape

        # ---- 逐时间步做 GCN 空间编码 ----
        gcn_outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # (num_nodes, in_features)
            h = self.gcn1(x_t, edge_index, edge_weight)
            h = torch.relu(h)
            h = self.gcn_dropout(h)
            h = self.gcn2(h, edge_index, edge_weight)
            h = torch.relu(h)
            gcn_outputs.append(h)

        # 堆叠: (batch_size, seq_len, gcn_hidden)
        gcn_seq = torch.stack(gcn_outputs, dim=1)

        # ---- LSTM 时序建模 ----
        lstm_out, _ = self.lstm(gcn_seq)
        last_out = lstm_out[:, -1, :]  # (batch_size, lstm_hidden)

        # ---- 输出 ----
        out = self.fc(last_out)  # (batch_size, output_dim)
        return out
