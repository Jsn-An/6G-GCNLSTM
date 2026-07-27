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
        self.gcn1 = GCNConv(in_features, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden)
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
            edge_index:   图边索引 (2, num_edges)
            edge_weight:  边权重 (num_edges,)，可选

        Returns:
            y_pred:  预测值 (batch_size, num_nodes, output_dim)
        """
        batch_size, seq_len, num_nodes, _ = x.shape

        # ---- Step 1: 逐时间步做 GCN 空间编码 ----
        gcn_outputs = []
        for t in range(seq_len):
            # 取出第 t 步所有 batch 的特征： (batch_size * num_nodes, in_features)
            x_t = x[:, t, :, :].reshape(batch_size * num_nodes, -1)

            # GCN 消息传递需要知道 batch 内的节点属于哪个图
            # 构造 batch 向量，每个 batch 样本有自己的图（边相同，节点特征不同）
            batch_vec = torch.arange(batch_size, device=x.device).repeat_interleave(num_nodes)
            # 对边索引进行偏移以适配 batching
            edge_index_batch = self._batch_edge_index(edge_index, num_nodes, batch_size, x.device)

            h = self.gcn1(x_t, edge_index_batch, edge_weight)
            h = torch.relu(h)
            h = self.gcn_dropout(h)
            h = self.gcn2(h, edge_index_batch, edge_weight)
            h = torch.relu(h)

            # 恢复形状: (batch_size, num_nodes, gcn_hidden)
            h = h.view(batch_size, num_nodes, self.gcn_hidden)
            gcn_outputs.append(h)

        # 堆叠: (batch_size, seq_len, num_nodes, gcn_hidden)
        gcn_seq = torch.stack(gcn_outputs, dim=1)

        # ---- Step 2: LSTM 时序建模 ----
        # 将 num_nodes 合并到 batch 维度:
        # (batch_size * num_nodes, seq_len, gcn_hidden)
        gcn_seq = gcn_seq.permute(0, 2, 1, 3).reshape(
            batch_size * num_nodes, seq_len, self.gcn_hidden
        )

        lstm_out, (h_n, c_n) = self.lstm(gcn_seq)
        # 取最后一个时间步的输出: (batch_size * num_nodes, lstm_hidden)
        last_out = lstm_out[:, -1, :]

        # ---- Step 3: 输出预测 ----
        out = self.fc(last_out)  # (batch_size * num_nodes, output_dim)
        out = out.view(batch_size, num_nodes, self.output_dim)

        return out

    @staticmethod
    def _batch_edge_index(
        edge_index: torch.Tensor,
        num_nodes: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """将单个图的 edge_index 复制并偏移，用于 mini-batch 训练。

        PyG 的 batching 机制：每个样本的节点索引需要偏移 i * num_nodes。
        """
        edge_list = []
        for i in range(batch_size):
            offset = i * num_nodes
            edge_list.append(edge_index + offset)
        return torch.cat(edge_list, dim=1).to(device)


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
        self.gcn1 = GCNConv(in_features, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden)
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
