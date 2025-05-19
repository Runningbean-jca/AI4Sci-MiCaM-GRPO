from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool

# -------------------------------
# 多层感知机 MLP 实现（用于嵌入/预测）
# -------------------------------
class MLP(nn.Module):
    def __init__(
        self,
        in_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: Optional[int] = None,
        dropout: float = 0.,
        act: nn = nn.ReLU(inplace=True),
        batch_norm: bool = True,
        bias: bool = True,
    ):
        super().__init__()

        # 构造每层的神经元个数列表
        if in_channels is not None:
            assert num_layers >= 1
            channel_list = [hidden_channels] * (num_layers - 1)
            channel_list = [in_channels] + channel_list + [out_channels]

        assert isinstance(channel_list, (tuple, list))
        assert len(channel_list) >= 2
        self.channel_list = channel_list

        self.dropout = dropout
        self.act = act

        # 构建线性层列表
        self.lins = torch.nn.ModuleList()
        pairwise = zip(channel_list[:-1], channel_list[1:])
        for in_channels, out_channels in pairwise:
            self.lins.append(nn.Linear(in_channels, out_channels, bias=bias))

        # 构建归一化层（除了最后一层）
        self.norms = torch.nn.ModuleList()
        for hidden_channels in channel_list[1:-1]:
            if batch_norm:
                norm = nn.BatchNorm1d(hidden_channels)
            else:
                norm = nn.Identity()
            self.norms.append(norm)

        self.reset_parameters()

    def reset_parameters(self):
    # 初始化所有层参数
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
    # 依次通过 MLP，每层：线性 -> Norm -> 激活 -> dropout
        x = self.lins[0](x)
        for lin, norm in zip(self.lins[1:], self.norms):
            x = norm(x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin.forward(x)
        return x

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self.channel_list)[1:-1]})'

# ---------------------
# GIN + 虚节点（Virtual Node）图神经网络模块
# ---------------------
class GIN_virtual(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels, edge_dim, depth, dropout, virtual: bool=True, pooling: str="mean", residual=True):
        super(GIN_virtual, self).__init__()
        # 模型结构参数
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_channels
        self.edge_dim = edge_dim
        self.depth = depth
        self.dropout = dropout
        self.residual = residual
        self.virtual = virtual
        self.pooling = pooling

        # 初始输入映射（先通过一层 MLP）
        self.in_layer = MLP(
            in_channels = in_channels,
            hidden_channels = hidden_channels,
            out_channels = hidden_channels,
            num_layers = 3,
            act = nn.ReLU(inplace=True),
            dropout = dropout,
        )

        # 虚节点嵌入（所有图共享一个初始虚节点）
        self.vitual_embed = nn.Embedding(1, hidden_channels)
        nn.init.constant_(self.vitual_embed.weight.data, 0)

        # 构建多个 GNN 层和虚节点层
        for i in range(depth):
        # 构建 GINEConv 层
            layer_nn = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, hidden_channels),
                nn.Dropout(dropout),
            )
            layer = GINEConv(nn=layer_nn, train_eps=True, edge_dim=edge_dim)

            self.add_module(f"GNN_layer_{i}", layer)

            # 虚节点 MLP 层（图级聚合 + 更新虚节点）
            virtual_layer = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, hidden_channels),
                nn.Dropout(dropout),
            )
            self.add_module(f"virtual_layer_{i}", virtual_layer)
        
        # 输出层：融合所有 GNN 层的中间表示 + 输入层输出
        self.out_layer = MLP(
            in_channels = in_channels + hidden_channels * (depth + 1),
            hidden_channels = hidden_channels,
            out_channels = out_channels,
            num_layers = 3,
            act = nn.ReLU(inplace=True),
            dropout = dropout,
        )

    def forward(self, x, edge_index, edge_attr, batch):
        if self.virtual:
            v = torch.zeros(batch[-1].item() + 1, dtype=torch.long).to(x.device) # 每个图一个虚节点（batch size）
            virtual_embed = self.vitual_embed(v)  # 虚节点嵌入

        x_list = [x, self.in_layer(x)]  # 输入原始特征和初始 MLP 映射

        for i in range(self.depth):
            GNN_layer = getattr(self, f"GNN_layer_{i}")
            if self.virtual:
                virtual_layer = getattr(self, f"virtual_layer_{i}")
                x_list[-1] = x_list[-1] + virtual_embed[batch]  # 将虚节点特征加入所有节点

            x = GNN_layer(x_list[-1], edge_index, edge_attr) # 图卷积

            if self.residual:
                x = x + x_list[-1]  # 残差连接
            
            x_list.append(x)

            if self.virtual:
            # 用图级池化 + 虚节点更新 MLP
                virtual_tmp = virtual_layer(global_add_pool(x_list[-1], batch) + virtual_embed)
                virtual_embed = virtual_embed + virtual_tmp if self.residual else virtual_tmp

        join_vecs = torch.cat(x_list, -1) # 拼接所有层的表示
        nodes_reps = self.out_layer(join_vecs) # 每个节点的最终嵌入

        # 图级表示（池化）
        if self.pooling == "mean":
            graph_reps = global_mean_pool(nodes_reps, batch)
        else:
            assert self.pooling == "add"
            graph_reps = global_add_pool(nodes_reps, batch)

        del x, x_list, join_vecs
        if self.virtual:
            del virtual_embed, virtual_tmp
        return nodes_reps, graph_reps  # 返回节点和图表示
