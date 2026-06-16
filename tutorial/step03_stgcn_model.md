# Step 03: ST-GCN 模型实现

## 学习目标

- [ ] 用 PyTorch 实现完整的 ST-GCN 模型
- [ ] 理解 `nn.Module` 中 `forward()` 的设计模式
- [ ] 理解 ST-GCN Block（空间GCN + 时间TCN）的实现
- [ ] 理解 3 种图划分策略的邻接矩阵构建
- [ ] 能独立 debug 模型各层的 tensor shape

## 参考文档

- [ST-GCN 原论文](https://arxiv.org/abs/1801.07455) Section 3
- [官方 ST-GCN 实现 (PyTorch)](https://github.com/yysijie/st-gcn)
- [PyTorch nn.Module 文档](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)

---

## 3.1 ST-GCN 整体架构

```
输入视频: (T_frames, 17_joints, 3_coords)
                    ↓
            [BatchNorm]
                    ↓
    ST-GCN Block 1 (64 channels) ─┐
    ST-GCN Block 2 (64 channels) ─┤
    ST-GCN Block 3 (64 channels) ─┤  ← 9个 Block
    ST-GCN Block 4 (128 channels) ┤
    ST-GCN Block 5 (128 channels) ┤
    ST-GCN Block 6 (128 channels) ┤
    ST-GCN Block 7 (256 channels) ┤
    ST-GCN Block 8 (256 channels) ┤
    ST-GCN Block 9 (256 channels) ─┘
                    ↓
            [Global Average Pooling]
                    ↓
            [FC Layer → num_classes]
                    ↓
            预测: class_probabilities
```

每个 ST-GCN Block 内部：
```
输入 x: (N, C_in, T, V)   V=关节数, T=帧数, N=batch
    │
    ├── 空间 GCN (Spatial Graph Conv)
    │   公式: Σ_k A_k @ x @ W_k   （k=3个划分）
    │
    ├── BatchNorm + ReLU
    │
    ├── 时间 CNN (Temporal Conv)
    │   1D Conv with kernel_size=9, same padding
    │
    ├── BatchNorm + ReLU
    │
    └── 残差连接 (如果 C_in != C_out 则用 1×1 Conv 对齐)
           ↓
输出: (N, C_out, T, V)
```

---

## 3.2 邻接矩阵构建

创建文件 `models/graph.py`

```python
"""构建 ST-GCN 所需的邻接矩阵"""
import numpy as np

# COCO-17 骨骼边
COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

def get_edge_to_hierarchy():
    """
    Spatial Configuration Partitioning:
    将邻居分为:
    - 根节点自身 (center)
    - 向心组 (centripetal): 比根节点更靠近重心的邻居
    - 离心组 (centrifugal): 比根节点更远离重心的邻居

    简化版：根据与根节点的距离来划分
    """
    from collections import defaultdict
    
    N = 17
    # 构建邻接表
    adj = defaultdict(list)
    for (u, v) in COCO_EDGES:
        adj[u].append(v)
        adj[v].append(u)
    
    # BFS 计算每个节点到节点0 (nose) 的跳数作为层级
    # 层级小的 = 更靠近身体中心
    level = {0: 0}
    queue = [0]
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if v not in level:
                level[v] = level[u] + 1
                queue.append(v)
    
    # 构建3个划分的邻接矩阵
    I = np.eye(N)          # 自连接 (根节点)
    A_centripetal = np.zeros((N, N))
    A_centrifugal = np.zeros((N, N))
    
    for (u, v) in COCO_EDGES:
        # 哪个节点离重心更近？
        if level[u] <= level[v]:
            # u更靠近中心 → 对于u，v是离心方向
            A_centrifugal[u, v] = 1
            A_centrifugal[v, u] = 1
            # 对于v，u是向心方向
            A_centripetal[v, u] = 1
            A_centripetal[u, v] = 1
        else:
            A_centrifugal[u, v] = 1
            A_centrifugal[v, u] = 1
            A_centripetal[v, u] = 1
            A_centripetal[u, v] = 1
    
    return [I, A_centripetal, A_centrifugal]

def normalize_adj_batch(adj_list):
    """
    对每个邻接矩阵做对称归一化: D^{-1/2} * (A + I) * D^{-1/2}
    返回: numpy array of shape (3, V, V)
    """
    normalized = []
    for A in adj_list:
        A_tilde = A + np.eye(A.shape[0]) + 1e-6  # 加自连接 + 防除零
        D_invsqrt = np.diag(np.power(A_tilde.sum(axis=1), -0.5))
        A_norm = D_invsqrt @ A_tilde @ D_invsqrt
        normalized.append(A_norm)
    return np.stack(normalized, axis=0)  # (3, 17, 17)
```

---

## 3.3 ST-GCN 核心模块

创建文件 `models/stgcn.py`

```python
"""PyTorch 实现的 ST-GCN 模型"""
import torch
import torch.nn as nn
import numpy as np
from models.graph import get_edge_to_hierarchy, normalize_adj_batch


class SpatialGraphConv(nn.Module):
    """空间图卷积层"""
    def __init__(self, in_channels, out_channels, num_joints=17):
        super().__init__()
        # 对 3 个划分各学一个权重矩阵
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1
        )
        # 边重要性权重 (可学习的 M_k 矩阵)
        self.edge_importance = nn.ParameterList([
            nn.Parameter(torch.ones(num_joints, num_joints))
            for _ in range(3)
        ])
    
    def forward(self, x, A):
        """
        x: (N, C_in, T, V)  注意：V在最后一维
        A: (3, V, V)  3个划分的归一化邻接矩阵
        返回: (N, C_out, T, V)
        """
        N, C, T, V = x.shape
        
        # x: (N*T, C, V) 在空间维度独立处理每一帧
        x = x.permute(0, 2, 1, 3).contiguous().view(N * T, C, V)
        
        # GCN: H_out = Σ_k (A_k ⊙ M_k) @ H_in @ W_k
        out = 0
        for k in range(3):
            # A_k ⊙ M_k
            A_k = A[k].to(x.device)  # (V, V)
            M_k = self.edge_importance[k]  # (V, V)
            A_weighted = A_k * M_k  # element-wise 乘积
            
            # A_weighted @ H_in: (V, V) @ (N*T, C, V).permute → 聚合邻居
            support = x @ A_weighted.T  # (N*T, C, V) ，每个节点的特征是其邻居加权和
            
            # 1×1 Conv 等效于 H_in @ W (变换特征维度)
            out += self.conv(support.unsqueeze(-1)).squeeze(-1)  # 绕了点弯，等效
        
        # 恢复形状
        out = out.view(N, T, -1, V).permute(0, 2, 1, 3).contiguous()
        return out


class TemporalConv(nn.Module):
    """时间卷积层（沿帧维度做 1D Conv）"""
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2  # 'same' padding
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0),
        )
    
    def forward(self, x):
        """
        x: (N, C, T, V)
        返回: (N, C_out, T, V)
        """
        return self.conv(x)


class STGCNBlock(nn.Module):
    """ST-GCN 基础模块: 空间GCN + 时间TCN + 残差"""
    def __init__(self, in_channels, out_channels, stride=1, temporal_kernel=9):
        super().__init__()
        
        # 空间图卷积
        self.spatial_conv = SpatialGraphConv(in_channels, out_channels)
        self.spatial_bn = nn.BatchNorm2d(out_channels)
        
        # 时间卷积
        self.temporal_conv = TemporalConv(
            out_channels, out_channels,
            kernel_size=temporal_kernel, stride=stride
        )
        self.temporal_bn = nn.BatchNorm2d(out_channels)
        
        # 残差连接
        self.residual = None
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, A):
        """
        x: (N, C_in, T, V)
        A: (3, V, V)
        """
        identity = x
        
        # 空间 GCN
        out = self.spatial_conv(x, A)
        out = self.spatial_bn(out)
        out = self.relu(out)
        
        # 时间 TCN
        out = self.temporal_conv(out)
        out = self.temporal_bn(out)
        
        # 残差连接
        if self.residual is not None:
            identity = self.residual(identity)
        
        out = out + identity
        out = self.relu(out)
        
        return out


class STGCN(nn.Module):
    """
    完整的 ST-GCN 模型
    
    输入: (N, C_in=3, T, V=17)
    输出: (N, num_classes)
    """
    def __init__(self, num_classes=10, in_channels=3, num_joints=17,
                 base_channels=64,
                 layout=[
                     [64, 64, 64],       # 第一阶段: 3个block, 64通道
                     [128, 128, 128],     # 第二阶段: 3个block, 128通道
                     [256, 256, 256],     # 第三阶段: 3个block, 256通道
                 ],
                 temporal_kernel=9,
                 dropout=0.5):
        super().__init__()
        
        self.num_joints = num_joints
        
        # 输入预处理
        self.data_bn = nn.BatchNorm2d(in_channels)
        
        # 构建邻接矩阵 (固定不变，不参与梯度)
        adj_list = get_edge_to_hierarchy()
        A = normalize_adj_batch(adj_list)  # (3, 17, 17)
        self.register_buffer('A', torch.tensor(A, dtype=torch.float32))
        
        # ST-GCN Blocks
        self.blocks = nn.ModuleList()
        in_ch = in_channels
        
        for stage_idx, stage_channels in enumerate(layout):
            for block_idx, out_ch in enumerate(stage_channels):
                stride = 2 if block_idx == 0 and stage_idx > 0 else 1
                self.blocks.append(
                    STGCNBlock(in_ch, out_ch, stride=stride,
                               temporal_kernel=temporal_kernel)
                )
                in_ch = out_ch
        
        # 全局池化 + 分类头
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_ch, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x: (N, 3, T, V) 或 (N, T, V, 3)
           其中 N=batch, T=frames, V=17_joints, 3=xyz
        
        返回: (N, num_classes)
        """
        # 如果输入是 (N, T, V, 3)，转换为 (N, 3, T, V)
        if x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2).contiguous()
        
        N, C, T, V = x.shape
        assert V == self.num_joints, f"Expected {self.num_joints} joints, got {V}"
        
        # BatchNorm 输入
        x = self.data_bn(x)
        
        # 经过所有 ST-GCN Blocks
        for block in self.blocks:
            x = block(x, self.A)
        
        # (N, 256, T, V) → Global Pool → (N, 256, 1, 1)
        x = self.global_pool(x)  # (N, 256, 1, 1)
        x = x.view(N, -1)        # (N, 256)
        
        # 分类头
        x = self.dropout(x)
        x = self.fc(x)  # (N, num_classes)
        
        return x
    
    def get_feature(self, x):
        """提取特征（用于可视化或迁移学习）"""
        if x.dim() == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2).contiguous()
        x = self.data_bn(x)
        for block in self.blocks:
            x = block(x, self.A)
        x = self.global_pool(x)
        return x.view(x.size(0), -1)


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    # 创建一个随机输入: batch=2, 300帧, 17关节, 3坐标
    x = torch.randn(2, 300, 17, 3)
    
    model = STGCN(num_classes=10)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    y = model(x)
    print(f"输入 shape: {x.shape}")
    print(f"输出 shape: {y.shape}")  # (2, 10)
    
    # 检查梯度流
    loss = y.sum()
    loss.backward()
    print(f"梯度正常: loss={loss.item():.4f}")
```

---

## 3.4 理解关键 Tensor Shape 变化

在 ST-GCN 的 forward 过程中，tensor shape 变换如下：

```
输入: (B, T, V, 3)  ← 我们通常这么存
     ↓ permute
     (B, 3, T, V)  ← PyTorch Conv2d 的输入格式

DataBN:
     (B, 3, T, V) → (B, 3, T, V)

Stage 1 (3 blocks, 64 ch):
     (B, 3,  T,   V) → ... → (B, 64,  T,   V)

Stage 2 (3 blocks, 128 ch):
     (B, 64, T,   V) → (B, 128, T/2, V)  ← stride=2 在第一层
     (B, 128, T/2, V) → (B, 128, T/2, V)

Stage 3 (3 blocks, 256 ch):
     (B, 128, T/2, V) → (B, 256, T/4, V) ← stride=2
     (B, 256, T/4, V) → (B, 256, T/4, V)

Global Average Pooling:
     (B, 256, T/4, V) → (B, 256, 1, 1)

Flatten + FC:
     (B, 256) → (B, 10)
```

---

## 检查要点

- [ ] 能画出 ST-GCN Block 的内部数据流
- [ ] 理解 `SpatialGraphConv` 中 3 个划分如何并行处理
- [ ] 理解 `edge_importance` (M_k) 矩阵的作用（学习边权重）
- [ ] 理解时间卷积的 `kernel_size=(9, 1)` 为什么第二维是 1
- [ ] 能解释残差连接的作用和实现

## 常见问题

**Q: 为什么时间卷积的 kernel 是 `(9, 1)`？**
A: `(9, 1)` 表示沿时间轴 kernel=9，沿关节轴 kernel=1（不做跨关节融合，因为空间融合已由 GCN 完成）。

**Q: `edge_importance` 练的是什么？**
A: 学习每条边的重要性权重。模型会自动学到哪些骨骼连接对动作识别更重要。

**Q: 我的数据帧数不固定怎么办？**
A: 需要做 padding 或 interpolation 使所有样本帧数一致（Step 04 会处理）。

---

## 下一步

完成本步后，进入 [Step 04: 数据预处理流水线](step04_data_pipeline.md) 构建数据加载管道。
