# Step 02: 图卷积理论 — 从零理解 GCN

## 学习目标

- [ ] 理解图的数学表示：邻接矩阵、度矩阵、拉普拉斯矩阵
- [ ] 理解图卷积的核心公式：从空域和谱域两个角度
- [ ] 手写一个 NumPy 版 GCN Layer（前向传播）
- [ ] 理解 ST-GCN 中"空间图卷积"的含义

## 参考文档

- [GCN 原论文 (Kipf & Welling, 2017)](https://arxiv.org/abs/1609.02907)
- [ST-GCN 原论文 (Yan et al., 2018)](https://arxiv.org/abs/1801.07455)
- [PyTorch Geometric GCN 实现](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html)

---

## 2.1 图的基本概念

### 什么是图？

图 (Graph) = 节点 (Nodes/Vertices) + 边 (Edges)

在骨骼动作识别中：
- **节点** = 17 个人体关节
- **边** = 骨骼连接（如：左肩→左肘，左肘→左腕）

### 图的数学表示：邻接矩阵 (Adjacency Matrix)

对于 17 个节点的骨骼，邻接矩阵 **A** 是 17×17 的矩阵：

```
A[i][j] = 1  如果节点 i 和节点 j 之间有一条边
A[i][j] = 0  否则（没有边）
```

COCO-17 骨骼的邻接矩阵可视化：

```
        0 nose
       / \
      1   2   eyes
      |   |
      3   4   ears
       \ /
       5-6    shoulders
       ...  (共17个节点，约16条边)
```

### 自连接 (Self-loop)

图卷积中，每个节点也需要"看到自己"。所以在邻接矩阵上加单位矩阵 **I**：

```
Ã = A + I    (加了自连接的邻接矩阵)
```

### 度矩阵 (Degree Matrix)

度矩阵 **D** 是对角矩阵，`D[i][i]` = 节点 i 的度数（连接边数，包括自连接）。

```python
D_tilde[i][i] = sum_j(Ã[i][j])   # 第i行所有元素之和
```

### 归一化

为了防止数值不稳定，需要对邻接矩阵做对称归一化：

```
Â = D^(-1/2) Ã D^(-1/2)
```

这就是著名的 **"renormalization trick"**。

---

## 2.2 GCN 前向传播公式

### 核心公式

```
H^(l+1) = σ( D^(-1/2) Ã D^(-1/2) * H^(l) * W^(l) )
```

其中：
- **H^(l)**: 第 l 层的节点特征矩阵，shape = `(N_nodes, d_in)`
- **H^(l+1)**: 第 l+1 层的节点特征矩阵，shape = `(N_nodes, d_out)`
- **W^(l)**: 可学习的权重矩阵，shape = `(d_in, d_out)`
- **σ**: 非线性激活函数（ReLU 等）
- **Â**: 归一化邻接矩阵，shape = `(N_nodes, N_nodes)`

### 直观理解

每一层 GCN 做的事情：
1. **聚合 (Aggregate)**: 每个节点从它的邻居节点收集特征
2. **变换 (Transform)**: 用权重矩阵 W 对聚合后的特征做线性变换
3. **激活 (Activate)**: 通过非线性函数

重复多层后，每个节点就融合了它 k-hop 邻居的信息（k = 层数）。

---

## 2.3 动手实践：手写 NumPy GCN Layer

创建文件 `scripts/02_gcn_from_scratch.py`

```python
"""Step 02: 用 NumPy 从零实现单个 GCN Layer"""
import numpy as np

# ============================================================
# 1. 构建 COCO-17 骨骼图的邻接矩阵
# ============================================================
N_JOINTS = 17

COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

def build_adjacency_matrix(n_nodes, edges, undirected=True):
    """从边列表构建邻接矩阵"""
    A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for (i, j) in edges:
        A[i, j] = 1.0
        if undirected:
            A[j, i] = 1.0  # COCO骨骼是无向图
    return A

A = build_adjacency_matrix(N_JOINTS, COCO_EDGES)
print("=== 邻接矩阵 A (17×17) ===")
print(f"  Shape: {A.shape}")
print(f"  非零元素数: {A.sum():.0f}")
print(f"  边数 (不含自环): {A.sum()/2:.0f}")

# ============================================================
# 2. 添加自连接 + 对称归一化
# ============================================================
def normalize_adjacency(A):
    """对称归一化: A_norm = D^(-1/2) * (A + I) * D^(-1/2)"""
    A_tilde = A + np.eye(A.shape[0])  # 加自连接
    
    # 度矩阵 D_tilde
    D_tilde_diag = A_tilde.sum(axis=1)  # (N,), 每行求和
    
    # D^(-1/2): 对角阵的 -1/2 次幂 = 每个元素取 -1/2 次幂
    D_invsqrt = np.diag(np.power(D_tilde_diag, -0.5))
    
    # 对称归一化: D^(-1/2) * A_tilde * D^(-1/2)
    A_norm = D_invsqrt @ A_tilde @ D_invsqrt
    return A_norm

A_norm = normalize_adjacency(A)
print("\n=== 归一化邻接矩阵 ===")
print(f"  每行和（应接近1）: {A_norm.sum(axis=1)[:5]}")

# ============================================================
# 3. GCN Layer 前向传播
# ============================================================
class GCNLayerNumPy:
    """
    NumPy 实现的单层 GCN
    
    前向传播: H_out = ReLU( A_norm @ H_in @ W )
    """
    def __init__(self, d_in, d_out, A_norm):
        self.A_norm = A_norm  # (N, N) 归一化邻接矩阵（固定不变）
        
        # 权重初始化（Kaiming 初始化）
        self.W = np.random.randn(d_in, d_out) * np.sqrt(2.0 / d_in)
        
        # 偏置
        self.b = np.zeros(d_out)
    
    def forward(self, H):
        """
        H: (N_joints, d_in)  节点特征矩阵
        返回: (N_joints, d_out)
        """
        # 1. 聚合邻居特征: A_norm @ H
        aggregated = self.A_norm @ H  # (N, d_in)
        
        # 2. 线性变换: X @ W
        transformed = aggregated @ self.W  # (N, d_out)
        
        # 3. 加偏置 + 激活
        output = transformed + self.b  # (N, d_out)
        output = np.maximum(0, output)  # ReLU
        
        return output

# ============================================================
# 4. 测试 GCN Layer
# ============================================================
print("\n=== 测试 GCN Layer ===")

# 模拟一帧 3D 骨骼数据: (17 joints, 3 coords)
np.random.seed(42)
H_input = np.random.randn(N_JOINTS, 3)  # 17个节点，每个节点3维特征(xyz)

print(f"  输入 shape: {H_input.shape}")
print(f"  输入 (前3个关节):\n{H_input[:3]}")

# 创建 GCN 层: d_in=3 (xyz坐标), d_out=64 (隐藏特征)
layer = GCNLayerNumPy(d_in=3, d_out=64, A_norm=A_norm)
H_output = layer.forward(H_input)

print(f"\n  输出 shape: {H_output.shape}")  # (17, 64)
print(f"  输出 (前3个关节，前6维):\n{H_output[:3, :6]}")

# ============================================================
# 5. 验证：检查邻居聚合效果
# ============================================================
print("\n=== 验证邻居聚合 ===")

# 节点5 (left_shoulder) 的邻居是: 7(left_elbow), 6(right_shoulder), 11(left_hip)
# 以及自连接(节点5自己)
node = 5
neighbors = [5, 7, 6, 11]  # 包括自连接
print(f"  节点 {node} (left_shoulder) 的邻居: {neighbors}")

# 检查归一化邻接矩阵中，节点5对应的行
row5 = A_norm[node]
for n in range(N_JOINTS):
    if row5[n] > 0.001:
        w = row5[n]
        print(f"   A_norm[{node}][{n}] = {w:.4f} (权重 = 1/sqrt(deg[{node}]*deg[{n}]))")

# 验证：所有邻居权重之和应接近 1
print(f"   该行非零权重之和: {row5.sum():.4f}")
```

运行：
```bash
python scripts/02_gcn_from_scratch.py
```

### 预期输出

```
=== 邻接矩阵 A (17×17) ===
  Shape: (17, 17)
  非零元素数: 32.0
  边数 (不含自环): 16.0

=== 归一化邻接矩阵 ===
  每行和（应接近1）: [0.9999 0.9999 1.0000 1.0000 1.0000]

=== 测试 GCN Layer ===
  输入 shape: (17, 3)
  输出 shape: (17, 64)
```

---

## 2.4 从 GCN 到 ST-GCN

### 空间图卷积 (Spatial Graph Conv)

就是上面讲的 GCN：在同一帧内，聚合每个关节的邻居关节信息。

> 公式: `f_out = A_norm @ f_in @ W`（在 N 个骨骼节点上做图卷积）

### 时间图卷积 (Temporal Graph Conv)

沿时间轴做卷积（类似 CNN 的 1D Conv）：

> 公式: 对于每个关节，沿 T 帧做卷积，`kernel_size` 如 9 或 3

### ST-GCN Block = 空间卷积 + 时间卷积

```
输入: (N_batch, C_in, T_frames, N_joints)
          ↓
    空间 GCN: (N_joints × N_joints 邻接矩阵)
    在每个帧上独立运行
          ↓
    时间 CNN: 1D Conv with kernel_size=k_temporal
    在每个关节上独立运行
          ↓
    BatchNorm + ReLU + Dropout
          ↓
输出: (N_batch, C_out, T_frames, N_joints)
```

### ST-GCN 的图划分策略（Partition Strategies）

原论文提出了 3 种划分策略来增强邻接矩阵的表达能力：

1. **Uni-labeling**: 所有邻居同等对待，只有 1 个邻接矩阵
2. **Distance partitioning**: 按距离分：根节点、近邻居(距离=1)、远邻居(距离≥2)，共 3 个邻接矩阵
3. **Spatial configuration partitioning**: 按空间位置分：根节点、向心组、离心组，共 3 个邻接矩阵 ← 论文推荐！

每个划分对应一个独立的邻接矩阵和可学习的权重，最终求和：

```
output = Σ_k (A_k @ H_in @ W_k)
```

其中 k = 1, 2, 3（3 个划分），`A_k` 是第 k 个邻接矩阵，`W_k` 是可学习的边权重。

---

## 检查要点

- [ ] 能手动画出 COCO-17 骨骼图的邻接矩阵（至少理解前 5 个关节）
- [ ] 理解对称归一化公式 `D^(-1/2) Ã D^(-1/2)` 的作用
- [ ] 能解释 GCN 前向传播的三步：聚合 → 变换 → 激活
- [ ] 理解 ST-GCN 中"空间图卷积"和"时间 1D 卷积"的区别
- [ ] 知道 3 种图划分策略各自的作用

## 常见问题

**Q: 为什么需要归一化邻接矩阵？**
A: 避免度数大的节点特征值爆炸，保证训练数值稳定。

**Q: 骨骼图是无向图还是有向图？**
A: 骨骼连接是无向的（边双向都有信息流动），但 ST-GCN 的划分策略可以引入方向性。

**Q: 我的数据每帧骨骼有 17 个点，输入维度是 (17, 3)，经过 GCN 后还是 (17, ?) 吗？**
A: 是的！GCN 不改变节点数量，只改变每个节点的特征维度。输入(17,3) → 输出(17, d_out)。

---

## 下一步

完成本步后，进入 [Step 03: ST-GCN 模型实现](step03_stgcn_model.md) 用 PyTorch 完整实现 ST-GCN。
