# Step 01: 环境搭建 & 数据探索

## 学习目标

- [ ] 配置 Python 虚拟环境和 PyTorch
- [ ] 理解 AIST++ 数据集的目录结构和数据格式
- [ ] 学会加载 `.pkl` 文件并探索数据维度
- [ ] 理解 COCO-17 骨骼关键点拓扑结构
- [ ] 可视化一段骨骼动画序列

## 参考文档

- [PyTorch 安装指南](https://pytorch.org/get-started/locally/)
- [AIST++ 官方文档](https://google.github.io/aichoreographer/)
- 项目中的 [aist++/readme.md](../aist++/readme.md)

---

## 1.1 创建虚拟环境

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate  # macOS/Linux

# 安装依赖
pip install torch torchvision
pip install numpy matplotlib scipy
pip install pickle5  # 可选，Python 默认 pickle 也可用
pip install tqdm tensorboard
```

## 1.2 AIST++ 数据结构

### 数据集路径

```
aist++/keypoints3d/
  ├── gBR_sBM_cAll_d04_mBR0_ch01.pkl   # break 舞种，第4个舞段，舞者 BR0，第1个镜头
  ├── gBR_sBM_cAll_d04_mBR0_ch02.pkl
  ├── ...
  ├── gPO_sBM_cAll_d11_mPO2_ch10.pkl   # pop 舞种
  └── ...
```

### 文件名解析规则

```
g{genre}_s{scene}_c{music}_d{dancer_id}_m{motion_id}_ch{choreo_id}.pkl

示例: gBR_sBM_cAll_d04_mBR0_ch01.pkl
  genre  = BR  (break)      ← 这就是动作标签！
  scene  = sBM
  camera = cAll
  dancer = d04 → 舞段4
  motion = mBR0 → 动作序列0
  choreo = ch01 → 镜头1
```

### 每个 `.pkl` 文件的结构

```python
{
    'keypoints3d':        np.ndarray, shape=(N_frames, 17, 3), dtype=float64
    'keypoints3d_optim':  np.ndarray, shape=(N_frames, 17, 3), dtype=float64
}
```

- `keypoints3d`: 原始 3D 关键点
- `keypoints3d_optim`: 优化后的 3D 关键点（推荐使用）
- 第 0 维：时间帧 (frames)，不同文件帧数不同
- 第 1 维：17 个关节 (COCO-17 格式)
- 第 2 维：x, y, z 坐标

### COCO-17 骨骼拓扑 (Skeleton Topology)

```
         0: nose (鼻子)
         |
    1: left_eye ---- 2: right_eye
   3: left_ear        4: right_ear
         |                  |
    5: left_shoulder -- 6: right_shoulder
         |                  |
    7: left_elbow      8: right_elbow
         |                  |
    9: left_wrist      10: right_wrist
         |                  |
   11: left_hip ------- 12: right_hip
         |                  |
   13: left_knee       14: right_knee
         |                  |
   15: left_ankle      16: right_ankle
```

COCO-17 骨骼连接边 (用于构建图的邻接矩阵):

```python
COCO_EDGES = [
    (0, 1), (0, 2),          # nose → eyes
    (1, 3), (2, 4),          # eyes → ears
    (5, 7), (7, 9),          # left arm
    (6, 8), (8, 10),         # right arm
    (5, 6),                  # shoulder center
    (5, 11), (6, 12),        # shoulders → hips
    (11, 12),                # hip center
    (11, 13), (13, 15),      # left leg
    (12, 14), (14, 16),      # right leg
]
```

---

## 1.3 动手实践：加载并探索数据

创建文件 `scripts/01_explore_data.py`

```python
"""Step 01: 数据加载与探索"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# ============================================================
# 1. 配置路径
# ============================================================
DATA_DIR = Path("aist++/keypoints3d")

# ============================================================
# 2. 收集所有 pkl 文件并按舞种分类
# ============================================================
def scan_dataset(data_dir):
    """扫描数据集，返回 {genre: [file_path, ...]}"""
    genre_files = defaultdict(list)
    for fname in os.listdir(data_dir):
        if fname.endswith(".pkl"):
            genre = fname.split("_")[0]  # e.g., "gBR"
            genre_files[genre].append(Path(data_dir) / fname)
    return dict(genre_files)

dataset = scan_dataset(DATA_DIR)

print("=== 数据集概览 ===")
total_samples = 0
for genre, files in sorted(dataset.items()):
    print(f"  {genre}: {len(files)} samples")
    total_samples += len(files)
print(f"  Total: {total_samples} samples")

# ============================================================
# 3. 加载一个样本，查看数据结构
# ============================================================
sample_path = list(dataset.values())[0][0]  # 取第一个舞种的第一个文件
print(f"\n=== 加载样本: {sample_path.name} ===")

with open(sample_path, 'rb') as f:
    data = pickle.load(f)

kp = data['keypoints3d_optim']  # shape (N_frames, 17, 3)
print(f"  形状: {kp.shape}")
print(f"  帧数: {kp.shape[0]}")
print(f"  关节数: {kp.shape[1]}")
print(f"  坐标维度: {kp.shape[2]}")
print(f"  值域范围: [{kp.min():.2f}, {kp.max():.2f}]")
print(f"  第一帧 nose (关节0) 坐标: x={kp[0,0,0]:.3f}, y={kp[0,0,1]:.3f}, z={kp[0,0,2]:.3f}")

# ============================================================
# 4. 统计每个样本的帧数分布
# ============================================================
print("\n=== 帧数统计（前10个样本）===")
frame_counts = []
for genre, files in dataset.items():
    for fp in files[:3]:  # 每个舞种取前3个
        with open(fp, 'rb') as f:
            data = pickle.load(f)
        n_frames = data['keypoints3d_optim'].shape[0]
        frame_counts.append(n_frames)
        print(f"  {fp.name}: {n_frames} frames")

print(f"\n  帧数范围: [{min(frame_counts)}, {max(frame_counts)}]")
print(f"  平均帧数: {np.mean(frame_counts):.1f}")
```

运行：
```bash
python scripts/01_explore_data.py
```

---

## 1.4 可视化骨骼序列

创建文件 `scripts/01_visualize_skeleton.py`

```python
"""Step 01 附加: 骨骼动画可视化"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

# COCO-17 骨骼边
COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# 左右半身颜色区分
LEFT_LIMBS = {(5,7),(7,9),(5,11),(11,13),(13,15)}   # 左手左脚
RIGHT_LIMBS = {(6,8),(8,10),(6,12),(12,14),(14,16)}  # 右手右脚

def load_sample(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['keypoints3d_optim']  # (T, 17, 3)

def visualize_skeleton_3d(keypoints, title="3D Skeleton", save_gif=False):
    """
    使用 matplotlib 3D 可视化骨骼动画
    keypoints: (T, 17, 3)
    """
    T = keypoints.shape[0]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 预计算所有帧的范围，保持视角稳定
    x_min, x_max = keypoints[:,:,0].min(), keypoints[:,:,0].max()
    y_min, y_max = keypoints[:,:,1].min(), keypoints[:,:,1].max()
    z_min, z_max = keypoints[:,:,2].min(), keypoints[:,:,2].max()
    max_range = max(x_max-x_min, y_max-y_min, z_max-z_min) / 2
    mid_x, mid_y, mid_z = (x_min+x_max)/2, (y_min+y_max)/2, (z_min+z_max)/2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    scatter = ax.scatter([], [], [], c='blue', s=50, marker='o')
    lines = []
    
    def update(frame_idx):
        kp = keypoints[frame_idx]  # (17, 3)
        
        # 更新散点
        scatter._offsets3d = (kp[:, 0], kp[:, 1], kp[:, 2])
        
        # 更新线段
        for line in lines:
            line.remove()
        lines.clear()
        
        for (i, j) in COCO_EDGES:
            color = 'red' if (i,j) in LEFT_LIMBS or (j,i) in LEFT_LIMBS else \
                    'green' if (i,j) in RIGHT_LIMBS or (j,i) in RIGHT_LIMBS else 'gray'
            line, = ax.plot(
                [kp[i, 0], kp[j, 0]],
                [kp[i, 1], kp[j, 1]],
                [kp[i, 2], kp[j, 2]],
                color=color, linewidth=2
            )
            lines.append(line)
        
        ax.set_title(f"{title} - Frame {frame_idx+1}/{T}")
        return [scatter] + lines
    
    ani = FuncAnimation(fig, update, frames=T, interval=100, blit=False)
    
    if save_gif:
        ani.save('skeleton_anim.gif', writer='pillow', fps=10)
        print("Saved: skeleton_anim.gif")
    else:
        plt.show()

if __name__ == "__main__":
    # 加载一个样本并可视化前60帧
    sample = Path("aist++/keypoints3d/gBR_sBM_cAll_d04_mBR0_ch01.pkl")
    kp = load_sample(sample)
    visualize_skeleton_3d(kp[:60], title=f"Skeleton: {sample.name}")
```

运行：
```bash
python scripts/01_visualize_skeleton.py
```

---

## 检查要点

- [ ] 能成功加载 `.pkl` 文件并打印 `shape`
- [ ] 理解数据维度 `(T, 17, 3)` 的含义
- [ ] 知道 COCO-17 格式有哪 17 个关节
- [ ] 知道骨骼边的连接方式（哪些关节之间有边）
- [ ] 能从文件名中提取动作标签（genre）

## 常见问题

**Q: `keypoints3d` vs `keypoints3d_optim` 用哪个？**
A: 用 `keypoints3d_optim`，它是经过优化的版本，噪声更少。

**Q: 为什么每帧是 (17, 3) 而不是 (17, 2)？**
A: AIST++ 提供的是 3D 骨骼关键点，包含深度信息(z轴)。

**Q: 数据已经是 COCO 格式，为什么还需要 lib/ 里的转换代码？**
A: `lib/` 中的代码用于 ARKit → COCO 的实时转换（从 iOS 设备接收原始数据流）。本教程直接使用预处理好的 `.pkl` 文件。

---

## 下一步

完成本步后，进入 [Step 02: 图卷积理论](step02_graph_conv_theory.md) 学习 GCN 的数学原理。
