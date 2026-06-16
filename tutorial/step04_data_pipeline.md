# Step 04: 数据预处理流水线

## 学习目标

- [ ] 编写从 `.pkl` 文件中提取标签和数据的数据加载器
- [ ] 理解骨骼数据的归一化方法（样本内 vs 数据集全局 vs 批量）
- [ ] 处理变长序列：padding、interpolation 到固定帧数
- [ ] 实现基本数据增强（随机旋转、平移、时间缩放）
- [ ] 构建完整的预处理 pipeline

## 参考文档

- [PyTorch Dataset 文档](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
- [AIST++ 数据集格式](https://google.github.io/aichoreographer/)
- [TSN 数据增强 (Wang et al.)](https://arxiv.org/abs/1608.00859)

---

## 4.1 数据预处理总览

```
原始数据 (.pkl)
    │
    ├── 1. 提取 keypoints3d_optim → (T_raw, 17, 3)
    ├── 2. 从文件名提取标签 → class_id (0-9)
    ├── 3. 固定帧数处理 (interpolation/padding) → (T_fixed=300, 17, 3)
    ├── 4. 归一化 (样本内 normalization)
    ├── 5. 数据增强 (可选: 旋转、翻转、噪声)
    └── 6. 转换为 tensor → (T, 17, 3) float32
```

## 4.2 标签映射

创建文件 `data/dataset.py`

```python
"""AIST++ 数据集加载器 + 预处理"""
import pickle
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

# ============================================================
# 1. 标签映射
# ============================================================
GENRE_NAMES = {
    'gBR': 'break',
    'gPO': 'pop',
    'gLO': 'lock',
    'gMH': 'middle_hiphop',
    'gLH': 'la_hiphop',
    'gHO': 'house',
    'gWA': 'waack',
    'gKR': 'krump',
    'gJS': 'street_jazz',
    'gJB': 'ballet_jazz',
}

def genre_to_label(genre_str):
    """将舞种字符串映射为整数标签 0-9"""
    genre_order = [
        'gBR', 'gPO', 'gLO', 'gMH', 'gLH',
        'gHO', 'gWA', 'gKR', 'gJS', 'gJB'
    ]
    return genre_order.index(genre_str)

def extract_genre_from_filename(filename):
    """从文件名中提取舞种标签"""
    # 文件名格式: g{genre}_s{scene}_c{music}_d{...}_m{...}_ch{...}.pkl
    return filename.split('_')[0]
```

## 4.3 序列长度处理

```python
# 继续写入 data/dataset.py

# ============================================================
# 2. 固定帧数处理
# ============================================================

def pad_or_truncate(keypoints, target_frames=300):
    """
    使序列帧数一致
    - 大于 target_frames: 中心裁剪
    - 小于 target_frames: 首尾padding（复制首帧/尾帧）

    Args:
        keypoints: (T, 17, 3) numpy array
        target_frames: 目标帧数
    Returns:
        keypoints: (target_frames, 17, 3)
    """
    T = keypoints.shape[0]
    
    if T >= target_frames:
        # 中心裁剪
        start = (T - target_frames) // 2
        return keypoints[start:start + target_frames]
    else:
        # Padding：首尾复制
        pad_before = (target_frames - T) // 2
        pad_after = target_frames - T - pad_before
        
        # 用边缘帧复制填充
        padded = np.concatenate([
            np.repeat(keypoints[:1], pad_before, axis=0),
            keypoints,
            np.repeat(keypoints[-1:], pad_after, axis=0),
        ], axis=0)
        return padded


def interpolate_frames(keypoints, target_frames=300):
    """
    通过线性插值统一帧数（比 padding 更平滑）
    
    Args:
        keypoints: (T, 17, 3) numpy array
        target_frames: 目标帧数
    Returns:
        keypoints: (target_frames, 17, 3)
    """
    T = keypoints.shape[0]
    old_indices = np.linspace(0, T - 1, T)
    new_indices = np.linspace(0, T - 1, target_frames)
    
    # 对每个关节、每个坐标独立插值
    interpolated = np.zeros((target_frames, 17, 3), dtype=keypoints.dtype)
    for joint in range(17):
        for coord in range(3):
            interpolated[:, joint, coord] = np.interp(
                new_indices, old_indices, keypoints[:, joint, coord]
            )
    return interpolated
```

## 4.4 归一化

```python
# ============================================================
# 3. 归一化
# ============================================================

def normalize_keypoints(keypoints):
    """
    样本内归一化：
    1. 以 spine (髋部中心) 为中心原点
    2. 缩放到单位方差

    髋部中心 ≈ (left_hip + right_hip) / 2 = (joint_11 + joint_12) / 2

    Args:
        keypoints: (T, 17, 3)
    Returns:
        keypoints: (T, 17, 3) 归一化后
        mean: 用于逆归一化
        std: 用于逆归一化
    """
    # 第一步：平移→以髋部为中心
    hip_center = (keypoints[:, 11, :] + keypoints[:, 12, :]) / 2  # (T, 3)
    centered = keypoints - hip_center[:, np.newaxis, :]  # (T, 17, 3)
    
    # 第二步：缩放→单位标准差
    # 计算所有关节坐标的标准差
    all_coords = centered.reshape(-1, 3)  # (T*17, 3)
    std = np.std(all_coords)
    if std < 1e-6:
        std = 1.0
    normalized = centered / std
    
    return normalized


def normalize_per_frame(keypoints):
    """
    逐帧归一化（处理相机移动的情况）
    每帧独立计算髋部中心
    """
    # 每帧髋部中心
    hip_center = (keypoints[:, 11, :] + keypoints[:, 12, :]) / 2  # (T, 3)
    centered = keypoints - hip_center[:, np.newaxis, :]
    
    # 全局 std 缩放
    std = np.std(centered.reshape(-1, 3))
    if std < 1e-6:
        std = 1.0
    return centered / std
```

## 4.5 数据增强

```python
# ============================================================
# 4. 数据增强 (Data Augmentation)
# ============================================================

class SkeletonAugmentation:
    """骨骼数据增强：仅在训练时使用"""
    
    @staticmethod
    def random_rotate(keypoints, max_angle=10):
        """
        绕 Y 轴（垂直轴）随机旋转
        模拟不同拍摄角度
        """
        angle = np.random.uniform(-max_angle, max_angle)
        theta = np.radians(angle)
        cos_a, sin_a = np.cos(theta), np.sin(theta)
        
        # 绕 Y 轴旋转矩阵 (只旋转 XZ 平面)
        R = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a],
        ], dtype=keypoints.dtype)
        
        # keypoints: (T, 17, 3)
        T, V = keypoints.shape[:2]
        flat = keypoints.reshape(-1, 3) @ R.T  # 矩阵乘法
        return flat.reshape(T, V, 3)
    
    @staticmethod
    def random_translate(keypoints, max_delta=0.1):
        """随机平移"""
        delta = np.random.uniform(-max_delta, max_delta, size=3)
        return keypoints + delta[np.newaxis, np.newaxis, :]
    
    @staticmethod
    def random_temporal_scale(keypoints, scale_range=(0.8, 1.2)):
        """随机时间缩放 → 模拟不同速度的动作"""
        scale = np.random.uniform(*scale_range)
        T = keypoints.shape[0]
        new_T = max(30, int(T * scale))
        
        old_indices = np.linspace(0, T - 1, T)
        new_indices = np.linspace(0, T - 1, new_T)
        
        scaled = np.zeros((new_T, 17, 3), dtype=keypoints.dtype)
        for j in range(17):
            for c in range(3):
                scaled[:, j, c] = np.interp(new_indices, old_indices, keypoints[:, j, c])
        
        return scaled  # 注意：帧数变了，需要后续 padding
    
    @staticmethod
    def random_noise(keypoints, sigma=0.005):
        """添加高斯噪声（模拟关键点检测误差）"""
        noise = np.random.normal(0, sigma, keypoints.shape).astype(keypoints.dtype)
        return keypoints + noise
```

## 4.6 完整预处理 Pipeline

```python
# ============================================================
# 5. 预处理 Pipeline 主函数
# ============================================================

def preprocess_sample(keypoints, target_frames=300, augment=False):
    """
    对单个样本执行完整的预处理流程

    Args:
        keypoints: (T_raw, 17, 3) numpy array
        target_frames: int, 目标帧数
        augment: bool, 是否启用数据增强（仅训练集用）
    Returns:
        tensor: (target_frames, 17, 3) torch.float32
    """
    # Step 1: 数据增强（在插值前做，利用原始数据）
    if augment:
        aug = SkeletonAugmentation
        keypoints = aug.random_rotate(keypoints)
        keypoints = aug.random_translate(keypoints)
        keypoints = aug.random_temporal_scale(keypoints)  # 帧数可能变化
        keypoints = aug.random_noise(keypoints)
    
    # Step 2: 固定帧数
    keypoints = interpolate_frames(keypoints, target_frames)
    
    # Step 3: 归一化
    keypoints = normalize_keypoints(keypoints)
    
    # Step 4: 转 tensor
    tensor = torch.tensor(keypoints, dtype=torch.float32)
    
    return tensor
```

## 4.7 测试预处理

```python
# ============================================================
# 6. 测试代码
# ============================================================
if __name__ == "__main__":
    DATA_DIR = Path("aist++/keypoints3d")
    
    # 加载一个样本
    import os
    files = os.listdir(DATA_DIR)
    sample_path = DATA_DIR / files[0]
    
    with open(sample_path, 'rb') as f:
        data = pickle.load(f)
    
    kp_raw = data['keypoints3d_optim']  # (T, 17, 3)
    print(f"原始数据: shape={kp_raw.shape}, range=({kp_raw.min():.2f}, {kp_raw.max():.2f})")
    
    # 预处理
    kp_processed = preprocess_sample(kp_raw, target_frames=300, augment=False)
    print(f"处理后: shape={kp_processed.shape}, range=({kp_processed.min():.2f}, {kp_processed.max():.2f})")
    
    # 测试数据增强
    kp_augmented = preprocess_sample(kp_raw, target_frames=300, augment=True)
    print(f"增强后: shape={kp_augmented.shape}")
    
    # 验证归一化效果
    print(f"\n归一化检查:")
    print(f"  mean={kp_processed.mean():.6f}")
    print(f"  std={kp_processed.std():.4f}")
```

运行：
```bash
python data/dataset.py
```

## 项目文件结构（至此）

```
capstone_assignment1/
├── tutorial/           # 本教程
├── models/
│   ├── graph.py        # 邻接矩阵构建
│   └── stgcn.py        # ST-GCN 模型
├── data/
│   └── dataset.py      # 数据加载 + 预处理 (本步创建)
├── scripts/
│   ├── 01_explore_data.py
│   └── 02_gcn_from_scratch.py
├── aist++/
│   └── keypoints3d/    # 原始数据
└── lib/                # ARKit 工具库
```

---

## 检查要点

- [ ] 能从文件名正确提取 10 种舞种标签
- [ ] 理解 interpolation vs padding 的优缺点
- [ ] 理解以髋部为中心原点做归一化的原理
- [ ] 知道训练时增强 (augmentation) 和测试时不增强的区别
- [ ] 预处理后的 tensor shape 固定为 `(300, 17, 3)`

## 常见问题

**Q: 为什么选择 300 帧？不怕信息丢失吗？**
A: AIST++ 的大多数样本在 600-900 帧左右，300 帧是一个平衡精度和计算量的选择。用插值而非裁剪，保留完整动作轨迹。

**Q: 为什么用髋部做归一化原点？**
A: 髋部是身体的几何中心，运动范围相对较小，是人体姿态标准化的自然参考点。

**Q: 数据增强会不会改变动作标签？**
A: 旋转、平移、时间缩放都不会改变动作类别。但镜像翻转需要谨慎——有些非对称动作（如 waack 单侧甩手）可能不适合翻转。

---

## 下一步

完成本步后，进入 [Step 05: 训练/测试集划分 & DataLoader](step05_dataloader.md) 构建 PyTorch DataLoader。
