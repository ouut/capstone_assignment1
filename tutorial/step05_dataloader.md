# Step 05: 训练/测试集划分 & PyTorch DataLoader

## 学习目标

- [ ] 按舞者/舞段划分训练集和测试集（避免数据泄漏）
- [ ] 理解 `torch.utils.data.Dataset` 接口
- [ ] 实现 `collate_fn` 处理 batch 数据
- [ ] 使用 `DataLoader` 进行高效批量加载
- [ ] 检查数据加载的正确性和类别均衡性

## 参考文档

- [PyTorch Dataset & DataLoader](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
- [PyTorch DataLoader 文档](https://pytorch.org/docs/stable/data.html)

---

## 5.1 数据集划分策略

### ⚠️ 关键原则：按舞者/舞段划分，而非随机按样本划分

如果随机划分样本，同一个舞者的同一个舞段可能同时出现在训练集和测试集中，
模型将更容易识别（因为看到过同一个人跳同一段舞），导致测试集准确率虚高。

正确做法是**按舞者划分**：

```
训练集: 舞者 0, 1, 2, 3   (80% 舞者)
验证集: 舞者 4            (10% 舞者)
测试集: 舞者 5            (10% 舞者)
```

从文件名中提取舞者 ID：`gBR_sBM_cAll_d04_mBR0_ch01.pkl` → `mBR0`

```python
# 提取舞者 ID
filename = "gBR_sBM_cAll_d04_mBR0_ch01.pkl"
parts = filename.split('_')
dancer_id = parts[4]  # "mBR0"
motion_id = parts[4]  # 也是 dancer 相关的

# 或者更精确地用 d{number} 标识舞段
# g{genre}_s{scene}_c{music}_d{dancer_num}_m{motion}_{choreo_id}.pkl
dancer_num = filename.split('_')[3]  # "d04"
```

本教程建议同时按舞段(`d{num}`)划分：
- 相同舞段的不同镜头 (`ch{num}`) 应该在同一集合中

---

## 5.2 实现 PyTorch Dataset

创建文件 `data/dataloader.py`

```python
"""PyTorch Dataset + DataLoader for AIST++"""
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split

from data.dataset import (
    GENRE_NAMES, genre_to_label, extract_genre_from_filename,
    preprocess_sample, interpolate_frames, normalize_keypoints
)

# ============================================================
# 1. 数据集划分
# ============================================================

def split_by_dancer(data_dir, test_size=0.1, val_size=0.1, random_seed=42):
    """
    按舞者划分数据集

    Returns:
        train_files, val_files, test_files: list of Path
    """
    # Step 1: 收集所有文件，按舞者分组
    dancer_files = defaultdict(list)
    for fname in os.listdir(data_dir):
        if fname.endswith('.pkl'):
            # 提取 dancer ID: g{genre}_s{scene}_c{music}_d{dancer_num}_...
            parts = fname.split('_')
            dancer_id = parts[3]  # "d04"
            dancer_files[dancer_id].append(Path(data_dir) / fname)
    
    dancer_ids = sorted(dancer_files.keys())
    print(f"总舞者数: {len(dancer_ids)}")
    for did in dancer_ids:
        print(f"  {did}: {len(dancer_files[did])} samples")
    
    # Step 2: 按舞者划分
    train_dancers, test_val_dancers = train_test_split(
        dancer_ids, test_size=test_size + val_size, random_state=random_seed
    )
    val_dancers, test_dancers = train_test_split(
        test_val_dancers,
        test_size=test_size / (test_size + val_size),
        random_state=random_seed
    )
    
    # Step 3: 收集各集合的文件
    train_files = [f for d in train_dancers for f in dancer_files[d]]
    val_files = [f for d in val_dancers for f in dancer_files[d]]
    test_files = [f for d in test_dancers for f in dancer_files[d]]
    
    print(f"\n划分结果:")
    print(f"  训练集: {len(train_files)} samples ({len(train_dancers)} dancers)")
    print(f"  验证集: {len(val_files)} samples ({len(val_dancers)} dancers)")
    print(f"  测试集: {len(test_files)} samples ({len(test_dancers)} dancers)")
    
    return train_files, val_files, test_files


def check_class_balance(files):
    """检查类别分布"""
    genre_counts = defaultdict(int)
    for fp in files:
        genre = extract_genre_from_filename(fp.name)
        genre_counts[genre] += 1
    print("\n类别分布:")
    for genre in sorted(genre_counts.keys()):
        name = GENRE_NAMES.get(genre, 'unknown')
        print(f"  {genre} ({name}): {genre_counts[genre]}")
    return dict(genre_counts)


# ============================================================
# 2. PyTorch Dataset
# ============================================================

class AISTDataset(Dataset):
    """
    AIST++ 动作识别数据集

    每个样本:
    - keypoints: (T=300, V=17, C=3) tensor
    - label: int (0-9)
    """
    
    def __init__(self, file_list, target_frames=300, augment=False,
                 cache_data=False):
        """
        Args:
            file_list: list of Path, 文件路径列表
            target_frames: int, 统一帧数
            augment: bool, 是否使用数据增强
            cache_data: bool, 是否预加载到内存（数据量大时慎用）
        """
        self.files = file_list
        self.target_frames = target_frames
        self.augment = augment
        self.cache_data = cache_data
        
        # 预提取标签（不需要读取文件）
        self.labels = [
            genre_to_label(extract_genre_from_filename(fp.name))
            for fp in self.files
        ]
        
        # 可选：预加载到内存
        self.cache = {}
        if cache_data:
            print(f"预加载 {len(self.files)} 个样本到内存...")
            for idx, fp in enumerate(self.files):
                self.cache[idx] = self._load_and_process(fp)
                if (idx + 1) % 100 == 0:
                    print(f"  {idx + 1}/{len(self.files)}")
    
    def _load_and_process(self, filepath):
        """加载单个 pkl 文件并预处理"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        keypoints = data['keypoints3d_optim']
        return preprocess_sample(
            keypoints, target_frames=self.target_frames,
            augment=self.augment
        )
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # 从缓存或磁盘加载
        if idx in self.cache:
            keypoints = self.cache[idx]
        else:
            keypoints = self._load_and_process(self.files[idx])
        
        label = self.labels[idx]
        
        return {
            'keypoints': keypoints,     # (T, 17, 3)
            'label': torch.tensor(label, dtype=torch.long),
            'filename': self.files[idx].name,
        }
    
    def sample_weights(self):
        """
        计算类别权重（用于处理不平衡数据集）
        """
        labels = np.array(self.labels)
        class_counts = np.bincount(labels, minlength=10)
        weights = 1.0 / class_counts
        sample_weights = weights[labels]
        return torch.tensor(sample_weights, dtype=torch.float32)


# ============================================================
# 3. Collate Function
# ============================================================

def collate_fn(batch):
    """
    将列表形式的 batch 转换为 tensor batch
    输入已经是固定帧数，直接 stack 即可
    """
    keypoints = torch.stack([item['keypoints'] for item in batch], dim=0)
    labels = torch.stack([item['label'] for item in batch])
    filenames = [item['filename'] for item in batch]
    
    return {
        'keypoints': keypoints,   # (B, T, V, 3)
        'label': labels,          # (B,)
        'filename': filenames,
    }


# ============================================================
# 4. DataLoader 工厂函数
# ============================================================

def create_dataloaders(data_dir, target_frames=300, batch_size=32,
                       num_workers=4, cache_data=False):
    """
    创建训练/验证/测试 DataLoader
    """
    # 划分数据集
    train_files, val_files, test_files = split_by_dancer(Path(data_dir))
    
    # 检查类别平衡
    print("\n=== 训练集 ===")
    check_class_balance(train_files)
    print("\n=== 测试集 ===")
    check_class_balance(test_files)
    
    # 创建 Dataset
    train_dataset = AISTDataset(
        train_files, target_frames=target_frames,
        augment=True, cache_data=cache_data
    )
    val_dataset = AISTDataset(
        val_files, target_frames=target_frames,
        augment=False, cache_data=cache_data
    )
    test_dataset = AISTDataset(
        test_files, target_frames=target_frames,
        augment=False, cache_data=cache_data
    )
    
    # 创建 DataLoader
    g = torch.Generator()
    g.manual_seed(42)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        generator=g, pin_memory=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


# ============================================================
# 5. 测试代码
# ============================================================
if __name__ == "__main__":
    DATA_DIR = "aist++/keypoints3d"
    
    # 测试划分
    train_files, val_files, test_files = split_by_dancer(Path(DATA_DIR))
    
    # 测试 Dataset
    train_dataset = AISTDataset(train_files, target_frames=300, augment=True)
    sample = train_dataset[0]
    print(f"\n第一个样本:")
    print(f"  keypoints shape: {sample['keypoints'].shape}")
    print(f"  label: {sample['label']}")
    print(f"  filename: {sample['filename']}")
    
    # 测试 DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True,
        num_workers=0, collate_fn=collate_fn
    )
    
    batch = next(iter(train_loader))
    print(f"\n一个 batch:")
    print(f"  keypoints shape: {batch['keypoints'].shape}")  # (4, 300, 17, 3)
    print(f"  labels shape: {batch['label'].shape}")        # (4,)
    print(f"  labels: {batch['label']}")
    print(f"  filenames: {batch['filename']}")
```

运行：
```bash
python data/dataloader.py
```

## 5.3 关键设计决策

### 1. 按舞者划分 vs 随机划分

```python
# ❌ 错误做法: 随机划分 → 数据泄漏
X_train, X_test = train_test_split(all_files, test_size=0.2)

# ✅ 正确做法: 按舞者划分
dancer_files = group_by_dancer(all_files)
train_dancers, test_dancers = train_test_split(dancer_ids, test_size=0.2)
```

### 2. `pin_memory=True` 的作用

当使用 GPU 时，DataLoader 会将数据放在锁页内存中，加速 CPU→GPU 传输。

### 3. `num_workers` 选择

- `num_workers=0`: 主进程加载（调试用）
- `num_workers=4`: 4 个子进程并行加载 I/O（生产用）
- 原则：不应超过 CPU 核心数

### 4. `prefetch_factor`

DataLoader 会提前加载 `prefetch_factor * batch_size` 个样本，减少 GPU 等待。

## 5.4 数据流完整图

```
aist++/keypoints3d/*.pkl
    │
    ├── split_by_dancer()
    │   ├── train_files (80%)
    │   ├── val_files (10%)
    │   └── test_files (10%)
    │
    ├── AISTDataset (继承 torch.utils.data.Dataset)
    │   ├── __getitem__(idx): 加载 → 预处理 → 增强 → tensor
    │   └── __len__()
    │
    ├── collate_fn(batch): stack → batch tensor
    │
    └── DataLoader
        ├── train_loader → (B, T, V, 3) + (B,) labels
        ├── val_loader
        └── test_loader
```

---

## 检查要点

- [ ] 理解为什么按舞者划分而不是随机按样本划分
- [ ] Dataset 返回的样本是 `(T, V, 3)` 格式
- [ ] DataLoader batch 输出是 `(B, T, V, 3)` 格式
- [ ] 训练集 `augment=True`，验证/测试集 `augment=False`
- [ ] 能检查类别分布是否均衡

## 常见问题

**Q: 我的数据量很大，内存装不下怎么办？**
A: 不要设置 `cache_data=True`，每次 `__getitem__` 从磁盘读取。或者使用 `lmdb` / `h5py` 做内存映射。

**Q: GPU 等待数据加载，怎么办？**
A: 增大 `num_workers`、启用 `pin_memory`、增大 `prefetch_factor`、或者把数据预处理后保存为 `.pt` 文件。

**Q: 类别不平衡怎么办？**
A: 使用 `WeightedRandomSampler` 或在损失函数中使用 `class_weights`。

---

## 下一步

完成本步后，进入 [Step 06: 模型训练循环](step06_training.md) 编写训练代码。
