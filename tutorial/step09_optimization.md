# Step 09: 模型优化 & 超参数调优

## 学习目标

- [ ] 理解影响 ST-GCN 性能的关键超参数
- [ ] 使用 Optuna 进行自动超参数搜索
- [ ] 尝试模型架构改进 (更多层、注意力机制)
- [ ] 使用 TensorBoard 对比不同实验
- [ ] 分析训练瓶颈并优化

## 参考文档

- [Optuna 文档](https://optuna.readthedocs.io/)
- [ST-GCN 改进论文 (2s-AGCN)](https://arxiv.org/abs/1904.11588)
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

---

## 9.1 关键超参数空间

| 超参数 | 范围 | 对性能的影响 |
|--------|------|-------------|
| `learning_rate` | 1e-4 ~ 1e-2 | 太大导致震荡，太小收敛慢 |
| `batch_size` | 16 ~ 128 | 大 batch 稳定但泛化差 |
| `base_channels` | 32 ~ 128 | 容量越大越强，越多越慢 |
| `dropout` | 0.2 ~ 0.7 | 正则化强度 |
| `weight_decay` | 1e-5 ~ 1e-2 | L2 正则化 |
| `temporal_kernel` | 3 ~ 15 | 更大 kernel 捕捉更长时间依赖 |
| `target_frames` | 100 ~ 500 | 更多帧=更多信息=更慢 |
| `optimizer` | Adam, AdamW, SGD | AdamW 通常最佳 |
| `scheduler` | Cosine, Plateau, Step | Cosine 最平滑 |

---

## 9.2 使用 Optuna 自动调参

创建文件 `tune_hyperparams.py`

```python
"""使用 Optuna 自动搜索最佳超参数"""
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import json
from datetime import datetime

from models.stgcn import STGCN
from data.dataloader import create_dataloaders


# ============================================================
# 目标函数 (Optuna 会最大化这个值)
# ============================================================
def objective(trial, data_dir="aist++/keypoints3d", max_epochs=30):
    """
    Optuna 目标函数：每个 trial 尝试一组超参数

    Args:
        trial: Optuna trial 对象
        data_dir: 数据集路径
        max_epochs: 每个 trial 的最大训练轮数（小一些加速搜索）
    """
    # ===== 超参数搜索空间 =====
    config = {
        # 模型架构
        'base_channels': trial.suggest_categorical('base_channels', [32, 64, 96, 128]),
        'dropout': trial.suggest_float('dropout', 0.2, 0.7),
        'temporal_kernel': trial.suggest_categorical('temporal_kernel', [3, 5, 7, 9]),
        'target_frames': trial.suggest_categorical('target_frames', [100, 200, 300]),
        
        # 优化器
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        
        # 固定参数
        'num_classes': 10,
        'num_epochs': max_epochs,
        'num_workers': 4,
    }
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"Trial #{trial.number}: {config}")
    print(f"{'='*60}")
    
    # ===== 数据 =====
    train_loader, val_loader, _ = create_dataloaders(
        data_dir,
        target_frames=config['target_frames'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )
    
    # ===== 模型 =====
    model = STGCN(
        num_classes=config['num_classes'],
        base_channels=config['base_channels'],
        dropout=config['dropout'],
        temporal_kernel=config['temporal_kernel'],
    ).to(device)
    
    # ===== 损失函数 + 优化器 + 调度器 =====
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
    
    # ===== 训练循环 (简化版) =====
    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = 8
    
    for epoch in range(1, max_epochs + 1):
        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            x = batch['keypoints'].to(device)
            y = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += x.size(0)
        
        train_acc = train_correct / train_total
        scheduler.step()
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['keypoints'].to(device)
                y = batch['label'].to(device)
                logits = model(x)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += x.size(0)
        
        val_acc = val_correct / val_total
        
        # 报告给 Optuna（用于剪枝）
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        # 跟踪最佳
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= early_stop_patience:
            break
    
    return best_val_acc


# ============================================================
# 运行超参数搜索
# ============================================================
def run_optuna_study(n_trials=50, study_name="stgcn_optimization"):
    """
    运行 Optuna 超参数搜索

    n_trials=50 预计耗时: ~2-4 小时（取决于 GPU）
    """
    # 使用 TPESampler (贝叶斯优化)
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    
    study.optimize(
        lambda trial: objective(trial),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    
    # 结果
    print(f"\n{'='*60}")
    print(f"超参数搜索完成!")
    print(f"{'='*60}")
    print(f"\n最佳 Trial: #{study.best_trial.number}")
    print(f"最佳验证准确率: {study.best_value:.4f}")
    print(f"\n最佳超参数:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    
    # 保存结果
    results = {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'best_trial': study.best_trial.number,
        'n_trials': n_trials,
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/optuna_{study_name}_{timestamp}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果保存到: {output_path}")
    
    # 可视化
    try:
        import optuna.visualization as vis
        fig1 = vis.plot_optimization_history(study)
        fig2 = vis.plot_param_importances(study)
        fig3 = vis.plot_parallel_coordinate(study)
        fig1.show()
        fig2.show()
    except ImportError:
        print("安装 plotly 以查看可视化: pip install plotly")
    
    return study


if __name__ == "__main__":
    study = run_optuna_study(n_trials=50)
```

运行：
```bash
pip install optuna plotly
python tune_hyperparams.py
```

---

## 9.3 模型架构改进方向

### 改进 1: 增加注意力机制

```python
class STGCNWithAttention(nn.Module):
    """
    ST-GCN + 时间注意力
    让模型学习关注序列中更重要的帧
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.stgcn = STGCN(*args, **kwargs)
        
        # 时间注意力层
        self.temporal_attention = nn.Sequential(
            nn.Linear(256, 64),  # 256 = last channel dim
            nn.Tanh(),
            nn.Linear(64, 1),
        )
    
    def forward(self, x):
        # 获取 ST-GCN 中间特征 (在 global pooling 之前)
        # ... 需要修改 STGCN 的 forward
        pass
```

### 改进 2: 多尺度时间卷积

```python
class MultiScaleTemporalConv(nn.Module):
    """多尺度时间卷积：同时用 kernel=3,5,7,9 卷积再合并"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.k3 = nn.Conv2d(in_channels, out_channels//4, (3, 1), padding=(1, 0))
        self.k5 = nn.Conv2d(in_channels, out_channels//4, (5, 1), padding=(2, 0))
        self.k7 = nn.Conv2d(in_channels, out_channels//4, (7, 1), padding=(3, 0))
        self.k9 = nn.Conv2d(in_channels, out_channels//4, (9, 1), padding=(4, 0))
    
    def forward(self, x):
        return torch.cat([self.k3(x), self.k5(x), self.k7(x), self.k9(x)], dim=1)
```

### 改进 3: 数据驱动邻接矩阵

ST-GCN 原论文之外，可以添加一个**全自适应邻接矩阵**，与骨骼固有邻接矩阵并行使用：

```python
# 在 SpatialGraphConv 中添加可学习的全局邻接矩阵
self.adaptive_adj = nn.Parameter(torch.randn(num_joints, num_joints) * 0.01)
# A_adaptive = softmax(adaptive_adj, dim=1)  → 数据驱动的图结构
```

---

## 9.4 性能剖析 (Profiling)

```python
def profile_model(model, input_shape=(1, 300, 17, 3), device='cuda'):
    """使用 PyTorch Profiler 分析性能瓶颈"""
    from torch.profiler import profile, ProfilerActivity
    
    x = torch.randn(input_shape).to(device)
    model = model.to(device).eval()
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(10):
                model(x)
    
    print("\n=== Profiling 结果 ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    
    return prof
```

---

## 9.5 实验管理

### TensorBoard 多实验对比

修改 `train.py` 中 TensorBoard 的 log_dir：

```python
# 每个超参数组合有自己的 log 子目录
writer = SummaryWriter(
    log_dir=f"logs/stgcn_lr{lr}_bs{batch_size}_ch{base_channels}"
)
```

然后在 TensorBoard 中可以对比多个实验的曲线。

### 实验记录表

记录每次实验的关键参数和结果，帮助系统化迭代：

```markdown
| Experiment | LR   | Batch | Ch | Dropout | TempK | Val Acc | Notes |
|-----------|------|-------|-----|---------|-------|---------|-------|
| baseline  | 1e-3 | 32    | 64  | 0.5     | 9     | 0.83    | -     |
| exp_01    | 1e-3 | 64    | 128 | 0.5     | 9     | 0.85    | ↑ch   |
| exp_02    | 5e-4 | 32    | 128 | 0.3     | 9     | 0.84    | ↓drop |
| exp_03    | 1e-3 | 32    | 128 | 0.5     | 15    | 0.86    | ↑tk   |
```

---

## 检查要点

- [ ] 理解每个超参数的含义和合理范围
- [ ] 能运行 Optuna 做自动化超参数搜索
- [ ] 知道如何用 TensorBoard 对比实验
- [ ] 理解学习率是影响最大的超参数

## 常见问题

**Q: Optuna 运行太慢了怎么办？**
A: 减少 `max_epochs`（如 15-20）；使用 `MedianPruner` 提前终止差的 trial；减少搜索空间维度。

**Q: 调参后验证集变好了但测试集没变好？**
A: 这是"对验证集过拟合"。需要更严格的验证策略（如 cross-validation），或增加数据量。

**Q: 最好的参数组合为什么不符合直觉？**
A: 深度学习超参数间存在复杂交互。相信实验结果而非直觉。

---

## 下一步

完成本步后，进入 [Step 10: 模型导出](step10_export.md) 将模型转换为部署格式。
