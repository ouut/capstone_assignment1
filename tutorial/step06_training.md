# Step 06: 模型训练循环

## 学习目标

- [ ] 编写标准的 PyTorch 训练循环
- [ ] 理解损失函数（CrossEntropyLoss）和优化器（SGD/Adam）
- [ ] 实现验证循环和 checkpoint 保存
- [ ] 使用 TensorBoard 记录训练曲线
- [ ] 处理常见训练问题（过拟合、学习率震荡）

## 参考文档

- [PyTorch 训练最佳实践](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [Adam 优化器论文](https://arxiv.org/abs/1412.6980)
- [TensorBoard with PyTorch](https://pytorch.org/docs/stable/tensorboard.html)

---

## 6.1 训练循环模板

创建文件 `train.py`

```python
"""ST-GCN 训练脚本"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time
import json
from datetime import datetime

from models.stgcn import STGCN
from data.dataloader import create_dataloaders


# ============================================================
# 1. 配置类
# ============================================================
class Config:
    # 数据
    data_dir = "aist++/keypoints3d"
    target_frames = 300
    num_classes = 10
    
    # 训练
    batch_size = 32
    num_epochs = 80
    learning_rate = 0.001
    weight_decay = 1e-4
    num_workers = 4
    
    # 模型
    base_channels = 64
    dropout = 0.5
    
    # 输出
    output_dir = "outputs"
    log_dir = "logs"
    checkpoint_dir = "checkpoints"
    
    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 早停
    early_stop_patience = 15
    
    # 混合精度
    use_amp = torch.cuda.is_available()


# ============================================================
# 2. 工具函数
# ============================================================
def format_time(seconds):
    """格式化时间"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class AverageMeter:
    """跟踪并计算均值"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ============================================================
# 3. 训练一个 epoch
# ============================================================
def train_one_epoch(model, dataloader, criterion, optimizer, scheduler,
                    device, epoch, writer, scaler=None):
    """
    训练一个 epoch
    """
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        x = batch['keypoints'].to(device, non_blocking=True)  # (B, T, V, 3)
        y = batch['label'].to(device, non_blocking=True)      # (B,)
        
        optimizer.zero_grad()
        
        # 混合精度训练
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(x)           # (B, num_classes)
            loss = criterion(logits, y)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # 计算准确率
        acc = (logits.argmax(dim=1) == y).float().mean()
        
        # 更新统计
        batch_size = x.size(0)
        losses.update(loss.item(), batch_size)
        top1.update(acc.item(), batch_size)
        
        # 每 20 个 batch 打印一次
        if batch_idx % 20 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch [{epoch:3d}] Batch [{batch_idx:3d}/{len(dataloader):3d}]  "
                  f"Loss: {losses.val:.4f} (avg: {losses.avg:.4f})  "
                  f"Acc: {top1.val:.3f} (avg: {top1.avg:.3f})  "
                  f"LR: {lr:.6f}")
    
    elapsed = time.time() - start_time
    
    # TensorBoard 记录
    if writer:
        writer.add_scalar('train/loss', losses.avg, epoch)
        writer.add_scalar('train/acc', top1.avg, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
    
    return losses.avg, top1.avg, elapsed


# ============================================================
# 4. 验证一个 epoch
# ============================================================
@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, writer):
    """验证"""
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    
    # 收集所有预测用于混淆矩阵
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        x = batch['keypoints'].to(device, non_blocking=True)
        y = batch['label'].to(device, non_blocking=True)
        
        logits = model(x)
        loss = criterion(logits, y)
        
        preds = logits.argmax(dim=1)
        
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())
        
        acc = (preds == y).float().mean()
        
        batch_size = x.size(0)
        losses.update(loss.item(), batch_size)
        top1.update(acc.item(), batch_size)
    
    # 汇总
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # 每个类别的准确率
    per_class_acc = {}
    for cls_idx in range(10):
        mask = (all_labels == cls_idx)
        if mask.sum() > 0:
            per_class_acc[cls_idx] = (all_preds[mask] == cls_idx).float().mean().item()
        else:
            per_class_acc[cls_idx] = 0.0
    
    if writer:
        writer.add_scalar('val/loss', losses.avg, epoch)
        writer.add_scalar('val/acc', top1.avg, epoch)
    
    return losses.avg, top1.avg, all_preds, all_labels, per_class_acc


# ============================================================
# 5. 主训练函数
# ============================================================
def train(config=None):
    """完整训练流程"""
    config = config or Config()
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir) / f"train_{timestamp}"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(output_dir / "config.json", 'w') as f:
        json.dump({k: v for k, v in vars(config).items()
                   if not k.startswith('_')}, f, indent=2, default=str)
    
    print(f"设备: {config.device}")
    print(f"输出目录: {output_dir}")
    
    # ===== 数据 =====
    print("\n加载数据...")
    train_loader, val_loader, test_loader = create_dataloaders(
        config.data_dir,
        target_frames=config.target_frames,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    
    # ===== 模型 =====
    print("\n创建模型...")
    model = STGCN(
        num_classes=config.num_classes,
        base_channels=config.base_channels,
        dropout=config.dropout,
    ).to(config.device)
    
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {n_params:,} (可训练: {n_trainable:,})")
    
    # ===== 损失函数 (带类别权重，处理不平衡) =====
    # 统计训练集各类别样本数
    train_labels = torch.tensor(train_loader.dataset.labels)
    class_counts = torch.bincount(train_labels, minlength=config.num_classes)
    class_weights = len(train_labels) / (config.num_classes * class_counts)
    class_weights = class_weights.to(config.device)
    
    print(f"类别分布: {class_counts.tolist()}")
    print(f"类别权重: {class_weights.tolist()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # ===== 优化器 =====
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # ===== 学习率调度器 =====
    # 余弦退火: 训练后期自动降低学习率
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
        eta_min=config.learning_rate * 0.01,
    )
    
    # ===== 混合精度 =====
    scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
    
    # ===== TensorBoard =====
    writer = SummaryWriter(log_dir=output_dir / "tensorboard")
    
    # ===== 训练循环 =====
    print(f"\n{'='*60}")
    print(f"开始训练: {config.num_epochs} epochs")
    print(f"{'='*60}")
    
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    train_history = []
    
    total_start = time.time()
    
    for epoch in range(1, config.num_epochs + 1):
        print(f"\n--- Epoch {epoch}/{config.num_epochs} ---")
        
        # 训练
        train_loss, train_acc, train_time = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            config.device, epoch, writer, scaler
        )
        
        # 验证
        val_loss, val_acc, val_preds, val_labels, per_class_acc = validate(
            model, val_loader, criterion, config.device, epoch, writer
        )
        
        # 学习率调度
        scheduler.step()
        
        # 记录历史
        train_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        })
        
        # 打印 epoch 摘要
        print(f"\n  Epoch {epoch:3d} Summary:")
        print(f"    Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  Time: {format_time(train_time)}")
        print(f"    Val   Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")
        print(f"    每类准确率: ", end="")
        for cls in sorted(per_class_acc.keys()):
            print(f"[{cls}]:{per_class_acc[cls]:.2f}", end=" ")
        print()
        
        # 保存最佳模型
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': vars(config),
            }
            torch.save(checkpoint, checkpoint_dir / "best_model.pt")
            print(f"  ✓ 保存最佳模型 (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
        
        # 定期保存 checkpoint
        if epoch % 10 == 0:
            torch.save(checkpoint, checkpoint_dir / f"checkpoint_epoch{epoch}.pt")
        
        # 早停
        if patience_counter >= config.early_stop_patience:
            print(f"\n早停: {config.early_stop_patience} epochs 内验证准确率未提升")
            print(f"最佳 epoch: {best_epoch}, 最佳准确率: {best_val_acc:.4f}")
            break
    
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"  总时间: {format_time(total_time)}")
    print(f"  最佳 Epoch: {best_epoch}")
    print(f"  最佳验证准确率: {best_val_acc:.4f}")
    
    # 保存训练历史
    with open(output_dir / "train_history.json", 'w') as f:
        json.dump(train_history, f, indent=2)
    
    writer.close()
    
    return model, train_history, best_val_acc


# ============================================================
# 6. 启动训练
# ============================================================
if __name__ == "__main__":
    config = Config()
    model, history, best_acc = train(config)
```

---

## 6.2 训练技巧 Checklist

### 学习率设置

```python
# 方法1: 固定学习率
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 方法2: 余弦退火（推荐）
scheduler = CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-5)

# 方法3: 验证停滞时衰减
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
```

### 过拟合应对

| 策略 | 代码 |
|------|------|
| Dropout | `nn.Dropout(p=0.5)` |
| 权重衰减 | `weight_decay=1e-4` |
| 数据增强 | `augment=True` (Step 04) |
| 早停 | `early_stop_patience=15` |
| Label Smoothing | `nn.CrossEntropyLoss(label_smoothing=0.1)` |

### 混合精度训练 (AMP)

```python
# 节省 GPU 显存 ~40%，略微加速
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    logits = model(x)
    loss = criterion(logits, y)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 6.3 监控训练

### TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir outputs/train_YYYYMMDD_HHMMSS/tensorboard --port 6006
```

然后打开浏览器 `http://localhost:6006`

关注曲线：
- `train/loss` vs `val/loss`: 如果 val loss 上升 → 过拟合
- `train/acc` vs `val/acc`: 如果差距大 → 过拟合
- `train/lr`: 确认学习率调度正常

### 预期训练曲线

```
Epoch 1:   Train Loss: 2.1  Acc: 0.30  Val Loss: 2.0  Acc: 0.35
Epoch 10:  Train Loss: 1.2  Acc: 0.55  Val Loss: 1.4  Acc: 0.52
Epoch 30:  Train Loss: 0.6  Acc: 0.78  Val Loss: 0.9  Acc: 0.72
Epoch 60:  Train Loss: 0.3  Acc: 0.92  Val Loss: 0.6  Acc: 0.80
Epoch 80:  Train Loss: 0.2  Acc: 0.96  Val Loss: 0.5  Acc: 0.83 ✓
```

---

## 6.4 运行训练

```bash
# 确保目录结构
mkdir -p models data scripts outputs checkpoints logs

# 运行训练
python train.py
```

## 6.5 断点续训

如果训练中断，可以从 checkpoint 恢复：

```python
def resume_training(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"从 Epoch {start_epoch} 恢复训练")
    return start_epoch
```

---

## 检查要点

- [ ] 训练循环中正确调用了 `model.train()` 和 `model.eval()`
- [ ] 使用 `torch.no_grad()` 包裹验证代码
- [ ] TensorBoard 记录了 loss 和 accuracy
- [ ] 最佳模型已保存到 `checkpoints/best_model.pt`
- [ ] 理解过拟合的预警信号

## 常见问题

**Q: Loss 不下降怎么办？**
A: 检查学习率是否太大/太小；检查数据归一化是否正确；尝试先用小数据过拟合。

**Q: 训练很慢，GPU 利用率低怎么办？**
A: 增大 batch_size；增加 num_workers；启用 pin_memory；检查是否有 CPU 瓶颈。

**Q: 验证集准确率波动很大怎么办？**
A: 增大 batch_size（获得更稳定的梯度）；减小学习率；检查数据增强是否过于激进。

---

## 下一步

完成本步后，进入 [Step 07: 模型评估](step07_evaluation.md) 深入分析模型性能。
