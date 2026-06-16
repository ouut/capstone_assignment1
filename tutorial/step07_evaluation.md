# Step 07: 模型评估 & 指标分析

## 学习目标

- [ ] 使用测试集评估模型最终性能
- [ ] 计算和解读混淆矩阵 (Confusion Matrix)
- [ ] 计算 per-class precision、recall、F1
- [ ] 可视化评估结果
- [ ] 分析模型的常见错误模式

## 参考文档

- [scikit-learn Metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)
- [混淆矩阵解读](https://en.wikipedia.org/wiki/Confusion_matrix)

---

## 7.1 评估脚本

创建文件 `evaluate.py`

```python
"""ST-GCN 模型评估脚本"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)
from pathlib import Path
import json

from models.stgcn import STGCN
from data.dataloader import create_dataloaders
from data.dataset import GENRE_NAMES


class ModelEvaluator:
    """
    模型评估器：加载训练好的模型，在测试集上计算各种指标
    """
    
    def __init__(self, checkpoint_path, data_dir="aist++/keypoints3d",
                 device=None):
        """
        Args:
            checkpoint_path: str, 最佳模型路径
            data_dir: str, 数据集路径
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        
        # 加载 checkpoint
        print(f"加载模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 恢复配置
        config = checkpoint.get('config', {})
        num_classes = config.get('num_classes', 10)
        base_channels = config.get('base_channels', 64)
        dropout = config.get('dropout', 0.5)
        
        # 创建模型并加载权重
        self.model = STGCN(
            num_classes=num_classes,
            base_channels=base_channels,
            dropout=dropout,
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"  验证准确率 (checkpoint): {checkpoint['val_acc']:.4f}")
        print(f"  Epoch (checkpoint): {checkpoint['epoch']}")
        
        self.num_classes = num_classes
        self.class_names = [
            GENRE_NAMES.get(f'g{g}', g)
            for g in ['BR','PO','LO','MH','LH','HO','WA','KR','JS','JB']
        ]
        
        # 加载数据
        _, _, self.test_loader = create_dataloaders(
            data_dir,
            target_frames=config.get('target_frames', 300),
            batch_size=config.get('batch_size', 32),
            num_workers=4,
        )
        print(f"  测试样本数: {len(self.test_loader.dataset)}")
    
    @torch.no_grad()
    def evaluate(self):
        """
        完整评估：输出所有指标
        """
        all_preds = []
        all_labels = []
        all_probs = []  # softmax 概率
        all_filenames = []
        
        for batch in self.test_loader:
            x = batch['keypoints'].to(self.device)
            y = batch['label'].to(self.device)
            
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            all_probs.append(probs.cpu())
            all_filenames.extend(batch['filename'])
        
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_probs = torch.cat(all_probs).numpy()
        
        # ========== 核心指标 ==========
        self.accuracy = accuracy_score(all_labels, all_preds)
        self.precision, self.recall, self.f1, _ = \
            precision_recall_fscore_support(all_labels, all_preds, average=None)
        
        # Macro/Micro 平均
        self.macro_f1 = np.mean(self.f1)
        self.micro_f1 = precision_recall_fscore_support(
            all_labels, all_preds, average='micro'
        )[2]
        
        self.confusion_mat = confusion_matrix(all_labels, all_preds)
        
        # 存储用于后续分析
        self.all_preds = all_preds
        self.all_labels = all_labels
        self.all_probs = all_probs
        self.all_filenames = all_filenames
        
        return self._summarize()
    
    def _summarize(self):
        """打印评估摘要"""
        print(f"\n{'='*60}")
        print(f"测试集评估结果")
        print(f"{'='*60}")
        print(f"  Accuracy:         {self.accuracy:.4f}")
        print(f"  Macro Precision:  {np.mean(self.precision):.4f}")
        print(f"  Macro Recall:     {np.mean(self.recall):.4f}")
        print(f"  Macro F1:         {self.macro_f1:.4f}")
        print(f"\n  Per-class metrics:")
        print(f"  {'Class':<18} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}")
        print(f"  {'-'*58}")
        for i in range(self.num_classes):
            support = np.sum(self.all_labels == i)
            print(f"  {self.class_names[i]:<18} "
                  f"{self.precision[i]:>10.4f} "
                  f"{self.recall[i]:>10.4f} "
                  f"{self.f1[i]:>10.4f} "
                  f"{support:>8}")
        
        return {
            'accuracy': self.accuracy,
            'macro_f1': self.macro_f1,
            'per_class_f1': self.f1.tolist(),
        }
    
    def get_confusion_matrix(self, normalize=False, save_path=None):
        """
        可视化混淆矩阵
        """
        cm = self.confusion_mat
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt='.2f' if normalize else 'd',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cmap='Blues', ax=ax,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title(
            f'Confusion Matrix {"(Normalized)" if normalize else ""}\n'
            f'Accuracy: {self.accuracy:.3f}\n'
            f'Test Set: {len(self.all_labels)} samples',
            fontsize=14
        )
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"保存混淆矩阵: {save_path}")
        
        plt.show()
        return fig
    
    def analyze_errors(self, top_k=5):
        """
        分析最常见的错误预测
        """
        errors = self.all_preds != self.all_labels
        error_indices = np.where(errors)[0]
        
        # 统计每对 (true → predicted) 的错误次数
        error_pairs = {}
        for idx in error_indices:
            pair = (self.all_labels[idx], self.all_preds[idx])
            error_pairs[pair] = error_pairs.get(pair, 0) + 1
        
        sorted_errors = sorted(error_pairs.items(), key=lambda x: -x[1])
        
        print(f"\n=== Top {top_k} 常见错误 ===")
        for (true, pred), count in sorted_errors[:top_k]:
            print(f"  {self.class_names[true]:<18} → "
                  f"{self.class_names[pred]:<18} : {count} 次")
    
    def get_top_k_accuracy(self, k=3):
        """
        Top-K 准确率：正确类别在 top-K 预测中即算正确
        """
        top_k_preds = np.argsort(-self.all_probs, axis=1)[:, :k]
        correct = np.any(top_k_preds == self.all_labels[:, np.newaxis], axis=1)
        topk_acc = np.mean(correct)
        print(f"\n  Top-{k} Accuracy: {topk_acc:.4f}")
        return topk_acc


# ============================================================
# 主函数
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='最佳模型 checkpoint 路径')
    parser.add_argument('--data_dir', type=str, default='aist++/keypoints3d')
    parser.add_argument('--output_dir', type=str, default='outputs/eval')
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = ModelEvaluator(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
    )
    
    # 运行评估
    results = evaluator.evaluate()
    
    # 可视化
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 混淆矩阵
    evaluator.get_confusion_matrix(
        normalize=False,
        save_path=output_dir / "confusion_matrix.png"
    )
    evaluator.get_confusion_matrix(
        normalize=True,
        save_path=output_dir / "confusion_matrix_norm.png"
    )
    
    # 错误分析
    evaluator.analyze_errors(top_k=10)
    evaluator.get_top_k_accuracy(k=3)
    
    # 保存结果
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n评估结果保存到: {output_dir}")


if __name__ == "__main__":
    # 可直接运行（不依赖 argparse，方便调试）
    # 替换为你的实际 checkpoint 路径
    evaluator = ModelEvaluator(
        checkpoint_path="outputs/train_YYYYMMDD_HHMMSS/checkpoints/best_model.pt",
    )
    evaluator.evaluate()
    evaluator.get_confusion_matrix(normalize=True)
    evaluator.analyze_errors()
    evaluator.get_top_k_accuracy(k=3)
```

运行：
```bash
python evaluate.py --checkpoint outputs/train_XXXX/checkpoints/best_model.pt
```

---

## 7.2 关键指标解读

### Accuracy (准确率)
```
Accuracy = 正确预测数 / 总样本数
```
- 优点：直观
- 缺点：类别不平衡时会误导（90% 样本都是 class_A，全预测 class_A 也有 90% accuracy）

### Precision (精确率)
```
Precision(A) = 预测为A且真的为A / 所有预测为A
```
- "模型说它是 A，实际上真的是 A 的概率"

### Recall (召回率)
```
Recall(A) = 预测为A且真的为A / 所有真的为A
```
- "所有真正的 A，模型找到了多少"

### F1 Score
```
F1 = 2 * Precision * Recall / (Precision + Recall)
```
- Precision 和 Recall 的调和平均，是综合指标

### Macro vs Micro
| 类型 | 计算方式 | 用途 |
|------|----------|------|
| Macro | 每个类别独立计算，然后平均 | 每类同等重要 |
| Micro | 将所有类别汇总后计算 | 每个样本同等重要 |

---

## 7.3 模型校准 (Calibration)

检查模型的概率预测是否"诚实"：

```python
def plot_reliability_diagram(y_true, y_probs, n_bins=10):
    """
    可靠性图：预测置信度 vs 实际准确率
    理想情况下二者相等（对角线）
    """
    from sklearn.calibration import calibration_curve
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 对所有类别分别绘制
    for i, class_name in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        prob_true, prob_pred = calibration_curve(y_bin, y_probs[:, i], n_bins=n_bins)
        ax.plot(prob_pred, prob_true, marker='o', label=class_name)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('Reliability Diagram')
    ax.legend()
    plt.show()
```

---

## 检查要点

- [ ] 能在测试集上复现模型的完整评估
- [ ] 理解 Accuracy、Precision、Recall、F1 的含义
- [ ] 能从混淆矩阵中识别模型困惑的类别对
- [ ] 知道 Macro F1 和 Micro F1 的区别

## 常见问题

**Q: 测试集准确率远低于验证集怎么办？**
A: 说明数据集划分不当或有数据泄漏，检查 Step 05 的划分策略。

**Q: 某两类总是混淆（如 break 和 krump）怎么办？**
A: 这些类别动作相近，可以考虑：增加模型容量、添加注意力机制、或者使用更细粒度的动作特征。

**Q: 某个类别 recall 很低怎么办？**
A: 提高该类别在损失函数中的权重，或增加该类别的训练样本（数据增强）。

---

## 下一步

完成本步后，进入 [Step 08: 模型推理 & 可视化](step08_inference.md) 使用模型对新样本进行预测。
