# Step 08: 模型测试 & 推理

## 学习目标

- [ ] 加载训练好的模型进行单样本推理
- [ ] 处理原始输入（不同帧数）的正确流程
- [ ] 可视化推理结果（动作序列 + 预测标签）
- [ ] 批量推理以提高吞吐量
- [ ] 暴露推理接口为后续 API 做准备

## 参考文档

- [PyTorch Model Inference](https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html)

---

## 8.1 推理引擎

创建文件 `inference.py`

```python
"""ST-GCN 推理引擎"""
import torch
import torch.nn as nn
import pickle
import numpy as np
from pathlib import Path
from typing import List, Union, Dict, Optional

from models.stgcn import STGCN
from data.dataset import (
    preprocess_sample, GENRE_NAMES,
    extract_genre_from_filename
)


# 标签映射（bidirectional）
IDX_TO_GENRE = {i: name for i, name in enumerate([
    'break', 'pop', 'lock', 'middle_hiphop', 'la_hiphop',
    'house', 'waack', 'krump', 'street_jazz', 'ballet_jazz'
])}
GENRE_TO_IDX = {v: k for k, v in IDX_TO_GENRE.items()}


class STGCNInference:
    """
    ST-GCN 推理引擎

    使用方式:
        engine = STGCNInference("path/to/best_model.pt")
        result = engine.predict("path/to/sample.pkl")
        print(result['class_name'], result['confidence'])
    """
    
    def __init__(self, checkpoint_path: str, device: str = None):
        """
        Args:
            checkpoint_path: 训练好的模型路径
            device: 'cuda', 'cpu', 或 None (自动选择)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载 checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        self.num_classes = config.get('num_classes', 10)
        self.target_frames = config.get('target_frames', 300)
        
        # 创建模型
        self.model = STGCN(
            num_classes=self.num_classes,
            base_channels=config.get('base_channels', 64),
            dropout=config.get('dropout', 0.5),
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"推理引擎已就绪: device={self.device}, "
              f"target_frames={self.target_frames}")
    
    @torch.no_grad()
    def predict(self, keypoints: np.ndarray, top_k: int = 3) -> Dict:
        """
        对单样本进行推理

        Args:
            keypoints: (T, 17, 3) 3D 骨骼关键点，T 可以是任意值
            top_k: 返回 top-K 预测

        Returns:
            {
                'class_id': int,
                'class_name': str,
                'confidence': float,
                'probabilities': list of (name, prob),
                'top_k': list of (name, prob),
            }
        """
        # 预处理
        tensor = preprocess_sample(
            keypoints,
            target_frames=self.target_frames,
            augment=False,
        ).unsqueeze(0).to(self.device)  # (1, T, V, 3)
        
        # 推理
        logits = self.model(tensor)  # (1, num_classes)
        probs = torch.softmax(logits, dim=1).squeeze(0)  # (num_classes,)
        
        # 获取预测
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()
        
        # Top-K
        topk_probs, topk_indices = torch.topk(probs, top_k)
        top_k_results = [
            (IDX_TO_GENRE[idx.item()], prob.item())
            for idx, prob in zip(topk_indices, topk_probs)
        ]
        
        # 所有类别的概率
        all_probs = [
            (IDX_TO_GENRE[i], probs[i].item())
            for i in range(self.num_classes)
        ]
        
        return {
            'class_id': pred_idx,
            'class_name': IDX_TO_GENRE[pred_idx],
            'confidence': confidence,
            'probabilities': all_probs,
            'top_k': top_k_results,
        }
    
    @torch.no_grad()
    def predict_batch(self, keypoints_list: List[np.ndarray],
                      batch_size: int = 32) -> List[Dict]:
        """
        批量推理

        Args:
            keypoints_list: list of (T, 17, 3) arrays
            batch_size: 批大小

        Returns:
            list of prediction dicts
        """
        results = []
        
        for i in range(0, len(keypoints_list), batch_size):
            batch_kps = keypoints_list[i:i + batch_size]
            
            # 预处理所有样本
            tensors = []
            for kp in batch_kps:
                t = preprocess_sample(kp, self.target_frames, augment=False)
                tensors.append(t)
            
            batch_tensor = torch.stack(tensors, dim=0).to(self.device)  # (B, T, V, 3)
            
            # 推理
            logits = self.model(batch_tensor)
            probs = torch.softmax(logits, dim=1)  # (B, num_classes)
            preds = probs.argmax(dim=1)  # (B,)
            confs = probs.gather(1, preds.unsqueeze(1)).squeeze(1)  # (B,)
            
            for j in range(probs.size(0)):
                results.append({
                    'class_id': preds[j].item(),
                    'class_name': IDX_TO_GENRE[preds[j].item()],
                    'confidence': confs[j].item(),
                })
        
        return results
    
    def predict_from_file(self, filepath: str) -> Dict:
        """直接从 pkl 文件推理"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        keypoints = data['keypoints3d_optim']
        
        result = self.predict(keypoints)
        # 附加文件名中的真实标签（如果有）
        genre = extract_genre_from_filename(Path(filepath).name)
        if genre:
            genre_name = GENRE_NAMES.get(genre, genre)
            result['filename_genre'] = genre_name
        
        return result


# ============================================================
# 可视化推理结果
# ============================================================
def visualize_prediction(keypoints: np.ndarray, prediction: Dict,
                          true_label: str = None, max_frames: int = 100):
    """
    可视化骨骼序列 + 预测结果

    Args:
        keypoints: (T, 17, 3) numpy array
        prediction: predict() 的返回值
        true_label: 真实标签（可选）
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左侧：骨骼的关键帧
    ax1 = axes[0]
    # 取第1帧，画骨骼
    from data.dataset import normalize_keypoints
    kp_norm = normalize_keypoints(keypoints).copy()
    
    # COCO edges
    edges = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 6), (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    ]
    
    # 取几帧叠加
    frames_to_show = [0, 20, 40, 60, 80]
    colors = plt.cm.viridis(np.linspace(0, 1, len(frames_to_show)))
    
    for fi, (fidx, color) in enumerate(zip(frames_to_show, colors)):
        if fidx >= len(kp_norm):
            break
        kp = kp_norm[fidx]
        alpha = 1.0 - fi * 0.15
        for (i, j) in edges:
            ax1.plot([kp[i, 0], kp[j, 0]], [kp[i, 1], kp[j, 1]],
                     color=color, alpha=alpha, linewidth=1.5,
                     label=f'Frame {fidx}' if i == 0 and j == 1 else None)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Skeleton Key Frames (XY plane)')
    ax1.legend(fontsize=8)
    ax1.set_aspect('equal')
    
    # 右侧：预测概率柱状图
    ax2 = axes[1]
    names = [p[0][:8] for p in prediction['probabilities']]
    probs = [p[1] for p in prediction['probabilities']]
    bars = ax2.bar(names, probs, color='steelblue')
    
    # 高亮预测的类别
    pred_idx = prediction['class_id']
    if pred_idx < len(bars):
        bars[pred_idx].set_color('coral')
    
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Probability')
    ax2.set_title(f"Prediction: {prediction['class_name']} "
                  f"(conf: {prediction['confidence']:.3f})")
    if true_label:
        ax2.set_title(f"Pred: {prediction['class_name']} "
                      f"| True: {true_label} "
                      f"(conf: {prediction['confidence']:.3f})")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    # 初始化推理引擎
    engine = STGCNInference(
        checkpoint_path="outputs/train_XXXX/checkpoints/best_model.pt"
    )
    
    # 单文件推理
    sample_file = "aist++/keypoints3d/gWA_sBM_cAll_d25_mWA0_ch04.pkl"
    result = engine.predict_from_file(sample_file)
    
    print(f"\n文件: {sample_file}")
    print(f"预测: {result['class_name']} (conf: {result['confidence']:.3f})")
    print(f"文件名标签: {result.get('filename_genre', 'N/A')}")
    print(f"\nTop-3 预测:")
    for name, prob in result['top_k']:
        print(f"  {name:20s}: {prob:.3f}")
    
    # 可视化
    with open(sample_file, 'rb') as f:
        data = pickle.load(f)
    kp = data['keypoints3d_optim']
    visualize_prediction(kp, result, true_label=result.get('filename_genre'))
```

---

## 8.2 推理 Benchmark

```python
def benchmark_inference(engine, test_loader, num_samples=100):
    """测量推理速度和吞吐量"""
    import time
    
    # 预热
    dummy = torch.randn(1, 300, 17, 3).to(engine.device)
    for _ in range(10):
        engine.model(dummy)
    
    torch.cuda.synchronize() if engine.device == 'cuda' else None
    
    # 测试延迟 (单个样本)
    times = []
    for i, batch in enumerate(test_loader):
        if i >= num_samples:
            break
        x = batch['keypoints'][:1].to(engine.device)
        
        torch.cuda.synchronize() if engine.device == 'cuda' else None
        t0 = time.perf_counter()
        
        engine.model(x)
        
        torch.cuda.synchronize() if engine.device == 'cuda' else None
        t1 = time.perf_counter()
        
        times.append((t1 - t0) * 1000)  # ms
    
    avg_latency = np.mean(times)
    p95_latency = np.percentile(times, 95)
    
    print(f"\n推理 Benchmark:")
    print(f"  平均延迟: {avg_latency:.2f} ms")
    print(f"  P95 延迟: {p95_latency:.2f} ms")
    print(f"  吞吐量: {1000 / avg_latency:.1f} samples/sec")
    
    return avg_latency, p95_latency
```

---

## 8.3 推理接口设计原则

为了后续 Step 11 的 API 设计，推理接口应满足：

1. **无状态**: 每次 `predict()` 独立，不依赖前一次调用
2. **灵活输入**: 接受不同帧数的数据，内部处理到 `target_frames`
3. **批量高效**: 支持 `predict_batch()` 提高吞吐
4. **返回结构化**: 总是返回 dict，便于 JSON 序列化

---

## 检查要点

- [ ] 能加载 checkpoint 并对新数据推理
- [ ] 预处理管道保持一致（归一化、帧数固定）
- [ ] 理解 `model.eval()` 和 `torch.no_grad()` 的作用
- [ ] 推理返回的是概率分布，不只是硬标签
- [ ] 单样本和批量推理结果一致

## 常见问题

**Q: 推理时用的预处理和训练时不一样怎么办？**
A: 务必保持完全一致！包括归一化方式、target_frames、插值方法。训练时 `augment=True`，推理时 `augment=False`。

**Q: 推理速度慢怎么办？**
A: 使用 GPU 推理、TorchScript 编译 (Step 10)、或使用 ONNX Runtime (Step 10)。

**Q: 为什么 inference 的结果和 validation 不一样？**
A: 检查 `model.eval()` 是否调用；检查预处理是否完全一致。

---

## 下一步

完成本步后，进入 [Step 09: 模型优化](step09_optimization.md) 提升模型性能。
