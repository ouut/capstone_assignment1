# Step 10: 模型导出 (TorchScript / ONNX)

## 学习目标

- [ ] 理解模型导出的意义：脱离 PyTorch 环境运行
- [ ] 将 ST-GCN 导出为 TorchScript (.pt)
- [ ] 将 ST-GCN 导出为 ONNX (.onnx)
- [ ] 验证导出模型的正确性和性能
- [ ] 为 Docker 部署准备轻量化模型

## 参考文档

- [TorchScript 文档](https://pytorch.org/docs/stable/jit.html)
- [ONNX 文档](https://onnx.ai/)
- [ONNX Runtime 文档](https://onnxruntime.ai/)

---

## 10.1 TorchScript vs ONNX 对比

| 特性 | TorchScript | ONNX |
|------|-------------|------|
| 生态 | PyTorch 原生 | 跨框架通用 |
| 优化 | 图优化 | 图优化 + ONNX Runtime |
| 推理引擎 | libtorch (C++) | ONNX Runtime, TensorRT |
| 部署灵活性 | PyTorch 生态内 | 任意框架 |
| 动态控制流 | ✅ 支持 if/for | ⚠️ 有限支持 |
| 推荐场景 | C++ 服务（PyTorch 栈） | 跨框架 / 高性能推理 |

本教程两种都支持，推荐 ONNX（更通用）。

---

## 10.2 导出 TorchScript

创建文件 `export_torchscript.py`

```python
"""将 ST-GCN 导出为 TorchScript 格式"""
import torch
import torch.nn as nn
from pathlib import Path
import time
import numpy as np

from models.stgcn import STGCN


def export_torchscript(checkpoint_path, output_path, method='trace'):
    """
    导出 TorchScript 模型

    Args:
        checkpoint_path: 训练好的 PyTorch checkpoint
        output_path: 输出 .pt 文件路径
        method: 'trace' (通过示例输入追踪) 或 'script' (分析代码)
    """
    print(f"加载 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    config = checkpoint.get('config', {})
    model = STGCN(
        num_classes=config.get('num_classes', 10),
        base_channels=config.get('base_channels', 64),
        dropout=config.get('dropout', 0.5),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    if method == 'trace':
        # Trace: 用示例输入追踪模型的执行图
        # 优点：简单，通常就够用
        # 缺点：不能处理动态控制流
        example_input = torch.randn(1, 300, 17, 3)
        
        traced_model = torch.jit.trace(model, example_input)
        
    elif method == 'script':
        # Script: 分析 Python 代码转成 TorchScript
        # 优点：支持动态控制流
        # 缺点：可能不兼容某些 Python 语法
        traced_model = torch.jit.script(model)
    
    # 保存
    traced_model.save(output_path)
    print(f"TorchScript 模型已导出: {output_path}")
    print(f"  文件大小: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
    
    return traced_model


def verify_torchscript(model_path):
    """验证 TorchScript 模型"""
    print(f"\n验证 TorchScript 模型: {model_path}")
    
    # 加载
    model = torch.jit.load(model_path)
    model.eval()
    
    # 测试推理
    test_input = torch.randn(1, 300, 17, 3)
    
    # 原始 PyTorch 输出（需要在加载 TorchScript 之前有）
    with torch.no_grad():
        output = model(test_input)
    
    print(f"  输入 shape: {test_input.shape}")
    print(f"  输出 shape: {output.shape}")
    print(f"  输出: {output.argmax().item()}, conf: {output.softmax(1).max().item():.4f}")
    
    # 性能对比
    model.eval()
    n_warmup = 10
    n_test = 100
    
    # 预热
    for _ in range(n_warmup):
        model(test_input)
    
    # 计时
    start = time.perf_counter()
    for _ in range(n_test):
        model(test_input)
    elapsed = time.perf_counter() - start
    
    avg_ms = (elapsed / n_test) * 1000
    print(f"  平均推理时间: {avg_ms:.2f} ms")
    print(f"  ✓ 验证通过")
    
    return model


# ============================================================
# 10.3 导出 ONNX
# ============================================================
def export_onnx(checkpoint_path, output_path, opset_version=14):
    """
    导出 ONNX 模型

    ONNX (Open Neural Network Exchange) 是开放的模型交换格式，
    可以被 ONNX Runtime、TensorRT、OpenVINO 等运行时加载。
    """
    import torch.onnx
    
    print(f"加载 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    config = checkpoint.get('config', {})
    model = STGCN(
        num_classes=config.get('num_classes', 10),
        base_channels=config.get('base_channels', 64),
        dropout=config.get('dropout', 0.5),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 准备示例输入
    example_input = torch.randn(1, 300, 17, 3)
    
    # 动态轴：batch 和帧数可变
    dynamic_axes = {
        'input': {0: 'batch_size', 1: 'num_frames'},
        'output': {0: 'batch_size'},
    }
    
    # 导出
    torch.onnx.export(
        model,
        example_input,
        output_path,
        export_params=True,        # 保存训练好的权重
        opset_version=opset_version,
        do_constant_folding=True,  # 常量折叠优化
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
    )
    
    print(f"ONNX 模型已导出: {output_path}")
    print(f"  文件大小: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  opset 版本: {opset_version}")

    return output_path


def verify_onnx(model_path):
    """验证 ONNX 模型"""
    import onnx
    import onnxruntime as ort
    
    print(f"\n验证 ONNX 模型: {model_path}")
    
    # 1. 检查模型合法性
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    print("  ✓ ONNX 模型检查通过")
    
    # 2. 打印模型 IO
    print(f"  输入: {[i.name for i in onnx_model.graph.input]}")
    print(f"  输出: {[o.name for o in onnx_model.graph.output]}")
    
    # 3. 测试推理
    session = ort.InferenceSession(model_path)
    
    test_input = np.random.randn(1, 300, 17, 3).astype(np.float32)
    output = session.run(None, {'input': test_input})
    
    print(f"  输入 shape: {test_input.shape}")
    print(f"  输出 shape: {output[0].shape}")
    print(f"  预测类别: {output[0].argmax()}")
    
    # 4. 性能 benchmark
    n_warmup = 10
    n_test = 200
    
    for _ in range(n_warmup):
        session.run(None, {'input': test_input})
    
    start = time.perf_counter()
    for _ in range(n_test):
        session.run(None, {'input': test_input})
    elapsed = time.perf_counter() - start
    
    avg_ms = (elapsed / n_test) * 1000
    print(f"  平均推理时间: {avg_ms:.2f} ms (ONNX Runtime)")
    print(f"  ✓ 验证通过")
    
    return session


# ============================================================
# 10.4 精度验证：确保导出模型与原模型输出一致
# ============================================================
def validate_accuracy(checkpoint_path, onnx_path, num_samples=100):
    """
    对比原模型和 ONNX 模型的输出，确保精度无损
    """
    import onnxruntime as ort
    import torch
    
    # 加载原模型
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', {})
    torch_model = STGCN(
        num_classes=config.get('num_classes', 10),
        base_channels=config.get('base_channels', 64),
        dropout=config.get('dropout', 0.5),
    )
    torch_model.load_state_dict(checkpoint['model_state_dict'])
    torch_model.eval()
    
    # 加载 ONNX 模型
    ort_session = ort.InferenceSession(onnx_path)
    
    torch.manual_seed(42)
    diffs = []
    
    for i in range(num_samples):
        # 随机输入
        x = torch.randn(1, 300, 17, 3)
        
        with torch.no_grad():
            torch_out = torch_model(x).numpy()
        
        ort_out = ort_session.run(None, {'input': x.numpy().astype(np.float32)})[0]
        
        # 最大绝对误差
        diff = np.max(np.abs(torch_out - ort_out))
        diffs.append(diff)
    
    max_diff = np.max(diffs)
    mean_diff = np.mean(diffs)
    
    print(f"\n精度验证 ({num_samples} 个样本):")
    print(f"  最大绝对误差: {max_diff:.2e}")
    print(f"  平均绝对误差: {mean_diff:.2e}")
    
    if max_diff < 1e-4:
        print(f"  ✓ 精度一致 (误差 < 1e-4)")
    else:
        print(f"  ⚠ 存在精度差异，请检查导出设置")
    
    return max_diff


# ============================================================
# 主函数：一键导出
# ============================================================
def export_all(checkpoint_path, output_dir="outputs/export"):
    """一键导出所有格式"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("开始模型导出")
    print("="*60)
    
    # 1. TorchScript
    ts_path = output_dir / "stgcn_scripted.pt"
    export_torchscript(checkpoint_path, ts_path, method='trace')
    verify_torchscript(ts_path)
    
    # 2. ONNX
    onnx_path = output_dir / "stgcn.onnx"
    export_onnx(checkpoint_path, onnx_path)
    verify_onnx(onnx_path)
    
    # 3. 精度验证
    validate_accuracy(checkpoint_path, onnx_path)
    
    print(f"\n所有模型已导出到: {output_dir}")
    print(f"  TorchScript: {ts_path}")
    print(f"  ONNX:        {onnx_path}")


if __name__ == "__main__":
    # 替换为你的 checkpoint 路径
    checkpoint = "outputs/train_XXXX/checkpoints/best_model.pt"
    export_all(checkpoint)
```

运行：
```bash
# 安装依赖
pip install onnx onnxruntime

# 导出模型
python export_torchscript.py
```

---

## 10.5 ONNX 图优化

```python
def optimize_onnx(input_path, output_path):
    """
    使用 ONNX 内置优化器提升推理速度
    
    优化包括：
    - 常量折叠
    - 算子融合 (如 Conv+BN 合并)
    - 冗余节点消除
    """
    import onnx
    from onnx import optimizer
    
    model = onnx.load(input_path)
    
    # 应用所有优化
    passes = optimizer.get_available_passes()
    optimized = optimizer.optimize(model, passes)
    
    onnx.save(optimized, output_path)
    
    size_before = Path(input_path).stat().st_size / 1024 / 1024
    size_after = Path(output_path).stat().st_size / 1024 / 1024
    print(f"优化前: {size_before:.2f} MB, 优化后: {size_after:.2f} MB")
    
    return output_path
```

---

## 10.6 推理速度对比 (Benchmark)

```python
def benchmark_all(checkpoint_path):
    """
    对比原始 PyTorch、TorchScript、ONNX Runtime 的推理速度
    """
    import onnxruntime as ort
    
    results = {}
    
    # === 原始 PyTorch ===
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint.get('config', {})
    torch_model = STGCN(
        num_classes=config.get('num_classes', 10),
        base_channels=config.get('base_channels', 64),
        dropout=config.get('dropout', 0.5),
    )
    torch_model.load_state_dict(checkpoint['model_state_dict'])
    torch_model.eval()
    
    x = torch.randn(1, 300, 17, 3)
    
    # 预热
    for _ in range(20):
        with torch.no_grad():
            torch_model(x)
    
    t0 = time.perf_counter()
    for _ in range(200):
        with torch.no_grad():
            torch_model(x)
    results['PyTorch'] = (time.perf_counter() - t0) / 200 * 1000
    
    # === TorchScript ===
    ts_model = torch.jit.load("outputs/export/stgcn_scripted.pt")
    ts_model.eval()
    
    for _ in range(20):
        ts_model(x)
    
    t0 = time.perf_counter()
    for _ in range(200):
        ts_model(x)
    results['TorchScript'] = (time.perf_counter() - t0) / 200 * 1000
    
    # === ONNX Runtime ===
    sess = ort.InferenceSession("outputs/export/stgcn.onnx")
    xn = x.numpy().astype(np.float32)
    
    for _ in range(20):
        sess.run(None, {'input': xn})
    
    t0 = time.perf_counter()
    for _ in range(200):
        sess.run(None, {'input': xn})
    results['ONNX Runtime'] = (time.perf_counter() - t0) / 200 * 1000
    
    print("\n=== 推理速度对比 (CPU) ===")
    for name, ms in results.items():
        print(f"  {name:15s}: {ms:.2f} ms/sample")
    
    # 速度提升
    baseline = results['PyTorch']
    for name, ms in results.items():
        if name != 'PyTorch':
            speedup = baseline / ms
            print(f"  {name} vs PyTorch: {speedup:.2f}x")
```

---

## 检查要点

- [ ] 成功导出 TorchScript 和 ONNX 两种格式
- [ ] ONNX 模型通过 `onnx.checker` 验证
- [ ] 导出模型的输出与原模型一致（误差 < 1e-4）
- [ ] 理解 TorchScript trace vs script 的区别
- [ ] 动态轴设置正确（支持不同 batch size 和帧数）

## 常见问题

**Q: ONNX 导出报错 "Unsupported operator"？**
A: 确保使用 opset 14 或更高版本；某些自定义操作可能不支持，需要换用等效的标准操作。

**Q: TorchScript trace 报错？**
A: 尝试 `script` 方法；或者是模型中使用了不被 TorchScript 支持的 Python 语法。

**Q: 导出后精度下降了很多？**
A: ONNX 默认使用 FP32，不会有精度损失。检查是否误用了 FP16 模式；对比时确保输入完全一致。

---

## 下一步

完成本步后，进入 [Step 11: Docker 容器化 & HTTP API](step11_docker_api.md) 构建可部署的服务。
