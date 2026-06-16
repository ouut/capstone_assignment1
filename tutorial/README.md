# ST-GCN on AIST++ Tutorial

## Learning Path: From Zero to GCP Deployment

本教程面向对 PyTorch 和 ST-GCN 不熟悉的开发者，采用自底向上的学习路径：

```
环境搭建 → 图卷积理论 → ST-GCN模型 → 数据加载 → 训练/测试 → 评估 → 推理 → 优化 → 导出 → Docker API → GCP部署
```

| Step | 主题 | 核心目标 | 预计时间 |
|------|------|----------|----------|
| [01](step01_env_and_data.md) | 环境搭建 & 数据探索 | 配置 PyTorch 环境，理解 AIST++ 数据格式 | 2h |
| [02](step02_graph_conv_theory.md) | 图卷积理论 | 从邻接矩阵到 GCN 的前向传播，手写实现 | 3h |
| [03](step03_stgcn_model.md) | ST-GCN 模型实现 | 完整 ST-GCN 模型代码，空间+时间图卷积 | 3h |
| [04](step04_data_pipeline.md) | 数据预处理流水线 | 骨骼数据归一化、图构建、序列处理 | 2h |
| [05](step05_dataloader.md) | 训练/测试集划分 & DataLoader | 数据集切分、批处理、数据增强 | 2h |
| [06](step06_training.md) | 模型训练循环 | 训练循环、损失函数、优化器、checkpoint | 2h |
| [07](step07_evaluation.md) | 模型评估 & 指标 | Accuracy、Confusion Matrix、per-class metrics | 2h |
| [08](step08_inference.md) | 模型测试 & 推理 | 单样本推理、批量推理、可视化 | 1.5h |
| [09](step09_optimization.md) | 模型优化 | 超参数搜索、学习率调度、模型剪枝 | 2h |
| [10](step10_export.md) | 模型导出 | TorchScript、ONNX 导出与验证 | 1.5h |
| [11](step11_docker_api.md) | Docker 容器化 & HTTP API | FastAPI 服务、Docker 镜像构建 | 3h |
| [12](step12_gcp_deploy.md) | GCP 部署 | Cloud Run 部署、自动扩缩容、监控 | 2h |

## 实验场景说明

本项目使用 **AIST++** 数据集中的 3D 人体骨骼关键点（COCO-17 格式），
使用 **ST-GCN (Spatial Temporal Graph Convolutional Network)** 进行动作识别任务。

数据维度: `(N_frames, 17_joints, 3_coords)` per sample

### 动作标签

AIST++ 包含 10 种舞蹈动作类别，根据文件名解析：
- `gBR` = break (霹雳舞)
- `gPO` = pop (机械舞)
- `gLO` = lock (锁舞)
- `gMH` = middle hip-hop
- `gLH` = LA style hip-hop
- `gHO` = house
- `gWA` = waack (甩手舞)
- `gKR` = krump (狂派舞)
- `gJS` = street jazz
- `gJB` = ballet jazz

## 前置知识

- Python 基础：类、函数、NumPy 数组操作
- 线性代数基础：矩阵乘法、邻接矩阵
- 了解 PyTorch 基本概念（tensor、nn.Module、autograd）

## 参考资源

- [ST-GCN 原论文](https://arxiv.org/abs/1801.07455)
- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [AIST++ 数据集](https://google.github.io/aichoreographer/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
