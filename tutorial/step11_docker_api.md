# Step 11: Docker 容器化 & HTTP API

## 学习目标

- [ ] 使用 FastAPI 构建 RESTful 推理 API
- [ ] 编写 Dockerfile 将服务容器化
- [ ] 理解 API 设计最佳实践（输入验证、错误处理、健康检查）
- [ ] 测试 API 端点和性能

## 参考文档

- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [Docker 文档](https://docs.docker.com/)
- [ONNX Runtime 部署指南](https://onnxruntime.ai/docs/)

---

## 11.1 API 服务代码

创建文件 `api/app.py`

```python
"""
ST-GCN 动作识别 HTTP API 服务

endpoints:
  GET  /health          - 健康检查
  POST /predict         - 单样本推理
  POST /predict_batch   - 批量推理
  GET  /info            - 模型信息
"""
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import time
import os

# ============================================================
# 应用初始化
# ============================================================
app = FastAPI(
    title="ST-GCN Action Recognition API",
    description="AIST++ 舞蹈动作识别服务",
    version="1.0.0",
)

# 全局变量（在 startup 事件中初始化）
session: ort.InferenceSession = None
model_info: Dict = {}

# 类别名称
CLASS_NAMES = [
    'break', 'pop', 'lock', 'middle_hiphop', 'la_hiphop',
    'house', 'waack', 'krump', 'street_jazz', 'ballet_jazz'
]

TARGET_FRAMES = 300
NUM_JOINTS = 17
NUM_COORDS = 3


# ============================================================
# Pydantic Models (请求/响应 schema)
# ============================================================
class KeypointsInput(BaseModel):
    """
    单样本输入：3D 骨骼关键点
    """
    keypoints: List[List[List[float]]] = Field(
        ...,
        description="3D keypoints array of shape (T, 17, 3)",
        example=[[[0.1, 0.2, 0.3] for _ in range(17)] for _ in range(300)]
    )
    
    class Config:
        schema_extra = {
            "example": {
                "keypoints": [[[0.0, 0.0, 0.0] for _ in range(17)] for _ in range(300)]
            }
        }


class BatchKeypointsInput(BaseModel):
    """批量输入"""
    samples: List[KeypointsInput] = Field(
        ..., min_items=1, max_items=256,
        description="批量骨骼数据，最多 256 个样本"
    )


class PredictionResult(BaseModel):
    class_name: str
    class_id: int
    confidence: float
    probabilities: Dict[str, float]
    inference_time_ms: float


class BatchPredictionResult(BaseModel):
    predictions: List[PredictionResult]
    total_time_ms: float
    avg_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float


class ModelInfo(BaseModel):
    model_format: str
    num_classes: int
    target_frames: int
    num_joints: int
    class_names: List[str]


# ============================================================
# 预处理（与训练时完全一致）
# ============================================================
def preprocess(keypoints: np.ndarray) -> np.ndarray:
    """
    预处理骨骼数据
    keypoints: (T, 17, 3) → (1, 300, 17, 3)
    """
    T = keypoints.shape[0]
    
    # 插值到固定帧数
    old_idx = np.linspace(0, T - 1, T)
    new_idx = np.linspace(0, T - 1, TARGET_FRAMES)
    
    interpolated = np.zeros((TARGET_FRAMES, NUM_JOINTS, NUM_COORDS), dtype=np.float32)
    for j in range(NUM_JOINTS):
        for c in range(NUM_COORDS):
            interpolated[:, j, c] = np.interp(new_idx, old_idx, keypoints[:, j, c])
    
    # 归一化（髋部中心 + 标准差缩放）
    hip_center = (interpolated[:, 11, :] + interpolated[:, 12, :]) / 2
    centered = interpolated - hip_center[:, np.newaxis, :]
    std = np.std(centered)
    if std < 1e-6:
        std = 1.0
    normalized = centered / std
    
    # 添加 batch 维度
    return normalized[np.newaxis, ...]  # (1, 300, 17, 3)


# ============================================================
# 生命周期事件
# ============================================================
@app.on_event("startup")
async def load_model():
    """服务启动时加载 ONNX 模型"""
    global session, model_info
    
    model_path = os.environ.get("MODEL_PATH", "outputs/export/stgcn.onnx")
    
    if not os.path.exists(model_path):
        print(f"⚠ 模型文件不存在: {model_path}")
        print(f"  请确保已运行 Step 10 导出模型")
        return
    
    session = ort.InferenceSession(model_path)
    
    model_info = {
        "model_format": "ONNX",
        "num_classes": len(CLASS_NAMES),
        "target_frames": TARGET_FRAMES,
        "num_joints": NUM_JOINTS,
        "class_names": CLASS_NAMES,
    }
    
    print(f"✓ 模型已加载: {model_path}")
    print(f"  ONNX Runtime 版本: {ort.__version__}")


# ============================================================
# API Endpoints
# ============================================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点（Kubernetes / Cloud Run 会定期访问）"""
    return HealthResponse(
        status="healthy" if session is not None else "model_not_loaded",
        model_loaded=session is not None,
        uptime_seconds=time.time() - app.state.start_time,
    )


@app.get("/info", response_model=ModelInfo)
async def get_model_info():
    """获取模型信息"""
    return ModelInfo(**model_info)


@app.post("/predict", response_model=PredictionResult)
async def predict(input_data: KeypointsInput):
    """
    单样本动作识别

    输入: keypoints 数组 (T, 17, 3)
    输出: 预测类别和概率
    """
    if session is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 解析输入
        kp = np.array(input_data.keypoints, dtype=np.float32)
        
        if kp.ndim != 3 or kp.shape[1] != 17 or kp.shape[2] != 3:
            raise ValueError(
                f"输入形状应为 (T, 17, 3)，实际为 {kp.shape}"
            )
        
        if kp.shape[0] < 10:
            raise ValueError(f"帧数太少，至少需要 10 帧，实际 {kp.shape[0]} 帧")
        
        # 预处理
        kp_processed = preprocess(kp)
        
        # 推理
        t0 = time.perf_counter()
        output = session.run(None, {'input': kp_processed})[0]  # (1, 10)
        inference_time = (time.perf_counter() - t0) * 1000
        
        # 后处理
        probs = _softmax(output[0])
        pred_idx = int(np.argmax(probs))
        
        return PredictionResult(
            class_name=CLASS_NAMES[pred_idx],
            class_id=pred_idx,
            confidence=float(probs[pred_idx]),
            probabilities={
                CLASS_NAMES[i]: float(probs[i])
                for i in range(len(CLASS_NAMES))
            },
            inference_time_ms=round(inference_time, 2),
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理错误: {str(e)}")


@app.post("/predict_batch", response_model=BatchPredictionResult)
async def predict_batch(input_data: BatchKeypointsInput):
    """
    批量推理（最多 256 个样本）
    """
    if session is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        samples = []
        for sample in input_data.samples:
            kp = np.array(sample.keypoints, dtype=np.float32)
            samples.append(preprocess(kp))
        
        batch_input = np.concatenate(samples, axis=0)  # (B, 300, 17, 3)
        
        # 批量推理
        t0 = time.perf_counter()
        outputs = session.run(None, {'input': batch_input})[0]
        total_time = (time.perf_counter() - t0) * 1000
        
        predictions = []
        for output in outputs:
            probs = _softmax(output)
            pred_idx = int(np.argmax(probs))
            predictions.append(PredictionResult(
                class_name=CLASS_NAMES[pred_idx],
                class_id=pred_idx,
                confidence=float(probs[pred_idx]),
                probabilities={
                    CLASS_NAMES[i]: float(probs[i])
                    for i in range(len(CLASS_NAMES))
                },
                inference_time_ms=round(total_time / len(samples), 2),
            ))
        
        return BatchPredictionResult(
            predictions=predictions,
            total_time_ms=round(total_time, 2),
            avg_time_ms=round(total_time / len(samples), 2),
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理错误: {str(e)}")


# ============================================================
# 工具函数
# ============================================================
def _softmax(x):
    """稳定的 softmax"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


@app.on_event("startup")
async def set_start_time():
    app.state.start_time = time.time()
```

---

## 11.2 Dockerfile

创建文件 `Dockerfile`

```dockerfile
# ============================================================
# ST-GCN 推理服务 Dockerfile
# ============================================================

# ---- Stage 1: 轻量 Python 基础镜像 ----
FROM python:3.10-slim AS builder

# 安装 ONNX Runtime (CPU 版)
# ONNX Runtime GPU 版安装: onnxruntime-gpu
RUN pip install --no-cache-dir \
    onnxruntime==1.16.0 \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.0 \
    numpy==1.24.3

# ---- Stage 2: 运行时镜像 ----
FROM python:3.10-slim

WORKDIR /app

# 从构建阶段复制已安装的包
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# 复制应用代码
COPY api/app.py api/
COPY outputs/export/stgcn.onnx models/

# 非 root 用户运行
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 环境变量
ENV MODEL_PATH=/app/models/stgcn.onnx
ENV PORT=8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

EXPOSE 8080

# 启动服务
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
```

---

## 11.3 docker-compose.yml

创建文件 `docker-compose.yml`

```yaml
version: '3.8'

services:
  stgcn-api:
    build:
      context: .
      dockerfile: Dockerfile
    image: stgcn-action-recognition:latest
    container_name: stgcn-api
    ports:
      - "8080:8080"
    environment:
      - MODEL_PATH=/app/models/stgcn.onnx
      - PORT=8080
    volumes:
      # 挂载模型文件（开发时方便更新）
      - ./outputs/export/stgcn.onnx:/app/models/stgcn.onnx:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 5s
      retries: 3
```

---

## 11.4 构建和运行

```bash
# 1. 确保模型已导出
python export_torchscript.py

# 2. 构建 Docker 镜像
docker build -t stgcn-action-recognition:latest .

# 3. 运行容器
docker run -d \
  --name stgcn-api \
  -p 8080:8080 \
  -v $(pwd)/outputs/export/stgcn.onnx:/app/models/stgcn.onnx:ro \
  stgcn-action-recognition:latest

# 或使用 docker-compose
docker-compose up -d

# 4. 验证服务
curl http://localhost:8080/health
```

---

## 11.5 API 测试

```bash
# 测试健康检查
curl http://localhost:8080/health

# 测试模型信息
curl http://localhost:8080/info

# 测试推理（发送一个真实的 pkl 文件数据）
python -c "
import pickle, json, requests
import numpy as np

# 加载一个样本
with open('aist++/keypoints3d/gWA_sBM_cAll_d25_mWA0_ch04.pkl', 'rb') as f:
    data = pickle.load(f)
kp = data['keypoints3d_optim'][:100].tolist()  # 取100帧

resp = requests.post('http://localhost:8080/predict', json={'keypoints': kp})
print(json.dumps(resp.json(), indent=2))
"

# 压测 (autocannon)
npx autocannon -c 10 -d 30 http://localhost:8080/predict \
  -m POST -H 'Content-Type: application/json' \
  -b '{"keypoints": [[[0.1,0.2,0.3] for _ in range(17)] for _ in range(300)]}'
```

---

## 11.6 API 设计要点

| 设计元素 | 实现 |
|---------|------|
| **输入验证** | Pydantic schema 自动验证形状和类型 |
| **错误处理** | 有意义的 HTTP 状态码 (400/500/503) |
| **健康检查** | `/health` 端点，Docker / K8s 可用 |
| **无状态** | 每次 `/predict` 独立，无共享状态 |
| **批量支持** | `/predict_batch` 减少网络开销 |
| **安全** | 非 root 用户运行；输入大小限制 (max 256) |

---

## 检查要点

- [ ] Docker 镜像构建成功，大小合理（<500MB）
- [ ] `/health` 返回 200
- [ ] `/predict` 返回正确的预测结果
- [ ] API 能处理错误输入（太短的序列、错误的形状等）
- [ ] 批量预测时间小于单样本预测时间 × batch size

## 常见问题

**Q: Docker 镜像太大？**
A: 使用 `python:3.10-slim` 基础镜像；确定不需要的包不要装；使用多阶段构建。

**Q: API 并发性能不好？**
A: 增加 `--workers` 参数；使用 gunicorn 作为进程管理；部署多个副本。

**Q: 模型加载很慢？**
A: 在 `startup` 事件中加载，避免每次请求时加载；使用 ONNX Runtime 而非 PyTorch。

---

## 下一步

完成本步后，进入 [Step 12: GCP 部署](step12_gcp_deploy.md) 将服务部署到 Google Cloud Platform。
