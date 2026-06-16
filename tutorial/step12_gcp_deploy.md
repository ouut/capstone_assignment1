# Step 12: GCP 部署 (Google Cloud Platform)

## 学习目标

- [ ] 理解 GCP Cloud Run 的 Serverless 部署模型
- [ ] 将 Docker 镜像推送到 Google Artifact Registry
- [ ] 部署到 Cloud Run 并配置自动扩缩容
- [ ] 设置 Cloud Monitoring 监控和告警
- [ ] 理解成本估算和优化

## 参考文档

- [Cloud Run 文档](https://cloud.google.com/run/docs)
- [Artifact Registry 文档](https://cloud.google.com/artifact-registry/docs)
- [Cloud Build 文档](https://cloud.google.com/build/docs)

---

## 12.1 架构总览

```
                    ┌─────────────┐
                    │  用户/客户端  │
                    └──────┬──────┘
                           │ HTTPS
                    ┌──────▼──────┐
                    │  Cloud CDN  │ (可选: 全球加速)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Cloud Run  │ ← Serverless, 自动扩缩容
                    │  (容器实例)  │
                    │  模型在容器内 │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──┐  ┌─────▼───┐  ┌────▼──────┐
       │ Cloud   │  │ Cloud   │  │ Error     │
       │ Logging │  │ Monitor │  │ Reporting │
       └─────────┘  └─────────┘  └───────────┘
```

## 12.2 GCP 部署策略对比

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **Cloud Run** ✅ | 无需管理服务器；自动扩缩；按请求计费 | 冷启动延迟（~1s）；内存上限 32GB | API 服务（推荐本教程） |
| **GKE (Kubernetes)** | 完全控制；适合微服务集群 | 管理复杂；成本高 | 大型项目，微服务架构 |
| **Compute Engine** | 完全控制 VM；固定成本 | 需要手动运维；不能自动扩缩 | 持续高负载 |
| **Vertex AI** | 专为 ML 设计；模型版本管理 | 学习成本高；与框架耦合 | 正式的 ML 平台 |

**本教程推荐 Cloud Run：最适合 API 服务，简单、便宜、自动扩缩。**

---

## 12.3 前置准备

### 1. 安装 Google Cloud CLI

```bash
# macOS
brew install --cask google-cloud-sdk

# 或从官网下载: https://cloud.google.com/sdk/docs/install

# 初始化
gcloud init
gcloud auth login
```

### 2. 创建 GCP 项目

```bash
# 创建项目
gcloud projects create stgcn-action-recognition --name="ST-GCN Action Recognition"

# 设置当前项目
gcloud config set project stgcn-action-recognition

# 启用必要的 API
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  monitoring.googleapis.com \
  logging.googleapis.com
```

### 3. 设置计费账户（如未设置）

在 [GCP Console](https://console.cloud.google.com/billing) 中关联计费账户。

> 💰 **成本估算**：Cloud Run 按请求数 + CPU/内存使用量计费。每月 200 万请求免费。一个 2GB 内存的实例处理 1000 次/天推理约 $5-15/月。

---

## 12.4 推送镜像到 Artifact Registry

```bash
# 1. 创建 Artifact Registry 仓库
gcloud artifacts repositories create stgcn-models \
  --repository-format=docker \
  --location=us-central1 \
  --description="ST-GCN model images"

# 2. 配置 Docker 认证
gcloud auth configure-docker us-central1-docker.pkg.dev

# 3. 构建镜像（使用 Cloud Build，不需要本地 Docker）
gcloud builds submit \
  --tag us-central1-docker.pkg.dev/stgcn-action-recognition/stgcn-models/stgcn-api:v1

# 或者本地构建再推送
docker build -t stgcn-api .
docker tag stgcn-api \
  us-central1-docker.pkg.dev/stgcn-action-recognition/stgcn-models/stgcn-api:v1
docker push \
  us-central1-docker.pkg.dev/stgcn-action-recognition/stgcn-models/stgcn-api:v1
```

---

## 12.5 部署到 Cloud Run

```bash
# 部署服务
gcloud run deploy stgcn-api \
  --image us-central1-docker.pkg.dev/stgcn-action-recognition/stgcn-models/stgcn-api:v1 \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 10 \
  --concurrency 80 \
  --timeout 300 \
  --allow-unauthenticated \
  --set-env-vars "MODEL_PATH=/app/models/stgcn.onnx"
```

### 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--memory` | 2Gi | 2GB 内存（模型 + ONNX Runtime 约需 1GB） |
| `--cpu` | 2 | 2 vCPU（提高推理速度） |
| `--min-instances` | 0 | 无请求时缩容到 0（省钱） |
| `--max-instances` | 10 | 最多 10 个实例（防止意外飙升） |
| `--concurrency` | 80 | 每个实例同时处理最多 80 个请求 |
| `--timeout` | 300 | 请求超时 5 分钟 |
| `--allow-unauthenticated` | - | 公开 API（不需要认证） |

部署成功后，会输出服务 URL：
```
Service URL: https://stgcn-api-xxxxx-uc.a.run.app
```

---

## 12.6 测试部署的服务

```bash
# 设置服务 URL
SERVICE_URL="https://stgcn-api-xxxxx-uc.a.run.app"

# 健康检查
curl $SERVICE_URL/health

# 推理测试
python -c "
import pickle, json, requests
import numpy as np

with open('aist++/keypoints3d/gWA_sBM_cAll_d25_mWA0_ch04.pkl', 'rb') as f:
    data = pickle.load(f)
kp = data['keypoints3d_optim'][:100].tolist()

resp = requests.post(
    '$SERVICE_URL/predict',
    json={'keypoints': kp},
    timeout=30
)
print(json.dumps(resp.json(), indent=2))
"
```

---

## 12.7 监控和告警

### Cloud Monitoring Dashboard

GCP 自动为 Cloud Run 提供以下指标：
- **请求数/秒** (Request count)
- **请求延迟** (Request latency) - P50, P95, P99
- **容器实例数** (Container instance count)
- **CPU 利用率** (CPU utilization)
- **内存使用率** (Memory utilization)
- **错误率** (4xx/5xx - Error count)

在 [Cloud Monitoring](https://console.cloud.google.com/monitoring) 中查看。

### 设置告警

```bash
# 创建告警策略: 错误率 > 5%
gcloud monitoring policies create \
  --display-name="ST-GCN High Error Rate" \
  --condition="metric=run.googleapis.com/request_count AND resource.type=cloud_run_revision AND metric.label.response_code_class=5xx" \
  --condition-threshold-value=0.05 \
  --condition-threshold-duration=300s \
  --notification-channels="email:your-email@example.com"
```

### 自定义日志

在 FastAPI 中添加日志（在 `api/app.py` 中）：

```python
import logging
import google.cloud.logging

# 初始化 Cloud Logging
client = google.cloud.logging.Client()
client.setup_logging()

@app.middleware("http")
async def log_requests(request, call_next):
    """记录每个请求的日志"""
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logging.info(f"{request.method} {request.url.path} {response.status_code} {duration:.3f}s")
    return response
```

---

## 12.8 持续部署 (CI/CD)

### 使用 Cloud Build 自动化

创建文件 `cloudbuild.yaml`

```yaml
steps:
  # Step 1: 构建 Docker 镜像
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-t'
      - 'us-central1-docker.pkg.dev/$PROJECT_ID/stgcn-models/stgcn-api:$COMMIT_SHA'
      - '-t'
      - 'us-central1-docker.pkg.dev/$PROJECT_ID/stgcn-models/stgcn-api:latest'
      - '.'

  # Step 2: 推送镜像
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'push'
      - 'us-central1-docker.pkg.dev/$PROJECT_ID/stgcn-models/stgcn-api:$COMMIT_SHA'

  # Step 3: 部署到 Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'stgcn-api'
      - '--image'
      - 'us-central1-docker.pkg.dev/$PROJECT_ID/stgcn-models/stgcn-api:$COMMIT_SHA'
      - '--region'
      - 'us-central1'
      - '--platform'
      - 'managed'
      - '--memory'
      - '2Gi'
      - '--cpu'
      - '2'

images:
  - 'us-central1-docker.pkg.dev/$PROJECT_ID/stgcn-models/stgcn-api:$COMMIT_SHA'
```

### 连接 GitHub 触发器

```bash
# 创建 Cloud Build 触发器（通过 Console 或 CLI）
gcloud builds triggers create github \
  --name="stgcn-deploy" \
  --repo-name="stgcn-action-recognition" \
  --branch-pattern="^main$" \
  --build-config="cloudbuild.yaml"
```

现在每次 push 到 `main` 分支，GCP 会自动构建并部署。

---

## 12.9 生产环境增强

### 1. 认证保护 API

```bash
# 移除公开访问，改为需要认证
gcloud run deploy stgcn-api \
  --no-allow-unauthenticated \
  ...

# 客户端使用 service account 调用
gcloud auth print-identity-token | \
  curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  $SERVICE_URL/predict
```

### 2. 自定义域名

```bash
# 映射自定义域名
gcloud beta run domain-mappings create \
  --service stgcn-api \
  --domain api.yourcompany.com \
  --region us-central1
```

### 3. 降低冷启动时间

- 设置 `--min-instances 1`（始终保留一个实例，月费约 $30）
- 使用更小的模型（Pruning / Quantization）
- 使用 `python:3.10-slim` 基础镜像（最小化镜像大小）
- 把模型放在容器内（而非从外部加载）

### 4. 模型预热

在 `Dockerfile` 中添加预热脚本：

```dockerfile
# 在 ENTRYPOINT 前做一次预热推理
RUN python -c "
import numpy as np
import onnxruntime as ort
session = ort.InferenceSession('/app/models/stgcn.onnx')
dummy = np.random.randn(1, 300, 17, 3).astype(np.float32)
session.run(None, {'input': dummy})
print('Model warmup complete')
"
```

---

## 12.10 成本优化

| 层级 | 配置 | 月费估算 |
|------|------|---------|
| 开发/测试 | min=0, 512MB, 1CPU | $0-5 |
| 小规模生产 | min=0, 1GB, 1CPU | $5-20 |
| 中规模生产 | min=1, 2GB, 2CPU | $30-50 |
| 大规模生产 | min=3, 4GB, 4CPU | $100+ |

省钱技巧：
- `--min-instances 0` = 无流量时不花钱
- `--concurrency` 设大一些（默认 80），提高单实例利用率
- 使用 `--cpu-throttling` 减少闲置 CPU 计费

---

## 12.11 完整部署检查清单

- [ ] GCP 项目已创建，API 已启用
- [ ] Docker 镜像已推送到 Artifact Registry
- [ ] Cloud Run 服务已部署，健康检查通过
- [ ] API 端点测试通过（单样本 + 批量）
- [ ] 监控仪表盘可访问
- [ ] 告警策略已配置
- [ ] CI/CD Pipeline (Cloud Build) 已配置
- [ ] 成本预算告警已设置（防止意外账单）

```bash
# 一键检查脚本
SERVICE_URL=$(gcloud run services describe stgcn-api --region us-central1 --format='value(status.url)')
echo "服务 URL: $SERVICE_URL"

# 健康检查
curl -s $SERVICE_URL/health | python -m json.tool

# 检查实例数
gcloud run services describe stgcn-api --region us-central1 --format='value(status.latestReadyRevisionName)'
```

---

## 检查要点

- [ ] 理解 Cloud Run 的 Serverless 模型（无需管理服务器）
- [ ] Docker 镜像成功推送到 Artifact Registry
- [ ] 服务部署并可通过 HTTPS 访问
- [ ] 理解 `min-instances` 对冷启动的影响
- [ ] 知道如何查看 Cloud Monitoring 指标

## 常见问题

**Q: 第一次请求很慢（冷启动）？**
A: 设置 `--min-instances 1` 保持一个热实例；或减小镜像大小加速启动。

**Q: 内存不足（OOM）？**
A: 增加 `--memory`；或使用 ONNX Runtime 的优化选项减少内存占用。

**Q: 如何限制 API 调用频率？**
A: 使用 Cloud Endpoints / Apigee；或自建 API Gateway；或在代码中加限流中间件。

**Q: GPU 推理怎么办？**
A: Cloud Run 不支持 GPU。如果必须用 GPU 推理，考虑使用 GKE + GPU 节点 或 Vertex AI Prediction。

---

## 总结

恭喜！你已完成整个 ST-GCN on AIST++ 的学习路线：

```
✅ Step 01: 环境搭建 & 数据探索
✅ Step 02: 图卷积理论 (从零手写 GCN)
✅ Step 03: ST-GCN 模型实现 (PyTorch)
✅ Step 04: 数据预处理流水线
✅ Step 05: 训练/测试集划分 & DataLoader
✅ Step 06: 模型训练循环
✅ Step 07: 模型评估 & 指标分析
✅ Step 08: 模型推理 & 可视化
✅ Step 09: 模型优化 & 超参数调优
✅ Step 10: 模型导出 (TorchScript / ONNX)
✅ Step 11: Docker 容器化 & HTTP API
✅ Step 12: GCP 部署 (Cloud Run)
```

现在你掌握了从数据处理到生产部署的完整 ML 流水线。🎉
