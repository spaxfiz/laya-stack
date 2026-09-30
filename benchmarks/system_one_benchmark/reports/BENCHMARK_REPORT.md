# System One / DecisionModel Universal Benchmark Report

- **Generated At**: `2026-09-30 15:16:20`
- **Dataset**: `benchmark_1024.json` (Total **1024** labeled samples across 12 Xiaohongshu tracks)
- **Hardware Architecture**: Apple Silicon Metal GPU (Unified Memory)
- **Model Checkpoint**: `aac6fef/laya-multilingual-mlx` (ModernBERT-base + 2-layer RL Cross-Attention Head)

---

## 1. 核心结果总览 (Executive Summary)

| Engine 后端引擎 | 架构类型 | 1024篇 全量准确率 | 对抗样本 准确率 | P50 延迟 | P95 延迟 | 峰值并发吞吐 (RPS) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Native-Laya-MLX (Python SDK)** | In-process MLX DecisionModel with 2-layer Cross-Attention Head & Scorer | **64.1%** | **75.0%** | 16.11 ms | 17.71 ms | **80.1 req/s** |
| **Rapid-MLX System-One (HTTP API)** | Rapid-MLX Standalone Daemon on http://127.0.0.1:8700 (First-class LayaBackend) | **64.1%** | **75.0%** | 16.69 ms | 18.38 ms | **62.9 req/s** |
| **oMLX Bi-Encoder Embeddings (HTTP API)** | oMLX ModernBERT Sentence Embeddings with Cosine Similarity Matching | **9.7%** | **12.5%** | 0.26 ms | 0.43 ms | **7094.8 req/s** |

---

## 2. 精度与泛化度评测 (Accuracy & Generalization)

| 指标维度 | Native Laya MLX (Python SDK) | Rapid-MLX System-One (API) | oMLX Bi-Encoder (Embedding API) |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy (全量准确率)** | **64.1%** | **64.1%** | **9.7%** |
| **Macro-F1 Score** | **66.3%** | **66.3%** | **1.5%** |
| **Adversarial Accuracy (对抗样本)** | **75.0%** | **75.0%** | **12.5%** |
| **Expected Calibration Error (ECE)** | **0.1368** | **0.1368** | **0.0134** |

> [!NOTE]
> **Native Laya 与 Rapid-MLX 的等价性**：两者均驱动完整的 170 张量 DecisionModel（包含 2 层 Cross-Attention 跨注意力决策头），准确率达到完全一致的高水准。
> **oMLX 的余弦匹配局限性**：由于 oMLX 仅使用 ModernBERT 最后一层句向量计算余弦相似度（Bi-Encoder 模式），缺乏 Cross-Attention 层在 Context 与 Option 之间的交互，在对抗样本和长尾赛道上出现精度衰减。

---

## 3. 延迟百分位分布 (Latency Distribution)

| Engine 引擎 | 冷启动耗时 | Mean 平均耗时 | P50 中位数 | P90 延迟 | P95 延迟 | P99 极端延迟 | 抖动 (StdDev) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Native-Laya-MLX (Python SDK)** | 136.6 ms | 16.13 ms | 16.11 ms | 16.79 ms | 17.71 ms | 22.49 ms | ±1.45 ms |
| **Rapid-MLX System-One (HTTP API)** | 42.7 ms | 16.85 ms | 16.69 ms | 17.65 ms | 18.38 ms | 21.09 ms | ±1.07 ms |
| **oMLX Bi-Encoder Embeddings (HTTP API)** | 0.5 ms | 0.29 ms | 0.26 ms | 0.36 ms | 0.43 ms | 0.70 ms | ±0.10 ms |

---

## 4. 并发压力与吞吐饱和曲线 (Concurrency Saturation)

### Native-Laya-MLX (Python SDK)

| 并发度 (Concurrency) | 吞吐量 (RPS) | P50 延迟 | P95 延迟 | 总耗时 |
| :---: | :---: | :---: | :---: | :---: |
| `c = 1` | **61.3 req/s** | 16.00 ms | 17.54 ms | 0.82 s |
| `c = 2` | **78.7 req/s** | 24.98 ms | 27.10 ms | 0.64 s |
| `c = 4` | **80.1 req/s** | 49.56 ms | 59.59 ms | 0.62 s |
| `c = 8` | **76.1 req/s** | 99.31 ms | 146.58 ms | 0.66 s |

### Rapid-MLX System-One (HTTP API)

| 并发度 (Concurrency) | 吞吐量 (RPS) | P50 延迟 | P95 延迟 | 总耗时 |
| :---: | :---: | :---: | :---: | :---: |
| `c = 1` | **47.3 req/s** | 17.75 ms | 24.23 ms | 1.06 s |
| `c = 2` | **62.4 req/s** | 31.68 ms | 33.62 ms | 0.80 s |
| `c = 4` | **62.3 req/s** | 63.48 ms | 66.23 ms | 0.80 s |
| `c = 8` | **62.9 req/s** | 124.95 ms | 131.47 ms | 0.79 s |

### oMLX Bi-Encoder Embeddings (HTTP API)

| 并发度 (Concurrency) | 吞吐量 (RPS) | P50 延迟 | P95 延迟 | 总耗时 |
| :---: | :---: | :---: | :---: | :---: |
| `c = 1` | **2964.6 req/s** | 0.29 ms | 0.64 ms | 0.02 s |
| `c = 2` | **5394.9 req/s** | 0.35 ms | 0.44 ms | 0.01 s |
| `c = 4` | **7094.8 req/s** | 0.53 ms | 0.79 ms | 0.01 s |
| `c = 8` | **6573.1 req/s** | 0.97 ms | 2.23 ms | 0.01 s |


---

## 5. 测试包使用指南 (Usage Guide)

### 快速复现评测
```bash
# 1. 运行完整测试包
./run_benchmark.sh

# 2. 单独测试精度与对抗样本
python -m benchmarks.system_one_benchmark.runners.run_accuracy

# 3. 单独测试延迟分布
python -m benchmarks.system_one_benchmark.runners.run_performance

# 4. 单独测试并发吞吐
python -m benchmarks.system_one_benchmark.runners.run_concurrency
```
