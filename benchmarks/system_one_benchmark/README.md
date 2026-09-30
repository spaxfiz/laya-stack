# System One / DecisionModel Universal Benchmark Suite

一套专为 Apple Silicon (MLX) 及轻量级结构化决策大模型（System One / DecisionModel / Laya / CLM）设计的通用、标准化基准测试套件。

本测试包基于 **1,024 篇真实小红书高动态信息流标注数据** 构建，涵盖 12 垂直赛道、标题党噱头辨识、商单软广判别、爆款指数评分以及对抗样本消歧。

---

## 目录结构

```text
system_one_benchmark/
├── README.md                           # 本指南
├── SPEC.md                             # 协议与规范标准定义
├── run_benchmark.sh                    # 一键全自动评测执行脚本
├── data/
│   └── benchmark_1024.json             # 1,024 篇小红书多维标签基准数据集
├── engines/                            # 被测引擎适配层
│   ├── base.py                         # 统一引擎抽象基类 (BaseDecisionEngine)
│   ├── native_laya.py                  # 本地原生 Python SDK (laya_mlx)
│   ├── rapid_mlx.py                    # Rapid-MLX HTTP API (:8700 /v1/systemone)
│   └── omlx_embed.py                   # oMLX 句向量余弦相似度 (:8090 /v1/embeddings)
├── metrics/                            # 评测指标计算库
│   ├── accuracy.py                     # Top-1 Accuracy, Precision, Recall, Macro-F1
│   ├── calibration.py                  # ECE (期望校准误差) 与分桶可靠性
│   └── performance.py                  # P50/P90/P95/P99 延迟分布、RPS 吞吐
├── runners/                            # 自动化评测运行入口
│   ├── run_all.py                      # 全流程测试总编排器
│   ├── run_accuracy.py                 # 单独跑 1,024 篇精度与对抗测试
│   ├── run_performance.py              # 单独跑 100 次冷热启动与延迟分布测试
│   └── run_concurrency.py              # 单独跑 c=1,2,4,8 并发吞吐压测
└── reports/                            # 自动生成的评测报告输出目录
    ├── BENCHMARK_REPORT.md             # 最新 Markdown 格式综合评测报告
    └── benchmark_results.json          # 机器可读的原始评测指标 JSON
```

---

## 快速开始 (Quickstart)

### 1. 准备环境
确保各后台推理服务已就绪：
* **Native Laya**: 本地 Python 虚拟环境安装 `laya-mlx`
* **Rapid-MLX**: 运行在 `http://127.0.0.1:8700` (`rapid-mlx system-one ...`)
* **oMLX**: 运行在 `http://127.0.0.1:8090`

### 2. 一键执行全量评测
```bash
./benchmarks/system_one_benchmark/run_benchmark.sh
```

### 3. 单项测试
```bash
# 单独测试 1024 篇准确率与对抗样本
.venv/bin/python -m benchmarks.system_one_benchmark.runners.run_accuracy

# 单独测试 100 次单请求延迟分布 (P50/P95/P99)
.venv/bin/python -m benchmarks.system_one_benchmark.runners.run_performance

# 单独测试多并发吞吐与饱和度曲线
.venv/bin/python -m benchmarks.system_one_benchmark.runners.run_concurrency
```

---

## 报告解读 (Metrics Interpretation)

1. **Accuracy & Macro-F1**: 衡量多类别不平衡下的综合分类能力。
2. **Adversarial Accuracy**: 衡量面对“苹果跳号”、“大疆无感传输”、“牦牛过马路”等歧义对抗样本时的跨注意力消歧能力。
3. **P50 / P95 Latency**: 真实业务下的人类感知延时指标（System One 架构通常在 15~20ms 内）。
4. **ECE (Expected Calibration Error)**: 衡量置信度是否保真。如果 ECE 接近 0，说明模型输出 90% 置信度时，实际准确率确实在 90% 左右。
