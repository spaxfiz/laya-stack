# 🚀 DecisionBench: System One 与决策模型标准化评测基准套件

[**简体中文**](README.md) | [**English**](README_en.md)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: Pytest](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Type: System One](https://img.shields.io/badge/architecture-System%20One-orange.svg)]()
[![Dataset: 1024 Samples](https://img.shields.io/badge/dataset-1024%20samples-purple.svg)]()

**DecisionBench** 是一个开源的、标准化的决策模型基准评测代码库与测试套件，专为评估 **System One（快思考系统）与决策模型（Decision Models）**（如 Laya、Rapid-MLX、CLM、轻量级分类器、流式路由网关及大模型决策头）而设计。

与传统只评估长文本 Token 生成吞吐量（TPS）的大模型基准不同，**DecisionBench** 专注于评估**毫秒级推理延迟（<20ms）**、**概率校准度（ECE）**以及**离散决策保真度**。

> [!IMPORTANT]
> **评测数据集语言说明 (Dataset Language Notice)**：  
> 当前 DecisionBench 内置的基准评测数据集（共 1,024 篇标注样本）**均为中文真实场景数据**，源自小红书 12 大垂直行业真实笔记。  
> *The current built-in benchmark dataset (1,024 annotated samples) consists entirely of real-world Chinese content.*

---

## 🎯 核心功能与特性

1. **1,024 篇真实中文标注数据集**：
   - 覆盖真实小红书 12 大垂直行业赛道。
   - 多任务联合真值标注：赛道分类、标题党识别（39.1%）、商业广告软文（6.0%）、爆款潜力等级（1-5 分）、一词多义对抗样本。
2. **5 大决策原语全覆盖**：
   - `choice`: 12 选 1 垂直行业离散分类。
   - `noul`: 布尔二元验证（标题党拦截、商业广告检测）。
   - `score`: 1-5 分连续/序数爆款潜力评分。
   - `rank`: 4 选 1 候选笔记语义排序（NDCG@3、MRR）。
   - `adversarial`: 跨赛道与一词多义抗干扰抗混淆压力测试。
3. **完善的指标体系**：
   - **分类指标**：Macro/Micro/Weighted Precision、Recall、F1、Accuracy、混淆矩阵。
   - **回归与序数**：MAE、MSE、RMSE、Spearman 秩相关系数、容差命中率。
   - **排序指标**：NDCG@3、MRR（平均倒数排名）、Top-1 / Top-2 命中率。
   - **校准与风控**：ECE（期望校准误差）、Brier Score。
   - **延迟与性能**：P50、P90、P95、P99、QPS 吞吐、延迟抖动。
4. **全能模型适配器**：
   - **HTTP REST 接口**：支持 Rapid-MLX System One API 及任意微服务。
   - **进程内 Python SDK**：支持 Apple Silicon 原生 MLX 模型（`laya_mlx.Agent`）。
   - **OpenAI 兼容协议**：支持 vLLM、Ollama、SGLang、Groq、OpenAI 接口。
   - **Mock 模型**：免环境依赖，供 CI/CD 自动化流水线快速冒烟测试。
5. **综合决策指数 (D-Index)**：
   - 综合衡量 **决策质量 (70%)** 与 **推理速度 (30%)**，一键判断模型是否达到生产级前置路由标准。

---

## 📂 目录架构

```
decision_bench/
├── pyproject.toml                     # PEP 621 标准包配置与脚本入口
├── README.md                          # 中文文档
├── README_en.md                       # 英文文档
├── decision_bench/
│   ├── dataset/                       # 1,024 篇中文真实标注数据集
│   │   ├── schema.py                  # Pydantic 数据集规范
│   │   ├── loader.py                  # 数据加载、过滤与统计分析
│   │   └── data/
│   │       └── decision_benchmark_1024.json
│   ├── models/                        # 模型适配器
│   │   ├── base.py                    # 统一抽象基类 BaseDecisionModel
│   │   ├── mock_model.py              # Mock 模型
│   │   ├── system_one_http.py         # HTTP 适配器 (Rapid-MLX)
│   │   ├── laya_mlx_model.py          # 原生 MLX 适配器
│   │   └── openai_model.py            # OpenAI 兼容适配器
│   ├── tasks/                         # 5 大原语评测任务
│   │   ├── track_choice.py            # 12 选 1 分类
│   │   ├── clickbait_noul.py          # 标题党验证
│   │   ├── ad_detection_noul.py       # 商业广告识别
│   │   ├── viral_score.py             # 爆款潜力打分
│   │   ├── ranking_task.py            # 候选排序
│   │   └── adversarial_stress.py      # 抗干扰压力测试
│   ├── metrics/                       # 评测指标库
│   ├── reporting/                     # 报表生成 (终端彩色表格、Markdown、JSON)
│   ├── evaluator.py                   # 编排引擎与 D-Index 计算
│   └── cli.py                         # CLI 交互入口
└── tests/                             # 11 项全套单元测试
```

---

## ⚡ 快速上手

### 1. 安装

```bash
pip install -e .
```

### 2. 命令行使用

```bash
# 查看数据集统计 (当前全量为中文样本)
decision-bench dataset-info

# 列出所有任务
decision-bench list-tasks

# 评测 HTTP 服务 (如 Rapid-MLX 运行在 8700 端口)
decision-bench run \
  --model http://127.0.0.1:8700 \
  --tasks all \
  --output-md reports/rapid_mlx_report.md \
  --output-json reports/rapid_mlx_report.json

# 评测本地 MLX 原生模型
decision-bench run \
  --model laya \
  --tasks choice,clickbait,score \
  --limit 50

# 快速冒烟测试
decision-bench run --model mock --limit 20

# 双模型对比 (A/B Test)
decision-bench compare reports/model_a.json reports/model_b.json
```

---

## 🐍 Python SDK 使用示例

```python
from decision_bench.evaluator import DecisionBenchEvaluator
from decision_bench.models import create_model
from decision_bench.reporting import print_report_to_console, save_markdown_report

# 1. 初始化模型
model = create_model("http://127.0.0.1:8700")

# 2. 启动评测
evaluator = DecisionBenchEvaluator()
report = evaluator.run(
    model=model,
    tasks=["choice", "clickbait", "ad", "score", "rank"],
    limit=50
)

# 3. 查看与保存报表
print_report_to_console(report)
save_markdown_report(report, "reports/benchmark.md")
```

---

## 📐 决策指数 (D-Index) 评分体系

$$\text{D-Index} = 0.70 \times \text{Quality} + 0.30 \times \text{Speed}$$

- **Quality Score (0-100)**：各项任务准确率、Top-1 命中率、序数打分一致性的宏平均分。
- **Speed Score (0-100)**：
  - $\text{P50} \le 15\text{ms} \implies 100$
  - $\text{P50} \le 50\text{ms} \implies 75 \sim 100$
  - $\text{P50} \le 200\text{ms} \implies 40 \sim 75$
  - $\text{P50} > 200\text{ms} \implies < 40$

### 生产可用性分级
- **$\ge 90$**：🌟 **Production-Grade S1**（超低延迟+高保真，满足前置毫秒级路由与风控门控）
- **$75 \sim 89$**：🟢 **Strong / Capable**（适合准实时流式过滤与在线内容评测）
- **$60 \sim 74$**：🟡 **Marginal**（延迟或准确率略有妥协，适合离线批处理）
- **$< 60$**：🔴 **Sub-optimal**（未达到 S1 标准，需重新微调或优化推理后端）

---

## 🧪 单元测试

```bash
pytest tests/ -v
```

---

## 📄 开源许可证

本项目采用 [MIT License](LICENSE) 开源许可证。
