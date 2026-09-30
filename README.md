# ⚡ Laya Stack & DecisionBench

[**简体中文**](README.md) | [**English**](README_en.md)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Apple Silicon MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-orange.svg)](https://github.com/ml-explore/mlx)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Type: System One](https://img.shields.io/badge/architecture-System%20One-brightgreen.svg)]()
[![Benchmark: 1024 Samples](https://img.shields.io/badge/benchmark-1024%20samples-purple.svg)]()

**Laya Stack** 是专为 **System One（快速决策系统）与决策模型（Decision Model）** 打造的全栈工具集与标准化评测平台。结合了基于 Apple Silicon MLX 架构的毫秒级决策推理 Studio、通用评测基准套件 **DecisionBench**，以及真实数据标注与采集流。

> [!IMPORTANT]
> **评测数据集语言说明 (Dataset Language Notice)**：  
> 当前 DecisionBench 内置的基准评测数据集（共 1,024 篇多任务标注样本）**均为中文真实场景数据**，源自小红书 12 大垂直行业。  
> *The current built-in benchmark dataset (1,024 annotated samples) consists entirely of real-world Chinese content covering 12 vertical domains.*

---

## 🌟 核心组成模块

仓库由三大协同模块构成：

```
laya-stack/
├── decision_bench/         # 🚀 DecisionBench: System One 通用标准化基准评测包
│   ├── dataset/            # 1,024 篇真实中文标注数据集 (涵盖 12 大垂直赛道)
│   ├── models/             # 通用模型适配器 (Rapid-MLX / 本地 MLX / OpenAI / Mock)
│   ├── tasks/              # 5 大核心决策原语任务 (Choice, Noul, Score, Rank, Adversarial)
│   ├── metrics/            # 精度、校准度 (ECE)、秩相关、分位延迟 (P50/P95/P99)
│   ├── evaluator.py        # 评测编排器与 D-Index (综合决策指数) 评分算法
│   └── cli.py              # 交互式 CLI 工具 (`decision-bench`)
├── src/laya_stack/         # 🖥️ Laya MLX Decision Studio: 决策推理后端与 Web UI
│   ├── server.py           # FastAPI 服务、模型路由网关、流式标注接口
│   ├── structured.py       # Pydantic / JSON Schema 自动映射与结构化决策
│   └── static/             # 交互式现代化 Web 控制台前端
└── extension/              # 🧩 Chrome 浏览器扩展 (真实小红书信息流实时标注与数据采集)
```

---

## 🚀 模块一：DecisionBench 通用决策模型评测套件

专为非自回归、单步前向推理的 Decision Model 定制，告别仅评估长文本生成速度（TPS）的传统方式，专注于**决策保真度**与**毫秒级延迟（<20ms）**。

### 1. 1,024 篇真实中文数据集
- **赛道覆盖**：旅游出行 (273)、科技数码 (99)、摄影剪辑 (77)、萌宠动物 (73)、美食探店 (65)、情感心理 (63)、家居生活 (63)、母婴育儿 (63)、美妆护肤 (62)、穿搭时尚 (62)、职场成长 (62)、运动健康 (62)。
- **多任务标注**：12 分类标准赛道、标题党/博眼球标签（39.1%）、商业广告软文标签（6.0%）、爆款潜力指数（1-5 分）、一词多义对抗样本（Adversarial）。

### 2. 5 大决策原语评测
- `choice`: 12 选 1 垂直分类（准确率、Macro-F1、混淆矩阵、ECE 校准误差）。
- `noul`: 布尔二元验证（标题党拦截、商业广告探测，精准率/召回率/Brier Score）。
- `score`: 1-5 分爆款潜力打分（MAE、RMSE、Spearman 秩相关系数、容差命中率）。
- `rank`: 4 选 1 候选笔记语义排序（NDCG@3、MRR 倒数排名、Top-1 命中率）。
- `adversarial`: 一词多义与跨领域实体抗干扰抗混淆压力测试。

### 3. 综合决策指数 (Decision Index, D-Index)
$$\text{D-Index} = 0.70 \times \text{Quality} + 0.30 \times \text{Speed}$$

- **Speed Score 标准**：$\text{P50} \le 15\text{ms}$ 得满分；$50\text{ms}$ 得 75 分；$>200\text{ms}$ 进入衰减区。
- **评级标准**：
  - $\ge 90$：🌟 **Production-Grade S1**（超低延迟+高保真，满足前置毫秒级路由）
  - $75 \sim 89$：🟢 **Strong / Capable**（适合准实时流式过滤与在线风控）
  - $60 \sim 74$：🟡 **Marginal**（适合离线或异步批处理）

### 4. CLI 快速使用

安装 DecisionBench：
```bash
pip install -e decision_bench
```

评测指令：
```bash
# 1. 查看数据集统计 (当前全量为中文样本)
decision-bench dataset-info

# 2. 列出所有任务
decision-bench list-tasks

# 3. 评测 HTTP 服务 (如 Rapid-MLX 运行在 8700 端口)
decision-bench run \
  --model http://127.0.0.1:8700 \
  --tasks all \
  --output-md reports/rapid_mlx_report.md \
  --output-json reports/rapid_mlx_report.json

# 4. 评测本地 MLX 原生模型
decision-bench run \
  --model laya \
  --tasks choice,clickbait,score \
  --limit 50

# 5. 双模型对比 (A/B Test)
decision-bench compare reports/model_a.json reports/model_b.json
```

---

## 🖥️ 模块二：Laya MLX Decision Studio

基于 Apple Silicon MLX 架构加速的 [Laya](https://nandhakishorm.github.io/laya/) 决策推理全功能 Web 服务与交互界面。

### 核心特性
- **Schema 结构化决策 (`agent.decide`)**：将任意 JSON Schema 或 Pydantic 模型自动转为单次前向决策，单次推理输出强类型结果。
- **置信度门控（Confidence Gating）**：支持 `min_confidence`，低于阈值自动标记或回退人工。
- **官方 5 大业务预设**：开箱即用的工单分诊 (`triage`)、邮件威胁 (`email`)、LLM 护栏 (`guard`)、内容风控 (`moderation`)、智能体路由 (`router`)。
- **向量初筛候选排序 (`shortlist`)**：面对海量候选先做余弦相似度初筛，再由决策核心裁决。

### 启动 Web 界面
```bash
# 启动本地服务
uv run laya-stack
# 或
.venv/bin/python -m laya_stack.server
```
在浏览器打开：👉 **http://localhost:8088**

---

## 🧩 模块三：小红书数据采集与实时标注扩展

位于 [`extension/`](extension/)：
- 支持小红书网页版信息流实时自动打标（品类识别、带货检测、爆款打分）。
- 点击插件图标可一键批量采集并导出真实标注样本，直接扩充 DecisionBench 测试集。
- 安装方式：Chrome 打开 `chrome://extensions/` ➔ 开启开发者模式 ➔ 加载已解压的扩展程序 ➔ 选择 `extension` 目录。

---

## 🧪 单元测试

运行 DecisionBench 全套测试：
```bash
pytest decision_bench/tests -v
```

---

## 📄 开源许可证

本项目采用 [MIT License](LICENSE) 开源许可证。
