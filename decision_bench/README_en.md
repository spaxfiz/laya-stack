# 🚀 DecisionBench: Standardized Benchmark Suite for System One & Decision Models

[**简体中文**](README.md) | [**English**](README_en.md)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: Pytest](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Type: System One](https://img.shields.io/badge/architecture-System%20One-orange.svg)]()
[![Dataset: 1024 Samples](https://img.shields.io/badge/dataset-1024%20samples-purple.svg)]()

**DecisionBench** is an open, standardized benchmark codebase and test suite specifically engineered to evaluate **Decision Models / System One architectures** (e.g. Laya, Rapid-MLX, CLM, fast classifiers, gating routers, and zero-shot decision models).

Unlike traditional LLM benchmarks that evaluate token generation throughput (tokens/sec) on long reasoning chains, **DecisionBench** assesses high-speed, non-autoregressive decision primitives where millisecond latency, probabilistic calibration, and discrete routing fidelity are critical.

> [!IMPORTANT]
> **Dataset Language Notice**:  
> The current built-in benchmark dataset (1,024 annotated samples) consists **entirely of real-world Chinese content**, curated across 12 Xiaohongshu (RED) vertical domains.

---

## 🎯 Key Capabilities & Highlights

1. **Standardized 1,024 Real-World Chinese Dataset**:
   - Curated from 1,000+ real-world Xiaohongshu (RED) notes across all **12 vertical tracks**.
   - Multi-task ground-truth annotations: Category track, Clickbait boolean flag (39.1%), Commercial Ad boolean flag (6.0%), Viral engagement potential (1-5), and Adversarial polysemy edge-cases.
2. **Standard Decision Primitives**:
   - `choice`: Discrete N-way classification (12-way vertical classification).
   - `noul`: Binary boolean proposition verification (Clickbait & Commercial Ad detection).
   - `score`: Continuous/ordinal rating on a 1-5 scale (Viral potential score).
   - `rank`: Top-K candidate semantic ranking against distractors (NDCG@3, MRR).
   - `adversarial`: Cross-domain polysemy stress test (e.g., Apple fruit vs iPhone, Yak travel vs traffic).
3. **Comprehensive Metric Suite**:
   - **Classification**: Macro/Micro/Weighted Precision, Recall, F1, Accuracy, Confusion Matrix.
   - **Regression**: MAE, MSE, RMSE, Spearman Rank Correlation, Tolerance Accuracy.
   - **Ranking**: NDCG@K, MRR (Mean Reciprocal Rank), Top-1 & Top-2 Hit Rates.
   - **Calibration**: Expected Calibration Error (ECE), Brier Score.
   - **Latency**: P50, P90, P95, P99, Throughput (QPS), Jitter.
4. **Universal Model Adapters**:
   - **REST / HTTP Models**: Rapid-MLX System One API, custom HTTP microservices.
   - **In-Process Python SDK**: Native Apple Silicon MLX models (`laya_mlx.Agent`).
   - **OpenAI-Compatible**: vLLM, Ollama, SGLang, Groq, OpenAI endpoints.
   - **Mock Model**: Deterministic, zero-dependency model for CI/CD and offline tests.
5. **Composite Decision Index (D-Index)**:
   - A single unified index combining **Decision Quality (70%)** and **Inference Speed (30%)** to determine if a model is production-ready for real-time edge gating.

---

## 📂 Repository Structure

```
decision_bench/
├── pyproject.toml                     # PEP 621 package metadata & scripts
├── README.md                          # Chinese documentation
├── README_en.md                       # English documentation
├── decision_bench/
│   ├── dataset/                       # 1,024 Chinese real-world annotated samples
│   │   ├── schema.py                  # Pydantic schema (BenchmarkSample, TaskType)
│   │   ├── loader.py                  # Data loading, filtering, & summary stats
│   │   └── data/
│   │       └── decision_benchmark_1024.json # 1,024 annotated benchmark samples
│   ├── models/                        # Model adapters
│   │   ├── base.py                    # BaseDecisionModel interface & DecisionOutput
│   │   ├── mock_model.py              # MockDecisionModel for CI & smoke tests
│   │   ├── system_one_http.py         # SystemOneHttpModel (Rapid-MLX / REST)
│   │   ├── laya_mlx_model.py          # LayaMlxModel (native in-process MLX)
│   │   └── openai_model.py            # OpenAICompatibleModel (vLLM / Ollama)
│   ├── tasks/                         # 5 core decision primitives
│   │   ├── track_choice.py            # 12-way vertical classification
│   │   ├── clickbait_noul.py          # Clickbait boolean verification
│   │   ├── ad_detection_noul.py       # Commercial ad boolean verification
│   │   ├── viral_score.py             # 1-5 viral score ordinal regression
│   │   ├── ranking_task.py            # 4-way candidate relevance ranking
│   │   └── adversarial_stress.py      # Hard / polysemy disambiguation stress test
│   ├── metrics/                       # Classification, regression, ranking, calibration, latency
│   ├── reporting/                     # Console, Markdown, and JSON export
│   ├── evaluator.py                   # Central benchmark runner & D-Index calculator
│   └── cli.py                         # Rich interactive CLI entrypoint
└── tests/                             # 11 unit tests
```

---

## ⚡ Quickstart

### 1. Installation

```bash
pip install -e .
```

### 2. CLI Usage

```bash
# View dataset statistics (Chinese real-world notes)
decision-bench dataset-info

# View all available benchmark tasks
decision-bench list-tasks

# Benchmark a remote HTTP endpoint (e.g. Rapid-MLX on port 8700)
decision-bench run \
  --model http://127.0.0.1:8700 \
  --tasks all \
  --output-md reports/rapid_mlx_report.md \
  --output-json reports/rapid_mlx_report.json

# Benchmark native in-process MLX model
decision-bench run \
  --model laya \
  --tasks choice,clickbait,score \
  --limit 50

# Fast dry-run smoke test
decision-bench run --model mock --limit 20

# Compare two models side-by-side (A/B Test)
decision-bench compare reports/model_a.json reports/model_b.json
```

---

## 🐍 Python SDK Usage

```python
from decision_bench.evaluator import DecisionBenchEvaluator
from decision_bench.models import create_model
from decision_bench.reporting import print_report_to_console, save_markdown_report

# 1. Connect to model
model = create_model("http://127.0.0.1:8700")

# 2. Run evaluation
evaluator = DecisionBenchEvaluator()
report = evaluator.run(
    model=model,
    tasks=["choice", "clickbait", "ad", "score", "rank"],
    limit=50
)

# 3. View & save reports
print_report_to_console(report)
save_markdown_report(report, "reports/benchmark.md")
```

---

## 📐 Scoring Formula: Decision Index (D-Index)

$$D = 0.70 \times \text{Quality} + 0.30 \times \text{Speed}$$

- **Quality Score (0-100)**: Macro-averaged accuracy, top-1 hit rate, and ordinal consistency across all active tasks.
- **Speed Score (0-100)**:
  - $\text{P50} \le 15\text{ms} \implies 100$
  - $\text{P50} \le 50\text{ms} \implies 75 - 100$
  - $\text{P50} \le 200\text{ms} \implies 40 - 75$
  - $\text{P50} > 200\text{ms} \implies < 40$

### Production Suitability Tiers
- **$\ge 90$**: 🌟 *Exceptional (Production-Grade System One)* - Suitable for pre-retrieval routing, real-time edge dispatch, and millisecond gating.
- **$75 - 89$**: 🟢 *Strong / Capable* - High fidelity, suitable for asynchronous stream filtering and online content evaluation.
- **$60 - 74$**: 🟡 *Marginal* - Latency or accuracy tradeoffs; suitable for offline batch pipelines.
- **$< 60$**: 🔴 *Sub-optimal* - Needs retraining, prior tuning, or serving engine optimization.

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
