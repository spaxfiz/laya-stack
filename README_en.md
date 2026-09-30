# ⚡ Laya Stack & DecisionBench

[**简体中文**](README.md) | [**English**](README_en.md)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Apple Silicon MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-orange.svg)](https://github.com/ml-explore/mlx)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Type: System One](https://img.shields.io/badge/architecture-System%20One-brightgreen.svg)]()
[![Benchmark: 1024 Samples](https://img.shields.io/badge/benchmark-1024%20samples-purple.svg)]()

**Laya Stack** is an open-source, full-stack platform and standardized evaluation benchmark for **System One / Decision Models** (such as Laya, Rapid-MLX, CLM, and fast deterministic gating classifiers). It provides millisecond-latency inference acceleration on Apple Silicon MLX, a comprehensive web studio, a data harvesting browser extension, and the **DecisionBench** evaluation suite.

> [!IMPORTANT]
> **Dataset Language Notice**:  
> The current built-in DecisionBench evaluation dataset (1,024 multi-task annotated samples) consists **entirely of real-world Chinese content**, curated across 12 Xiaohongshu (RED) vertical domains.

---

## 🌟 Architecture & Core Modules

The repository consists of three integrated modules:

```
laya-stack/
├── decision_bench/         # 🚀 DecisionBench: Universal Benchmark Suite for Decision Models
│   ├── dataset/            # 1,024 Chinese real-world annotated samples (12 vertical domains)
│   ├── models/             # Universal model adapters (Rapid-MLX, Native MLX, OpenAI, Mock)
│   ├── tasks/              # 5 core decision primitives (Choice, Noul, Score, Rank, Adversarial)
│   ├── metrics/            # Accuracy, calibration (ECE), rank correlation, latencies (P50/P95/P99)
│   ├── evaluator.py        # Evaluation orchestrator & D-Index scoring engine
│   └── cli.py              # Interactive CLI tool (`decision-bench`)
├── src/laya_stack/         # 🖥️ Laya MLX Decision Studio: Backend & Web UI
│   ├── server.py           # FastAPI service, model router gateway, stream inference
│   ├── structured.py       # Pydantic / JSON Schema automatic mapping for structured decisions
│   └── static/             # Modern interactive Web Studio interface
└── extension/              # 🧩 Chrome Extension (Real-time feed note labeling & dataset harvesting)
```

---

## 🚀 Module 1: DecisionBench Evaluation Suite

Tailored specifically for non-autoregressive, single-forward-pass Decision Models. Unlike traditional LLM benchmarks focused on token generation throughput (tokens/sec), **DecisionBench** measures **decision fidelity** and **sub-20ms latency**.

### 1. 1,024 Real-World Chinese Samples
- **Domain Coverage**: Travel (273), Tech & Digital (99), Photography & Video (77), Pets (73), Food & Dining (65), Psychology & Relationships (63), Home Living (63), Parenting (63), Beauty & Skincare (62), Fashion (62), Career Growth (62), Fitness & Health (62).
- **Multi-Task Ground Truth**: 12-way vertical classification, clickbait/sensationalism detection (39.1%), commercial sponsorship detection (6.0%), viral engagement rating (1-5), and polysemous adversarial stress cases.

### 2. 5 Core Decision Primitives
- `choice`: 12-way discrete track classification (Accuracy, Macro-F1, Confusion Matrix, ECE calibration).
- `noul`: Binary boolean verification (Clickbait filtering, commercial ad detection, Precision/Recall/Brier Score).
- `score`: Continuous/ordinal rating on a 1-5 scale (MAE, RMSE, Spearman rank correlation, tolerance accuracy).
- `rank`: 4-way candidate note semantic ranking against negative distractors (NDCG@3, MRR, Top-1 Hit Rate).
- `adversarial`: Cross-domain polysemy and entity disambiguation stress test.

### 3. Composite Decision Index (D-Index)
$$\text{D-Index} = 0.70 \times \text{Quality} + 0.30 \times \text{Speed}$$

- **Speed Score**: $\text{P50} \le 15\text{ms} \implies 100$; $50\text{ms} \implies 75$; $>200\text{ms} \implies \text{decay zone}$.
- **Production Tiers**:
  - $\ge 90$: 🌟 **Production-Grade S1** (Sub-20ms + high fidelity; ready for pre-retrieval routing).
  - $75 \sim 89$: 🟢 **Strong / Capable** (Suitable for streaming moderation and real-time gating).
  - $60 \sim 74$: 🟡 **Marginal** (Suitable for offline batch processing).

### 4. CLI Quickstart

Install DecisionBench:
```bash
pip install -e decision_bench
```

Run benchmarks:
```bash
# 1. View dataset statistics (Chinese real-world notes)
decision-bench dataset-info

# 2. List all available benchmark tasks
decision-bench list-tasks

# 3. Benchmark a remote HTTP endpoint (e.g. Rapid-MLX on port 8700)
decision-bench run \
  --model http://127.0.0.1:8700 \
  --tasks all \
  --output-md reports/rapid_mlx_report.md \
  --output-json reports/rapid_mlx_report.json

# 4. Benchmark native in-process MLX model
decision-bench run \
  --model laya \
  --tasks choice,clickbait,score \
  --limit 50

# 5. Compare two models side-by-side (A/B Test)
decision-bench compare reports/model_a.json reports/model_b.json
```

---

## 🖥️ Module 2: Laya MLX Decision Studio

A full-featured Web UI and API service powered by Apple Silicon MLX acceleration for [Laya](https://nandhakishorm.github.io/laya/).

### Highlights
- **Schema Structured Decisions (`agent.decide`)**: Converts any JSON Schema or Pydantic model into a single-forward-pass query with strongly typed outputs.
- **Confidence Gating**: Configurable `min_confidence` to automatically flag or fallback on low-certainty fields.
- **5 Official Presets**: Preconfigured triage, email threat detection, prompt guardrails, content moderation, and agent routing.
- **Embedding Shortlist (`shortlist`)**: Pre-filters large option sets with cosine similarity prior to final decision resolution.

### Launching the Studio
```bash
uv run laya-stack
# or
.venv/bin/python -m laya_stack.server
```
Open in browser: 👉 **http://localhost:8088**

---

## 🧩 Module 3: Content Tagging & Harvesting Extension

Located in [`extension/`](extension/):
- Live browser extension for real-time note categorization, ad detection, and viral scoring directly on Xiaohongshu explore feeds.
- One-click batch harvesting to export annotated samples directly into DecisionBench.
- Install: Open Chrome `chrome://extensions/` ➔ Enable Developer Mode ➔ Load Unpacked ➔ Select `extension` directory.

---

## 🧪 Unit Tests

Run the full DecisionBench test suite:
```bash
pytest decision_bench/tests -v
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
