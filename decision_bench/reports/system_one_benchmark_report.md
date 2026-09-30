# 🚀 DecisionBench: Decision Model Benchmark Report

**Model Under Test**: `system-one-http`  
**Generated At**: `2026-09-30T20:06:32.328443`  
**Platform**: `macOS-27.0-arm64-arm-64bit` (`arm`)  

---
## 🏆 Overall Decision Index (D-Index)

| Metric | Score | Rating |
| :--- | :--- | :--- |
| **Decision Quality Score** | **64.5 / 100** | 🟡 Acceptable / Marginal |
| **Latency & Speed Score** | **99.0 / 100** | 🌟 Exceptional (Production-Grade S1) |
| **Composite Decision Index (D-Index)** | **74.8 / 100** | 🟡 Acceptable / Marginal |

---
## 📊 Task-by-Task Performance Summary

| Task Key | Task Name | Primitive | Samples | Primary Metric | P50 Latency | P95 Latency | Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `choice` | track_choice | `choice` | 30 | **Acc: 66.7%** | 24.5 ms | 521.43 ms | 8.58 req/s |
| `clickbait` | clickbait_noul | `noul` | 30 | **Acc: 80.0%** | 12.1 ms | 190.53 ms | 25.66 req/s |
| `ad` | ad_detection_noul | `noul` | 30 | **Acc: 76.7%** | 12.07 ms | 20.87 ms | 55.65 req/s |
| `score` | viral_score | `score` | 30 | **Within-1: 80.0%** | 17.3 ms | 142.97 ms | 27.62 req/s |
| `rank` | candidate_ranking | `rank` | 30 | **Top-1: 33.3%** | 13.46 ms | 25.22 ms | 52.89 req/s |
| `adversarial` | adversarial_stress | `adversarial` | 8 | **RobAcc: 50.0%** | 19.18 ms | 21.83 ms | 47.63 req/s |

---
## 🔍 Detailed Metric Breakdown

### 📌 Task: `track_choice` (choice)
> 12-Way Xiaohongshu Vertical Domain Track Classification

- **Evaluated Samples**: `30`
- **Total Time**: `3.5s` (Errors: `0`)
- **Latency (ms)**: P50=`24.5`, P90=`420.86`, P95=`521.43`, P99=`624.23`, Mean=`115.38`

#### Metrics:
```json
{
  "accuracy": 0.6667,
  "macro_precision": 0.4151,
  "macro_recall": 0.5,
  "macro_f1": 0.4269,
  "ece": 0.24
}
```

#### Per-Class Classification Report:
| Category Track | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| 科技数码 | 0.5556 | 1.0 | 0.7143 | 5 |
| 萌宠动物 | 0.6 | 1.0 | 0.75 | 3 |
| 美妆护肤 | 0.0 | 0.0 | 0.0 | 0 |
| 穿搭时尚 | 0.0 | 0.0 | 0.0 | 0 |
| 美食探店 | 0.0 | 0.0 | 0.0 | 1 |
| 职场成长 | 0.0 | 0.0 | 0.0 | 0 |
| 家居生活 | 0.0 | 0.0 | 0.0 | 0 |
| 运动健康 | 0.0 | 0.0 | 0.0 | 0 |
| 母婴育儿 | 0.0 | 0.0 | 0.0 | 0 |
| 旅游出行 | 1.0 | 0.5 | 0.6667 | 18 |
| 情感心理 | 0.0 | 0.0 | 0.0 | 0 |
| 摄影剪辑 | 0.75 | 1.0 | 0.8571 | 3 |

### 📌 Task: `clickbait_noul` (noul)
> Binary Clickbait and Sensationalism Identification

- **Evaluated Samples**: `30`
- **Total Time**: `1.17s` (Errors: `0`)
- **Latency (ms)**: P50=`12.1`, P90=`91.95`, P95=`190.53`, P99=`309.48`, Mean=`37.9`

#### Metrics:
```json
{
  "accuracy": 0.8,
  "macro_f1": 0.4445,
  "clickbait_precision": 0.0,
  "clickbait_recall": 0.0,
  "clickbait_f1": 0.0,
  "clickbait_support": 6,
  "ece": 0.1976,
  "brier_score": 0.188
}
```

### 📌 Task: `ad_detection_noul` (noul)
> Commercial Advertisement and Sponsored Post Detection

- **Evaluated Samples**: `30`
- **Total Time**: `0.54s` (Errors: `0`)
- **Latency (ms)**: P50=`12.07`, P90=`19.25`, P95=`20.87`, P99=`114.86`, Mean=`16.76`

#### Metrics:
```json
{
  "accuracy": 0.7667,
  "macro_f1": 0.434,
  "ad_precision": 0.0,
  "ad_recall": 0.0,
  "ad_f1": 0.0,
  "ad_support": 0,
  "ece": 0.1528,
  "brier_score": 0.1917
}
```

### 📌 Task: `viral_score` (score)
> Viral Engagement Potential Score Rating (1-5 Scale)

- **Evaluated Samples**: `30`
- **Total Time**: `1.09s` (Errors: `0`)
- **Latency (ms)**: P50=`17.3`, P90=`46.77`, P95=`142.97`, P99=`251.19`, Mean=`34.98`

#### Metrics:
```json
{
  "mae": 0.7267,
  "mse": 0.8231,
  "rmse": 0.9072,
  "pearson_r": 0.256,
  "spearman_rho": 0.1989,
  "exact_match": 0.3667,
  "within_1_level": 0.8
}
```

### 📌 Task: `candidate_ranking` (rank)
> 4-Way Candidate Semantic Relevance Ranking

- **Evaluated Samples**: `30`
- **Total Time**: `0.57s` (Errors: `0`)
- **Latency (ms)**: P50=`13.46`, P90=`22.78`, P95=`25.22`, P99=`69.37`, Mean=`17.67`

#### Metrics:
```json
{
  "top1_hit_rate": 0.3333,
  "top2_hit_rate": 0.6333,
  "mrr": 0.5861,
  "ndcg_at_3": 0.5893
}
```

### 📌 Task: `adversarial_stress` (adversarial)
> Adversarial Polysemy and Cross-Domain Entity Disambiguation

- **Evaluated Samples**: `8`
- **Total Time**: `0.17s` (Errors: `0`)
- **Latency (ms)**: P50=`19.18`, P90=`21.04`, P95=`21.83`, P99=`22.46`, Mean=`19.63`

#### Metrics:
```json
{
  "robustness_accuracy": 0.5,
  "macro_f1": 0.4278,
  "hard_subset_accuracy": null,
  "adversarial_subset_accuracy": 0.5,
  "total_adversarial_tested": 8,
  "failure_case_count": 4
}
```

#### Adversarial Stress Test Edge Cases (Sample Failures):
- **Title**: `出门旅游尤其国外，千万别开大疆的无感传输`
  - Expected: `旅游出行` | Predicted: `科技数码`
  - Notes: *包含大疆无人机数码产品，但核心语境是旅游出行安全避坑指南*
- **Title**: `不会白吃的，给你推车省点油钱`
  - Expected: `旅游出行` | Predicted: `情感心理`
  - Notes: *民间互助口语化标题，词面无高频行业关键词*
- **Title**: `菌子季到了，吃吃喝喝胃根本装不下`
  - Expected: `美食探店` | Predicted: `萌宠动物`
  - Notes: *涉及云南地理与生物实体，但核心语义为美食探店*
- **Title**: `买个这玩意花了1.4w，被家里人骂惨了……`
  - Expected: `旅游出行` | Predicted: `科技数码`
  - Notes: *标题极端诱导反差，实为户外装备自驾旅行种草*

---
## 💡 System One Architectural Assessment & Guidelines

✅ **Latency Grade**: High-speed real-time gating capable (<20ms P50). Suitable for pre-retrieval routing and edge dispatch.
⚠️ **Decision Fidelity**: Model exhibits category confusion on subtle boundaries; recommend track prior smoothing or fine-tuning.
