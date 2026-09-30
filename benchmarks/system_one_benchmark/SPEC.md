# System One / DecisionModel Benchmark Specification (v1.0)

## 1. 概述与设计哲学

本规范定义了 **System One（快思决策系统）/ DecisionModel** 的通用评测标准与测试协议。

### 核心设计原则
1. **非生成式架构（Non-Autoregressive）**：区别于传统生成式大模型逐字 Decode 的评测，System One 评测关注**单次前向传播（Single-Forward）**下的多维决策置信度与对数几率。
2. **多题型统一抽象（Typed Primitives）**：
   - `choice`: 离散候选分类（N 选 1）
   - `score`: 连续/离散评分量化（1-5 分）
   - `noul`: 布尔逻辑裁决（True/False，如标题党、广告标识）
   - `rank`: 语义相关度对比排序（Top-K 候选打分）
3. **真实场景与对抗鲁棒性（Real-world & Adversarial）**：
   - 包含真实小红书高动态信息流、口语化缩写与网络流行语；
   - 重点评测多义实体消歧（如“苹果”手机 vs 水果、“大疆”无人机产品 vs 旅游出行安全）。

---

## 2. 评测指标定义 (Metrics Definitions)

### 2.1 精度与鲁棒性指标
* **Top-1 Accuracy**: 全量样本预测类别与真值匹配比例。
* **Macro-F1**: 各类别 F1-Score 的无加权宏平均，防止高频类别掩盖长尾赛道缺陷。
* **Adversarial Accuracy**: 专门针对多义词、反转诱导标题（标题极端恐吓、正文平和种草）等对抗样本的判定准确率。
* **Expected Calibration Error (ECE)**: 预测置信度与实际准确率之间的期望校准误差：
  $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$

### 2.2 性能与硬件效率指标
* **Forward Latency (ms)**:
  - `P50 (Median)`: 中位数响应耗时
  - `P90 / P95 / P99`: 尾部极端延迟
  - `Jitter (StdDev)`: 延迟标准差，衡量执行确定性
* **Throughput (RPS / QPS)**: 每秒可处理的决策请求数。
* **Concurrency Plateau**: 并发扩展下的最大饱和吞吐量。

---

## 3. 标准数据格式 (Dataset Schema)

```json
{
  "id": "s1_bench_0001",
  "title": "笔记标题文本",
  "snippet": "笔记正文抽取片段（可选）",
  "ground_truth_track": "目标赛道（12赛道之一）",
  "ground_truth_is_clickbait": true,
  "ground_truth_is_ad": false,
  "ground_truth_viral_score": 4,
  "difficulty": "easy | medium | hard | adversarial"
}
```

---

## 4. 适配引擎规范 (Engine Protocol)

每个被测引擎必须实现 `BaseDecisionEngine` 接口：
```python
def predict(self, state: str, questions: Dict[str, Any]) -> Dict[str, Any]:
    ...
```
返回值遵循 System One 标准 Wire 格式，必须包含 `answers` 字典与 `latency_ms`。
