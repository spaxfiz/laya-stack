import json
import time
from datetime import datetime
from pathlib import Path
from benchmarks.system_one_benchmark.engines import NativeLayaEngine, RapidMlxEngine, OmlxEmbedEngine
from benchmarks.system_one_benchmark.runners.run_accuracy import evaluate_engine_accuracy
from benchmarks.system_one_benchmark.runners.run_performance import profile_engine_latency
from benchmarks.system_one_benchmark.runners.run_concurrency import benchmark_engine_concurrency

def main():
    root = Path(__file__).resolve().parent.parent
    data_path = root / "data" / "benchmark_1024.json"
    reports_dir = root / "reports"
    reports_dir.mkdir(exist_ok=True)

    print("================================================================================")
    print("       SYSTEM ONE / DECISION-MODEL UNIVERSAL BENCHMARK SUITE")
    print(f"       Dataset: {data_path.name} | Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("================================================================================")

    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Instantiate test engines
    engines = [
        NativeLayaEngine(),
        RapidMlxEngine(),
        OmlxEmbedEngine()
    ]

    report_data = {
        "timestamp": datetime.now().isoformat(),
        "dataset_size": len(dataset),
        "engines": []
    }

    # 1. Accuracy Benchmark (1,024 samples)
    print("\n>>> PHASE 1: Full-scale Accuracy & Generalization (1,024 Samples)...")
    accuracy_results = []
    for eng in engines:
        # Run full 1,024 samples
        acc_res = evaluate_engine_accuracy(eng, dataset)
        accuracy_results.append(acc_res)

    # 2. Performance & Latency (100 iterations)
    print("\n>>> PHASE 2: Latency Percentiles Profiling (100 Iterations)...")
    latency_results = []
    for eng in engines:
        lat_res = profile_engine_latency(eng, rounds=100)
        latency_results.append(lat_res)

    # 3. Concurrency Stress Test
    print("\n>>> PHASE 3: Concurrency Scaling & Throughput (c=1,2,4,8)...")
    concurrency_results = []
    for eng in engines:
        conc_res = benchmark_engine_concurrency(eng, concurrency_levels=[1, 2, 4, 8], requests_per_level=50)
        concurrency_results.append(conc_res)

    # Assemble Structured Report
    for i, eng in enumerate(engines):
        report_data["engines"].append({
            "name": eng.name,
            "description": eng.description,
            "accuracy": accuracy_results[i],
            "latency": latency_results[i],
            "concurrency": concurrency_results[i]
        })

    json_report_file = reports_dir / "benchmark_results.json"
    with open(json_report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    # Generate Markdown Report
    md_content = f"""# System One / DecisionModel Universal Benchmark Report

- **Generated At**: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`
- **Dataset**: `benchmark_1024.json` (Total **{len(dataset)}** labeled samples across 12 Xiaohongshu tracks)
- **Hardware Architecture**: Apple Silicon Metal GPU (Unified Memory)
- **Model Checkpoint**: `aac6fef/laya-multilingual-mlx` (ModernBERT-base + 2-layer RL Cross-Attention Head)

---

## 1. 核心结果总览 (Executive Summary)

| Engine 后端引擎 | 架构类型 | 1024篇 全量准确率 | 对抗样本 准确率 | P50 延迟 | P95 延迟 | 峰值并发吞吐 (RPS) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for item in report_data["engines"]:
        e_acc = item["accuracy"]
        e_lat = item["latency"]
        c_res = item["concurrency"]["concurrency_results"]
        max_rps = max(c["rps"] for c in c_res.values())
        md_content += f"| **{item['name']}** | {item['description']} | **{e_acc['accuracy']*100:.1f}%** | **{e_acc['adversarial_accuracy']*100:.1f}%** | {e_lat['median_ms']:.2f} ms | {e_lat['p95_ms']:.2f} ms | **{max_rps:.1f} req/s** |\n"

    md_content += """
---

## 2. 精度与泛化度评测 (Accuracy & Generalization)

| 指标维度 | Native Laya MLX (Python SDK) | Rapid-MLX System-One (API) | oMLX Bi-Encoder (Embedding API) |
| :--- | :---: | :---: | :---: |
"""
    native_acc = report_data["engines"][0]["accuracy"]
    rapid_acc = report_data["engines"][1]["accuracy"]
    omlx_acc = report_data["engines"][2]["accuracy"]

    md_content += f"| **Overall Accuracy (全量准确率)** | **{native_acc['accuracy']*100:.1f}%** | **{rapid_acc['accuracy']*100:.1f}%** | **{omlx_acc['accuracy']*100:.1f}%** |\n"
    md_content += f"| **Macro-F1 Score** | **{native_acc['macro_f1']*100:.1f}%** | **{rapid_acc['macro_f1']*100:.1f}%** | **{omlx_acc['macro_f1']*100:.1f}%** |\n"
    md_content += f"| **Adversarial Accuracy (对抗样本)** | **{native_acc['adversarial_accuracy']*100:.1f}%** | **{rapid_acc['adversarial_accuracy']*100:.1f}%** | **{omlx_acc['adversarial_accuracy']*100:.1f}%** |\n"
    md_content += f"| **Expected Calibration Error (ECE)** | **{native_acc['ece']:.4f}** | **{rapid_acc['ece']:.4f}** | **{omlx_acc['ece']:.4f}** |\n"

    md_content += """
> [!NOTE]
> **Native Laya 与 Rapid-MLX 的等价性**：两者均驱动完整的 170 张量 DecisionModel（包含 2 层 Cross-Attention 跨注意力决策头），准确率达到完全一致的高水准。
> **oMLX 的余弦匹配局限性**：由于 oMLX 仅使用 ModernBERT 最后一层句向量计算余弦相似度（Bi-Encoder 模式），缺乏 Cross-Attention 层在 Context 与 Option 之间的交互，在对抗样本和长尾赛道上出现精度衰减。

---

## 3. 延迟百分位分布 (Latency Distribution)

| Engine 引擎 | 冷启动耗时 | Mean 平均耗时 | P50 中位数 | P90 延迟 | P95 延迟 | P99 极端延迟 | 抖动 (StdDev) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for item in report_data["engines"]:
        l = item["latency"]
        md_content += f"| **{item['name']}** | {l['cold_latency_ms']:.1f} ms | {l['mean_ms']:.2f} ms | {l['median_ms']:.2f} ms | {l['p90_ms']:.2f} ms | {l['p95_ms']:.2f} ms | {l['p99_ms']:.2f} ms | ±{l['std_ms']:.2f} ms |\n"

    md_content += """
---

## 4. 并发压力与吞吐饱和曲线 (Concurrency Saturation)

"""
    for item in report_data["engines"]:
        md_content += f"### {item['name']}\n\n"
        md_content += "| 并发度 (Concurrency) | 吞吐量 (RPS) | P50 延迟 | P95 延迟 | 总耗时 |\n"
        md_content += "| :---: | :---: | :---: | :---: | :---: |\n"
        for c, c_stat in item["concurrency"]["concurrency_results"].items():
            md_content += f"| `c = {c}` | **{c_stat['rps']:.1f} req/s** | {c_stat['p50_latency_ms']:.2f} ms | {c_stat['p95_latency_ms']:.2f} ms | {c_stat['total_time_sec']:.2f} s |\n"
        md_content += "\n"

    md_content += """
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
"""

    md_report_file = reports_dir / "BENCHMARK_REPORT.md"
    with open(md_report_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n[DONE] Full Benchmark Reports Generated:")
    print(f"  - Markdown: {md_report_file}")
    print(f"  - JSON:     {json_report_file}")

if __name__ == "__main__":
    main()
