"""Markdown report generator for DecisionBench."""

from typing import Dict, Any
from pathlib import Path
from decision_bench.evaluator import BenchmarkReport


def generate_markdown_report(report: BenchmarkReport) -> str:
    """Generate comprehensive GitHub-flavored Markdown benchmark report."""
    md = []
    md.append(f"# 🚀 DecisionBench: Decision Model Benchmark Report")
    md.append(f"\n**Model Under Test**: `{report.model_name}`  ")
    md.append(f"**Generated At**: `{report.timestamp}`  ")
    md.append(f"**Platform**: `{report.environment.get('platform', 'N/A')}` (`{report.environment.get('processor', 'N/A')}`)  \n")

    md.append("---")
    md.append("## 🏆 Overall Decision Index (D-Index)")
    md.append("")
    md.append(f"| Metric | Score | Rating |")
    md.append(f"| :--- | :--- | :--- |")

    def _rating(score: float) -> str:
        if score >= 90: return "🌟 Exceptional (Production-Grade S1)"
        if score >= 75: return "🟢 Strong / Capable"
        if score >= 60: return "🟡 Acceptable / Marginal"
        return "🔴 Needs Improvement"

    md.append(f"| **Decision Quality Score** | **{report.overall_quality_score:.1f} / 100** | {_rating(report.overall_quality_score)} |")
    md.append(f"| **Latency & Speed Score** | **{report.overall_speed_score:.1f} / 100** | {_rating(report.overall_speed_score)} |")
    md.append(f"| **Composite Decision Index (D-Index)** | **{report.decision_index:.1f} / 100** | {_rating(report.decision_index)} |")
    md.append("")

    md.append("---")
    md.append("## 📊 Task-by-Task Performance Summary")
    md.append("")
    md.append("| Task Key | Task Name | Primitive | Samples | Primary Metric | P50 Latency | P95 Latency | Throughput |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for k, row in report.summary_table.items():
        md.append(
            f"| `{k}` | {row['task_name']} | `{row['type']}` | {row['samples']} | "
            f"**{row['primary_metric']}** | {row['p50_ms']} ms | {row['p95_ms']} ms | {row['qps']} req/s |"
        )
    md.append("")

    md.append("---")
    md.append("## 🔍 Detailed Metric Breakdown")
    md.append("")

    for task_key, tr in report.task_results.items():
        md.append(f"### 📌 Task: `{tr.task_name}` ({tr.task_type.value})")
        md.append(f"> {tr.description}")
        md.append("")
        md.append(f"- **Evaluated Samples**: `{tr.num_samples}`")
        md.append(f"- **Total Time**: `{tr.execution_time_seconds}s` (Errors: `{tr.error_count}`)")
        md.append(f"- **Latency (ms)**: P50=`{tr.latency.get('p50_ms')}`, P90=`{tr.latency.get('p90_ms')}`, P95=`{tr.latency.get('p95_ms')}`, P99=`{tr.latency.get('p99_ms')}`, Mean=`{tr.latency.get('mean_ms')}`")
        md.append("")
        md.append("#### Metrics:")
        md.append("```json")
        import json
        clean_metrics = {k: v for k, v in tr.metrics.items() if k not in ("confusion_matrix", "per_class", "top_failures")}
        md.append(json.dumps(clean_metrics, indent=2, ensure_ascii=False))
        md.append("```")
        md.append("")

        # Per-class table if available
        if "per_class" in tr.metrics and tr.metrics["per_class"]:
            md.append("#### Per-Class Classification Report:")
            md.append("| Category Track | Precision | Recall | F1-Score | Support |")
            md.append("| :--- | :--- | :--- | :--- | :--- |")
            for cat, scores in tr.metrics["per_class"].items():
                md.append(f"| {cat} | {scores.get('precision', 0.0)} | {scores.get('recall', 0.0)} | {scores.get('f1', 0.0)} | {scores.get('support', 0)} |")
            md.append("")

        # Top failures if adversarial
        if "top_failures" in tr.metrics and tr.metrics["top_failures"]:
            md.append("#### Adversarial Stress Test Edge Cases (Sample Failures):")
            for f in tr.metrics["top_failures"][:5]:
                md.append(f"- **Title**: `{f.get('title')}`")
                md.append(f"  - Expected: `{f.get('expected')}` | Predicted: `{f.get('predicted')}`")
                if f.get("notes"):
                    md.append(f"  - Notes: *{f.get('notes')}*")
            md.append("")

    md.append("---")
    md.append("## 💡 System One Architectural Assessment & Guidelines")
    md.append("")
    if report.overall_speed_score >= 80:
        md.append("✅ **Latency Grade**: High-speed real-time gating capable (<20ms P50). Suitable for pre-retrieval routing and edge dispatch.")
    else:
        md.append("⚠️ **Latency Grade**: Elevated latency (>50ms). Consider serving via C++/MLX compiled runtime or batch pipelining.")

    if report.overall_quality_score >= 80:
        md.append("✅ **Decision Fidelity**: High discrimination fidelity across discrete choice and boolean verification.")
    else:
        md.append("⚠️ **Decision Fidelity**: Model exhibits category confusion on subtle boundaries; recommend track prior smoothing or fine-tuning.")
    md.append("")

    return "\n".join(md)


def save_markdown_report(report: BenchmarkReport, output_path: str) -> None:
    """Save markdown report to disk."""
    content = generate_markdown_report(report)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
