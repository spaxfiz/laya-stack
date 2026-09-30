import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from benchmarks.system_one_benchmark.engines import BaseDecisionEngine, NativeLayaEngine, RapidMlxEngine, OmlxEmbedEngine
from benchmarks.system_one_benchmark.metrics import compute_classification_metrics, compute_ece

CATEGORIES = [
    "科技数码", "萌宠动物", "美妆护肤", "穿搭时尚", "美食探店", "职场成长",
    "家居生活", "运动健康", "母婴育儿", "旅游出行", "情感心理", "摄影剪辑"
]

def evaluate_engine_accuracy(engine: BaseDecisionEngine, dataset: List[Dict[str, Any]], limit: Optional[int] = None) -> Dict[str, Any]:
    items = dataset[:limit] if limit else dataset
    y_true = []
    y_pred = []
    confidences = []
    latencies = []

    # Adversarial subsets
    adv_true = []
    adv_pred = []

    questions = {
        "track": {
            "type": "choice",
            "instructions": "判断小红书笔记的垂直行业分类赛道",
            "criteria": {c: None for c in CATEGORIES}
        }
    }

    print(f"\n[Evaluating {engine.name}] running on {len(items)} samples...")
    t0 = time.perf_counter()

    for idx, sample in enumerate(items):
        title = sample["title"]
        snip = sample.get("snippet")
        gt = sample["ground_truth_track"]
        is_adv = sample.get("difficulty") == "adversarial"
        
        state = f"小红书笔记标题: {title.strip()}\n正文片段: {snip.strip()}" if snip else f"小红书笔记标题: {title.strip()}"
        
        try:
            res = engine.predict(state, questions)
            ans = res["answers"]["track"]
            pred = ans["choice"]
            conf = ans.get("confidence", 0.5)
            lat = res.get("latency_ms", 0.0)
        except Exception as e:
            pred = "ERROR"
            conf = 0.0
            lat = 0.0

        y_true.append(gt)
        y_pred.append(pred)
        confidences.append(conf)
        latencies.append(lat)

        if is_adv:
            adv_true.append(gt)
            adv_pred.append(pred)

        if (idx + 1) % 200 == 0 or (idx + 1) == len(items):
            cur_acc = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true)
            print(f"  Processed {idx + 1}/{len(items)} -> Acc: {cur_acc:.1%}")

    total_time = time.perf_counter() - t0
    cls_metrics = compute_classification_metrics(y_true, y_pred, CATEGORIES)
    ece_metrics = compute_ece(confidences, y_true, y_pred)

    adv_acc = sum(1 for yt, yp in zip(adv_true, adv_pred) if yt == yp) / max(len(adv_true), 1)

    return {
        "engine_name": engine.name,
        "engine_desc": engine.description,
        "total_evaluated": len(items),
        "total_time_sec": round(total_time, 2),
        "avg_throughput_qps": round(len(items) / total_time, 1),
        "accuracy": cls_metrics["accuracy"],
        "macro_f1": cls_metrics["macro_f1"],
        "macro_precision": cls_metrics["macro_precision"],
        "macro_recall": cls_metrics["macro_recall"],
        "adversarial_accuracy": round(adv_acc, 4),
        "adversarial_count": len(adv_true),
        "ece": ece_metrics["ece"],
        "per_class": cls_metrics["per_class"]
    }

if __name__ == "__main__":
    from typing import Optional
    data_path = Path("benchmarks/system_one_benchmark/data/benchmark_1024.json")
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    engines = [
        NativeLayaEngine(),
        RapidMlxEngine(),
        OmlxEmbedEngine()
    ]

    results = []
    for eng in engines:
        res = evaluate_engine_accuracy(eng, dataset)
        results.append(res)

    print("\n" + "=" * 80)
    print("  ACCURACY BENCHMARK SUMMARY (1,024 SAMPLES)")
    print("=" * 80)
    print(f"{'Engine':<35} | {'Accuracy':<10} | {'Macro F1':<10} | {'Adv Acc':<10} | {'ECE':<8}")
    print("-" * 80)
    for r in results:
        print(f"{r['engine_name']:<35} | {r['accuracy']*100:6.1f}%    | {r['macro_f1']*100:6.1f}%    | {r['adversarial_accuracy']*100:6.1f}%    | {r['ece']:6.4f}")
