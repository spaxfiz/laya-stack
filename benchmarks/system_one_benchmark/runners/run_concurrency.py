import time
import concurrent.futures
from typing import Any, Dict, List
import numpy as np
from benchmarks.system_one_benchmark.engines import BaseDecisionEngine, NativeLayaEngine, RapidMlxEngine, OmlxEmbedEngine

CATEGORIES = [
    "科技数码", "萌宠动物", "美妆护肤", "穿搭时尚", "美食探店", "职场成长",
    "家居生活", "运动健康", "母婴育儿", "旅游出行", "情感心理", "摄影剪辑"
]

def benchmark_engine_concurrency(engine: BaseDecisionEngine, concurrency_levels: List[int] = [1, 2, 4, 8], requests_per_level: int = 100) -> Dict[str, Any]:
    state = "小红书笔记标题: 苹果再一次跳号！iOS 20 提前泄露，全面转向本地AI交互\n正文片段: 深度测试端侧小模型与Siri交互能力"
    questions = {
        "track": {
            "type": "choice",
            "instructions": "判断小红书笔记的垂直行业分类赛道",
            "criteria": {c: None for c in CATEGORIES}
        }
    }

    results_by_c = {}

    for c in concurrency_levels:
        latencies = []
        t0 = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as executor:
            futures = [executor.submit(engine.predict, state, questions) for _ in range(requests_per_level)]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    res = fut.result()
                    latencies.append(res.get("latency_ms", 0.0))
                except Exception:
                    latencies.append(999.0)

        total_time = time.perf_counter() - t0
        rps = requests_per_level / max(total_time, 1e-4)

        results_by_c[c] = {
            "concurrency": c,
            "total_time_sec": round(total_time, 3),
            "rps": round(rps, 1),
            "p50_latency_ms": round(float(np.median(latencies)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2)
        }

    return {
        "engine_name": engine.name,
        "concurrency_results": results_by_c
    }

if __name__ == "__main__":
    engines = [
        NativeLayaEngine(),
        RapidMlxEngine(),
        OmlxEmbedEngine()
    ]

    print("=" * 80)
    print("  CONCURRENCY THROUGHPUT & SATURATION BENCHMARK (100 REQ / LEVEL)")
    print("=" * 80)
    for eng in engines:
        print(f"\n>> {eng.name}:")
        print(f"   {'Concurrency':<12} | {'RPS':<10} | {'P50 Latency':<14} | {'P95 Latency':<14} | {'Total Time':<10}")
        print("   " + "-" * 65)
        res = benchmark_engine_concurrency(eng, concurrency_levels=[1, 2, 4, 8], requests_per_level=100)
        for c, data in res["concurrency_results"].items():
            print(f"   c = {c:<8} | {data['rps']:6.1f} rps | {data['p50_latency_ms']:8.2f} ms    | {data['p95_latency_ms']:8.2f} ms    | {data['total_time_sec']:6.2f} s")
