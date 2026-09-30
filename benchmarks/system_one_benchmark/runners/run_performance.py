import json
import time
from typing import Any, Dict, List
from benchmarks.system_one_benchmark.engines import BaseDecisionEngine, NativeLayaEngine, RapidMlxEngine, OmlxEmbedEngine
from benchmarks.system_one_benchmark.metrics import compute_performance_stats

CATEGORIES = [
    "科技数码", "萌宠动物", "美妆护肤", "穿搭时尚", "美食探店", "职场成长",
    "家居生活", "运动健康", "母婴育儿", "旅游出行", "情感心理", "摄影剪辑"
]

def profile_engine_latency(engine: BaseDecisionEngine, rounds: int = 100, warmup: int = 5) -> Dict[str, Any]:
    state = "小红书笔记标题: 富士山🗻初雪，终于拍到人生照片\n正文片段: 河口湖机位绝了，清晨无风倒影"
    questions = {
        "track": {
            "type": "choice",
            "instructions": "判断小红书笔记的垂直行业分类赛道",
            "criteria": {c: None for c in CATEGORIES}
        }
    }

    # Cold start test
    t_cold_0 = time.perf_counter()
    engine.predict(state, questions)
    cold_latency = (time.perf_counter() - t_cold_0) * 1000

    # Warmup
    for _ in range(warmup):
        engine.predict(state, questions)

    # Measured runs
    latencies = []
    t_start = time.perf_counter()
    for _ in range(rounds):
        t0 = time.perf_counter()
        engine.predict(state, questions)
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
    total_time = time.perf_counter() - t_start

    stats = compute_performance_stats(latencies, total_time)
    stats["cold_latency_ms"] = round(cold_latency, 2)
    stats["engine_name"] = engine.name
    return stats

if __name__ == "__main__":
    engines = [
        NativeLayaEngine(),
        RapidMlxEngine(),
        OmlxEmbedEngine()
    ]

    print("=" * 80)
    print("  LATENCY & THROUGHPUT PROFILING (100 ITERATIONS)")
    print("=" * 80)
    print(f"{'Engine':<32} | {'Cold(ms)':<8} | {'Mean':<8} | {'P50':<8} | {'P95':<8} | {'P99':<8} | {'RPS':<6}")
    print("-" * 80)
    for eng in engines:
        res = profile_engine_latency(eng, rounds=100)
        print(f"{res['engine_name']:<32} | {res['cold_latency_ms']:8.1f} | {res['mean_ms']:6.2f}ms | {res['median_ms']:6.2f}ms | {res['p95_ms']:6.2f}ms | {res['p99_ms']:6.2f}ms | {res['rps']:5.1f}")
