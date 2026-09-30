import time
import json
import statistics
import concurrent.futures
import urllib.request
from pathlib import Path
import numpy as np
import laya_mlx

# Load Native Python SDK Agent
model_path = str(
    Path.home() / ".cache/huggingface/hub/models--aac6fef--laya-multilingual-mlx/snapshots/f2b4faf51023039425946074e2cf1361d2db11d5"
)
agent = laya_mlx.Agent(model_path)

RAPID_URL = "http://127.0.0.1:8700/v1/systemone"

# Test Payload
categories = ["科技数码", "萌宠动物", "美妆护肤", "穿搭时尚", "美食探店", "职场成长", "家居生活", "运动健康", "母婴育儿", "旅游出行", "情感心理", "摄影剪辑"]
state = "小红书笔记标题: 富士山🗻初雪，终于拍到人生照片\n正文片段: 河口湖机位绝了，清晨无风倒影"
questions = {
    "track": {
        "type": "choice",
        "instructions": "判断小红书笔记的垂直行业分类赛道",
        "criteria": {c: None for c in categories}
    }
}

payload_json = json.dumps({"state": state, "questions": questions}).encode("utf-8")

def call_native():
    t0 = time.perf_counter()
    res = agent.predict(state, questions)
    elapsed = (time.perf_counter() - t0) * 1000
    return elapsed, res["answers"]["track"]["choice"]

def call_rapid():
    t0 = time.perf_counter()
    req = urllib.request.Request(
        RAPID_URL,
        data=payload_json,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = (time.perf_counter() - t0) * 1000
    return elapsed, data["answers"]["track"]["choice"]

def run_benchmark(name, fn, iterations=50):
    # Warmup
    for _ in range(5):
        fn()
    
    latencies = []
    for _ in range(iterations):
        elapsed, _ = fn()
        latencies.append(elapsed)
    
    return {
        "mean": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99),
        "min": min(latencies),
        "max": max(latencies),
        "std": statistics.stdev(latencies) if len(latencies) > 1 else 0
    }

print("=" * 70)
print("  BENCHMARK 1: 单请求串行延迟 (Sequential Latency, 50 次循环)")
print("=" * 70)

native_stats = run_benchmark("Native Python SDK", call_native, 50)
rapid_stats = run_benchmark("Rapid-MLX HTTP API", call_rapid, 50)

print(f"{'指标':<18} | {'Native Python SDK':<18} | {'Rapid-MLX (8700)':<18} | 差异")
print("-" * 70)
for k, label in [("mean", "平均耗时 (Mean)"), ("median", "中位数 (Median)"), ("p95", "P95 延迟"), ("p99", "P99 延迟"), ("min", "最小耗时 (Min)"), ("max", "最大耗时 (Max)"), ("std", "标准差 (StdDev)")]:
    nv = native_stats[k]
    rv = rapid_stats[k]
    diff = f"{rv - nv:+6.2f} ms"
    print(f"{label:<18} | {nv:8.2f} ms         | {rv:8.2f} ms         | {diff}")

print("\n" + "=" * 70)
print("  BENCHMARK 2: 小红书 61 篇验证集全量真实推断 (Real-world Val Set)")
print("=" * 70)

with open("data/val_61.json") as f:
    val_data = json.load(f)

# Native Val Set
t0 = time.perf_counter()
for item in val_data:
    s = f"小红书笔记标题: {item['title']}\n正文片段: {item.get('snippet', '')}"
    agent.predict(s, questions)
native_val_time = time.perf_counter() - t0
native_qps = len(val_data) / native_val_time

# Rapid Val Set
t0 = time.perf_counter()
for item in val_data:
    s = f"小红书笔记标题: {item['title']}\n正文片段: {item.get('snippet', '')}"
    req = urllib.request.Request(
        RAPID_URL,
        data=json.dumps({"state": s, "questions": questions}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        resp.read()
rapid_val_time = time.perf_counter() - t0
rapid_qps = len(val_data) / rapid_val_time

print(f"Native Python SDK: 总耗时 {native_val_time:6.2f}s | 吞吐量: {native_qps:5.1f} notes/sec")
print(f"Rapid-MLX API:     总耗时 {rapid_val_time:6.2f}s | 吞吐量: {rapid_qps:5.1f} notes/sec")

print("\n" + "=" * 70)
print("  BENCHMARK 3: 高并发吞吐压测 (Concurrent Benchmark, 100 requests)")
print("=" * 70)

for concurrency in [4, 8]:
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(call_rapid) for _ in range(100)]
        results = [f.result() for f in futures]
    c_time = time.perf_counter() - t0
    c_latencies = [r[0] for r in results]
    print(f"Rapid-MLX [并发={concurrency:2d}] -> 总耗时: {c_time:5.2f}s | RPS: {100/c_time:5.1f} req/s | 平均延迟: {statistics.mean(c_latencies):6.2f}ms | P95: {np.percentile(c_latencies, 95):6.2f}ms")

