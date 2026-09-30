#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"

echo "=========================================================="
echo " Starting System One / DecisionModel Benchmark Suite"
echo " Working Directory: $ROOT"
echo "=========================================================="

export PYTHONPATH="$ROOT:$PYTHONPATH"
cd "$ROOT"

.venv/bin/python -m benchmarks.system_one_benchmark.runners.run_all

echo "=========================================================="
echo " Benchmark Complete! View report at:"
echo " benchmarks/system_one_benchmark/reports/BENCHMARK_REPORT.md"
echo "=========================================================="
