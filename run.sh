#!/usr/bin/env bash
set -e

# Activate venv if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run test suite
echo "[1/3] Running Sampler Unit Tests..."
python3 -m unittest discover tests

# Run benchmark
echo "[2/3] Executing Benchmark..."
python3 run_benchmark.py "$@"

echo "[3/3] Done!"
