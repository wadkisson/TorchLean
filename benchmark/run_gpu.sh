#!/usr/bin/env bash
# Full GPU benchmark suite (1k + 10k steps, all six runs).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 benchmark/run_benchmarks.py --device cuda --build "$@"
