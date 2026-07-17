#!/usr/bin/env bash
# Profile TorchLean (LeanProfiler) vs PyTorch on the gpt2-500m 7×-era config
# (32L × 1024d, seq 128, vocab 50257, batch 1). Baseline then: TL ~604 ms, PT ~87 ms.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Keep this tiny: each iter is a full ~500M fwd+bwd+AdamW. Short runs also force accum=1.
ITERS="${BENCH_MAX_ITERS:-3}"
OUT="$ROOT/benchmark/out"
mkdir -p "$OUT"

gpu_mem() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "==> GPU memory"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits \
      | awk -F', ' '{printf "  gpu%s %s: used=%s MiB free=%s MiB / %s MiB\n", $1, $2, $4, $5, $3}'
  fi
}

echo "==> ensuring shakespeare data"
python3 - <<'PY'
from pathlib import Path
import urllib.request
p = Path("benchmark/data/tinyshakespeare.txt")
p.parent.mkdir(parents=True, exist_ok=True)
if not p.exists():
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        p,
    )
print(p, p.stat().st_size, "bytes")
PY

echo "==> build TorchLean benchmark_gpt2"
lake -R -K cuda=true -K libtorch=true build benchmark_gpt2

# TorchLean first: needs the most free VRAM (fp32 + untied LM head + AdamW moments).
# Running PyTorch first left the machine looking fine after process exit on some runs, but
# we still hit cuda OOM on the first TL train.step — start TL on a clean-ish GPU.
gpu_mem
echo "==> profile TorchLean / LeanProfiler (BENCH_MAX_ITERS=$ITERS)"
LEAN_PROFILE=1 \
BENCH_MAX_ITERS="$ITERS" \
LEAN_PROFILE_OUT="$OUT/torchlean-trace.json" \
  lake -R -K cuda=true -K libtorch=true exe benchmark_gpt2 --device cuda

gpu_mem
echo "==> profile PyTorch (BENCH_MAX_ITERS=$ITERS)"
PROFILE=1 BENCH_MAX_ITERS="$ITERS" python3 benchmark/train_gpt2.py

echo "==> compare"
python3 benchmark/compare_profiles.py "$OUT/torchlean-trace.json"

echo "HTML report (if LeanProfiler wrote one): ${OUT}/torchlean-trace.json.html"
echo "Open Perfetto traces: https://ui.perfetto.dev"
