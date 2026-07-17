#!/usr/bin/env bash
# Nsight Systems capture of TorchLean GPT-2 train steps on DGX.
# Focus: cudaMalloc / cudaFree thrash during forward (see cudaapisum).
#
# Usage (on DGX, after building):
#   BENCH_MAX_ITERS=3 ./benchmark/run_nsys_fwd.sh
#
# Old Nsight (2020.x) writes .qdrep + .sqlite; newer writes .nsys-rep.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ITERS="${BENCH_MAX_ITERS:-3}"
OUT="$ROOT/benchmark/out"
mkdir -p "$OUT"
REPORT="$OUT/fwd_nsys"

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

echo "==> build benchmark_gpt2"
lake -R -K cuda=true -K libtorch=true build benchmark_gpt2

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys not found; install Nsight Systems or add it to PATH" >&2
  exit 1
fi

echo "==> nsys profile (BENCH_MAX_ITERS=$ITERS) → $REPORT"
rm -f "$REPORT".qdrep "$REPORT".sqlite "$REPORT".nsys-rep

# Keep capture lean: CUDA API + GPU kernels. Skip NVTX/OSRT noise on old nsys.
BENCH_MAX_ITERS="$ITERS" \
BENCH_SAMPLE=0 \
  nsys profile \
    -o "$REPORT" \
    --force-overwrite=true \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    lake -R -K cuda=true -K libtorch=true exe benchmark_gpt2 --device cuda

echo "==> cuda API summary"
if [[ -f "$REPORT.sqlite" ]]; then
  nsys stats --report cudaapisum "$REPORT.sqlite" || nsys stats --report cuda_api_sum "$REPORT.sqlite" || true
elif [[ -f "$REPORT.nsys-rep" ]]; then
  nsys stats --report cuda_api_sum "$REPORT.nsys-rep" || true
else
  echo "no nsys report found at $REPORT.{sqlite,nsys-rep}" >&2
  exit 1
fi

echo "done: $REPORT.*"
