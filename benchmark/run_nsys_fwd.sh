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

# Keep capture lean for old Nsight (DGX has 2020.x): avoid --cpuctxsw / --sample flags.
BENCH_MAX_ITERS="$ITERS" \
BENCH_SAMPLE=0 \
  nsys profile \
    -o "$REPORT" \
    --force-overwrite=true \
    --trace=cuda \
    lake -R -K cuda=true -K libtorch=true exe benchmark_gpt2 --device cuda

echo "==> cuda API summary"
# Old Nsight (2020.3.2) often fails event processing but still writes a usable .qdrep;
# `nsys stats` on the .qdrep regenerates .sqlite.
REP=""
if [[ -f "$REPORT.qdrep" ]]; then
  REP="$REPORT.qdrep"
elif [[ -f "$REPORT.sqlite" ]]; then
  REP="$REPORT.sqlite"
elif [[ -f "$REPORT.nsys-rep" ]]; then
  REP="$REPORT.nsys-rep"
else
  echo "no nsys report found at $REPORT.{qdrep,sqlite,nsys-rep}" >&2
  exit 1
fi
nsys stats --report cudaapisum "$REP" 2>&1 | tee "$OUT/cudaapisum.txt" \
  || nsys stats --report cuda_api_sum "$REP" 2>&1 | tee "$OUT/cudaapisum.txt" \
  || true

echo "done: $REPORT.*"
