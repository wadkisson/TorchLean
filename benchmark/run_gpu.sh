#!/usr/bin/env bash
# TorchLean vs PyTorch benchmarks on CUDA (1000, 10000, 100000 steps).
#
# LeanProfiler (TorchLean only): set `leanProfilerEnabled := true` at the top of
#   NN/Examples/Models/Supervised/Mlp.lean
#   NN/Examples/Models/Vision/Cnn.lean
#   NN/Examples/Models/Sequence/Gpt2.lean
# then `lake build -K cuda=true` before running. Profiles: data/profiles/{mlp,cnn,gpt2}.json
set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS="benchmark/results.log"
: > "$RESULTS"
exec > >(tee -a "$RESULTS") 2>&1

python3 scripts/datasets/download_example_data.py --auto-mpg --cifar10 --tiny-shakespeare
lake build -K cuda=true

for STEPS in 1000 10000 100000; do
  echo
  echo "========== steps=$STEPS =========="

  echo "--- torchlean_mlp ---"
  time lake exe -K cuda=true torchlean mlp --cuda --fast-kernels --steps "$STEPS" --log false

  echo "--- torchlean_cnn ---"
  time lake exe -K cuda=true torchlean cnn --cuda --fast-kernels --steps "$STEPS" --log false

  echo "--- torchlean_gpt2 ---"
  time lake exe -K cuda=true torchlean gpt2 --cuda --fast-kernels --tiny-shakespeare --generate 0 --steps "$STEPS" --log false

  echo "--- pytorch_mlp ---"
  time python3 benchmark/pytorch/train_mlp.py --steps "$STEPS" --device cuda

  echo "--- pytorch_cnn ---"
  time python3 benchmark/pytorch/train_cnn.py --steps "$STEPS" --device cuda

  echo "--- pytorch_gpt2 ---"
  time python3 benchmark/pytorch/train_gpt2.py --steps "$STEPS" --device cuda
done

echo
echo "Wrote $RESULTS"
