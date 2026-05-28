#!/usr/bin/env bash
# TorchLean vs PyTorch benchmarks on CUDA (1000 + 10000 steps, 12 timed runs).
set -euo pipefail
cd "$(dirname "$0")/.."

python3 scripts/datasets/download_example_data.py --auto-mpg --cifar10 --tiny-shakespeare
lake build -K cuda=true

for STEPS in 1000 10000; do
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
