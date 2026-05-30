#!/usr/bin/env python3
"""Train the CIFAR CNN benchmark (PyTorch mirror of `torchlean cnn`).

LeanProfiler applies to the TorchLean runner only — see `Cnn.lean` (`leanProfilerEnabled`).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.optim as optim

from benchmark.config import CNN_LR, CNN_SEED
from benchmark.pytorch.data import iter_cifar_batches
from benchmark.pytorch.models import TorchLeanCnn, cross_entropy_one_hot_mean


def train(steps: int, device: str) -> float:
    torch.manual_seed(CNN_SEED)
    model = TorchLeanCnn().to(device)
    optimizer = optim.Adam(model.parameters(), lr=CNN_LR)
    batches = iter_cifar_batches()

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        x, y = next(batches)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = cross_entropy_one_hot_mean(model(x), y)
        loss.backward()
        optimizer.step()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    elapsed = train(args.steps, args.device)
    print(f"pytorch_cnn steps={args.steps} device={args.device} seconds={elapsed:.3f}")


if __name__ == "__main__":
    main()
