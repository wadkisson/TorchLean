#!/usr/bin/env python3
"""Train the Auto MPG MLP benchmark (PyTorch mirror of `torchlean mlp`).

LeanProfiler applies to the TorchLean runner only — see `Mlp.lean` (`leanProfilerEnabled`).
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

from benchmark.config import MLP_LR, MLP_SEED
from benchmark.pytorch.data import iter_auto_mpg_batches
from benchmark.pytorch.models import TorchLeanMlp


def train(steps: int, device: str) -> float:
    torch.manual_seed(MLP_SEED)
    model = TorchLeanMlp().to(device)
    optimizer = optim.Adam(model.parameters(), lr=MLP_LR)
    batches = iter_auto_mpg_batches()

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        x, y = next(batches)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = torch.mean((model(x) - y) ** 2)
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
    print(f"pytorch_mlp steps={args.steps} device={args.device} seconds={elapsed:.3f}")


if __name__ == "__main__":
    main()
