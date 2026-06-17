"""Shared helpers for the small LiRPA certificate producers.

These functions are intentionally tiny.  The exporter scripts still spell out the model-specific
graph and parameters; this module only keeps the boring interval arithmetic and JSON writing in one
place so the fixture producers do not drift apart.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def centered_box(center: list[float], eps: float) -> tuple[list[float], list[float]]:
    """Return the interval box `[center - eps, center + eps]` coordinatewise."""

    return [x - eps for x in center], [x + eps for x in center]


def affine_interval(
    weights: list[list[float]],
    bias: list[float],
    lo: list[float],
    hi: list[float],
) -> tuple[list[float], list[float]]:
    """Propagate interval bounds through `weights @ x + bias`."""

    out_lo: list[float] = []
    out_hi: list[float] = []
    for row, b in zip(weights, bias):
        lo_i = b
        hi_i = b
        for a, x_lo, x_hi in zip(row, lo, hi):
            p = a * x_lo
            q = a * x_hi
            lo_i += min(p, q)
            hi_i += max(p, q)
        out_lo.append(lo_i)
        out_hi.append(hi_i)
    return out_lo, out_hi


def matmul_interval(
    weights: list[list[float]],
    lo: list[float],
    hi: list[float],
) -> tuple[list[float], list[float]]:
    """Propagate interval bounds through a bias-free matrix multiply."""

    return affine_interval(weights, [0.0 for _ in weights], lo, hi)


def relu_interval(lo: list[float], hi: list[float]) -> tuple[list[float], list[float]]:
    """Propagate interval bounds through elementwise ReLU."""

    return [max(0.0, x) for x in lo], [max(0.0, x) for x in hi]


def softmax_interval(lo: list[float], hi: list[float]) -> tuple[list[float], list[float]]:
    """Compute conservative elementwise bounds for softmax over one vector interval."""

    elo = [math.exp(x) for x in lo]
    ehi = [math.exp(x) for x in hi]
    total_lo = sum(elo)
    total_hi = sum(ehi)
    out_lo: list[float] = []
    out_hi: list[float] = []
    for i in range(len(lo)):
        out_lo.append(elo[i] / (elo[i] + (total_hi - ehi[i])))
        out_hi.append(ehi[i] / (ehi[i] + (total_lo - elo[i])))
    return out_lo, out_hi


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write one certificate JSON payload with stable indentation."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return out
