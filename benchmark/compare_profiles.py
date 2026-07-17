#!/usr/bin/env python3
"""Compare LeanProfiler vs PyTorch span totals from a paired benchmark run.

Expects:
  benchmark/out/pytorch-spans.json
  LeanProfiler summary printed to a captured log, OR
  benchmark/out/torchlean-spans.json (optional normalized export)

LeanProfiler writes a Chrome trace at LEAN_PROFILE_OUT and a terminal summary.
This script reads the Chrome trace + pytorch-spans.json and prints a side-by-side table.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"


def load_pytorch(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    return {row["name"]: row for row in data["spans"]}


def load_chrome_trace(path: Path) -> dict[str, dict]:
    """Aggregate complete events (ph=X) by name — LeanProfiler (ns) or our PyTorch export (us)."""
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and "traceEvents" in raw:
        events = raw["traceEvents"]
        unit = raw.get("displayTimeUnit", "us")
    else:
        events = raw
        unit = "us"
    # Convert duration to milliseconds.
    scale = 1e-6 if unit == "ns" else 1e-3
    totals_ms: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for ev in events:
        if not isinstance(ev, dict) or ev.get("ph") != "X":
            continue
        name = ev.get("name")
        if not name:
            continue
        totals_ms[name] += float(ev.get("dur", 0.0)) * scale
        counts[name] += 1
    out: dict[str, dict] = {}
    for name, total_ms in totals_ms.items():
        out[name] = {
            "name": name,
            "total_ms": total_ms,
            "count": counts[name],
            "mean_ms": total_ms / max(counts[name], 1),
        }
    return out


def main() -> int:
    pt_path = OUT / "pytorch-spans.json"
    tl_trace = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "torchlean-trace.json"

    if not pt_path.exists():
        print(f"missing {pt_path}; run PROFILE=1 python3 benchmark/train_gpt2.py first", file=sys.stderr)
        return 1
    if not tl_trace.exists():
        print(
            f"missing {tl_trace}; run LEAN_PROFILE=1 LEAN_PROFILE_OUT={tl_trace} "
            "lake -R -K cuda=true exe benchmark_gpt2 --device cuda",
            file=sys.stderr,
        )
        return 1

    pt = load_pytorch(pt_path)
    tl = load_chrome_trace(tl_trace)
    # Drop LeanProfiler's top-level main wrapper if present — compare shared span names.
    names = sorted(set(pt) | set(tl) - {"main"})
    print(f"{'span':<24} {'pt_ms':>12} {'tl_ms':>12} {'tl/pt':>10} {'pt_n':>6} {'tl_n':>6}")
    print("-" * 76)
    for name in names:
        p = pt.get(name)
        t = tl.get(name)
        p_ms = p["total_ms"] if p else float("nan")
        t_ms = t["total_ms"] if t else float("nan")
        ratio = (t_ms / p_ms) if p and t and p_ms > 0 else float("nan")
        p_n = p["count"] if p else 0
        t_n = t["count"] if t else 0
        print(f"{name:<24} {p_ms:>12.2f} {t_ms:>12.2f} {ratio:>10.2f} {p_n:>6} {t_n:>6}")
    print()
    print("tl/pt > 1 means TorchLean host span is slower than CUDA-synced PyTorch span.")
    print(f"pytorch: {pt_path}")
    print(f"torchlean trace: {tl_trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
