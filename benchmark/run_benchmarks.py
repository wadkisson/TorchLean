#!/usr/bin/env python3
"""
Run TorchLean vs PyTorch training benchmarks for MLP, CNN, and GPT-2.

Each of the six configurations is timed individually at 1k and 10k optimizer steps.
Results are written to benchmark/results.json and printed as a table.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.config import STEP_COUNTS

BENCHMARK_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BENCHMARK_DIR / "results.json"


@dataclass
class RunSpec:
    name: str
    framework: str
    model: str
    kind: str  # "subprocess" | "python"


@dataclass
class TimedRun:
    name: str
    framework: str
    model: str
    steps: int
    seconds: float
    ok: bool
    detail: str


RUNS: list[RunSpec] = [
    RunSpec("torchlean_mlp", "torchlean", "mlp", "subprocess"),
    RunSpec("torchlean_cnn", "torchlean", "cnn", "subprocess"),
    RunSpec("torchlean_gpt2", "torchlean", "gpt2", "subprocess"),
    RunSpec("pytorch_mlp", "pytorch", "mlp", "python"),
    RunSpec("pytorch_cnn", "pytorch", "cnn", "python"),
    RunSpec("pytorch_gpt2", "pytorch", "gpt2", "python"),
]


def torchlean_cmd(model: str, steps: int, device: str) -> list[str]:
    lake = ["lake", "exe"]
    if device == "cuda":
        lake.append("-K")
        lake.append("cuda=true")
    lake.append("torchlean")
    cmd = lake + [model, f"--{device}", "--steps", str(steps), "--log", "false"]
    if device == "cuda":
        cmd.append("--fast-kernels")
    if model == "gpt2":
        cmd.extend(["--tiny-shakespeare", "--generate", "0"])
    return cmd


def pytorch_cmd(model: str, steps: int, device: str) -> list[str]:
    script = BENCHMARK_DIR / "pytorch" / f"train_{model}.py"
    return [sys.executable, str(script), "--steps", str(steps), "--device", device]


def run_subprocess(cmd: list[str], cwd: Path) -> tuple[bool, float, str]:
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
    )
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        return False, elapsed, detail
    return True, elapsed, (proc.stdout or "").strip()


def run_python(cmd: list[str], cwd: Path) -> tuple[bool, float, str]:
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
    )
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        return False, elapsed, detail
    return True, elapsed, (proc.stdout or "").strip()


def check_cuda_available() -> None:
    try:
        import torch
    except ImportError as e:
        raise SystemExit("PyTorch is required for CUDA benchmarks: pip install torch") from e
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested but PyTorch reports no GPU.\n"
            "Check nvidia-smi, CUDA drivers, and install a CUDA-enabled torch wheel."
        )
    print(f"PyTorch CUDA device: {torch.cuda.get_device_name(0)}")


def build_torchlean(device: str) -> None:
    cmd = ["lake", "build"]
    if device == "cuda":
        cmd = ["lake", "build", "-K", "cuda=true"]
    print(f"Building TorchLean ({'CUDA' if device == 'cuda' else 'CPU'})...")
    subprocess.check_call(cmd, cwd=REPO_ROOT)


def ensure_data() -> None:
    prep = REPO_ROOT / "scripts" / "datasets" / "download_example_data.py"
    missing = []
    for rel in (
        "data/real/auto_mpg/auto_mpg.csv",
        "data/real/cifar10/cifar10_train_X.npy",
        "data/real/text/tiny_shakespeare.txt",
    ):
        if not (REPO_ROOT / rel).exists():
            missing.append(rel)
    if not missing:
        return
    print("Preparing missing benchmark datasets...")
    subprocess.check_call(
        [sys.executable, str(prep), "--auto-mpg", "--cifar10", "--tiny-shakespeare"],
        cwd=REPO_ROOT,
    )


def run_one(spec: RunSpec, steps: int, device: str) -> TimedRun:
    if spec.kind == "subprocess":
        cmd = torchlean_cmd(spec.model, steps, device)
        ok, elapsed, detail = run_subprocess(cmd, REPO_ROOT)
    else:
        cmd = pytorch_cmd(spec.model, steps, device)
        ok, elapsed, detail = run_python(cmd, REPO_ROOT)

    return TimedRun(
        name=spec.name,
        framework=spec.framework,
        model=spec.model,
        steps=steps,
        seconds=elapsed,
        ok=ok,
        detail=detail,
    )


def print_table(results: list[TimedRun]) -> None:
    print()
    print(f"{'run':<18} {'steps':>6} {'seconds':>10} {'status':>8}")
    print("-" * 46)
    for row in results:
        status = "ok" if row.ok else "failed"
        print(f"{row.name:<18} {row.steps:>6} {row.seconds:>10.3f} {status:>8}")
        if not row.ok and row.detail:
            for line in row.detail.splitlines()[:6]:
                print(f"  {line}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
        help="Device for both TorchLean and PyTorch runs (default: cuda).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="*",
        default=list(STEP_COUNTS),
        help="Step counts to benchmark (default: 1000 10000).",
    )
    parser.add_argument(
        "--only",
        choices=[r.name for r in RUNS],
        nargs="*",
        help="Run a subset of the six benchmarks.",
    )
    parser.add_argument(
        "--skip-data-prep",
        action="store_true",
        help="Do not auto-download missing example datasets.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Run `lake build` (or `lake build -K cuda=true`) before benchmarks.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick sanity check: 10 steps only (use on CPU before the full 1k/10k suite).",
    )
    args = parser.parse_args()

    specs = RUNS
    if args.only:
        wanted = set(args.only)
        specs = [r for r in RUNS if r.name in wanted]

    if args.smoke:
        args.steps = [10]

    if args.device == "cuda":
        check_cuda_available()

    if not args.skip_data_prep:
        ensure_data()

    if args.build:
        build_torchlean(args.device)

    results: list[TimedRun] = []
    for spec in specs:
        for steps in args.steps:
            cmd = (
                torchlean_cmd(spec.model, steps, args.device)
                if spec.kind == "subprocess"
                else pytorch_cmd(spec.model, steps, args.device)
            )
            print(f"Running {spec.name} ({steps} steps on {args.device})...")
            print(f"  cmd: {' '.join(cmd)}")
            results.append(run_one(spec, steps, args.device))

    RESULTS_PATH.write_text(json.dumps([asdict(r) for r in results], indent=2) + "\n")
    print_table(results)
    print(f"\nWrote {RESULTS_PATH}")
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
