#!/usr/bin/env python3
"""One-file GPT-2 (~500M): build, train on GPU, sample every so often.

Matching LeanProfiler spans (set PROFILE=1 or LEAN_PROFILE=1):
  load.data, init.model, sample.generate, sample.embed_gather/sample.forward (Lean-only),
  train.batch, train.step, train.loss_eval

Outputs (when profiling):
  benchmark/out/pytorch-spans.json   — per-span totals (ms)
  benchmark/out/pytorch-trace.json   — Chrome/Perfetto-style events
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.request
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- config: last week's ~7× gpt2-500m (TL ~604 ms vs PT ~87 ms) ---
VOCAB = 50257
BLOCK = 128
N_LAYER = 32
N_HEAD = 16
N_EMBD = 1024
BATCH = 1  # match TorchLean / 7×-era harness
ACCUM = 8
LR = 3e-4
MAX_ITERS_DEFAULT = 2000
SAMPLE_EVERY = 25
MAX_NEW = 120
DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "out"
SHAKESPEARE = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

assert N_EMBD % N_HEAD == 0
HEAD_DIM = N_EMBD // N_HEAD


def _env_on(name: str) -> bool:
    v = os.environ.get(name)
    return v is not None and v not in ("", "0", "false")


PROFILE = _env_on("PROFILE") or _env_on("LEAN_PROFILE")
MAX_ITERS = int(os.environ.get("BENCH_MAX_ITERS", MAX_ITERS_DEFAULT))


def _do_sample() -> bool:
    v = os.environ.get("BENCH_SAMPLE")
    if v in ("1", "true"):
        return True
    if v in ("0", "false"):
        return False
    # Short profile runs: skip iter-0 120-token generate (matches TorchLean).
    return MAX_ITERS >= SAMPLE_EVERY


def _accum() -> int:
    raw = os.environ.get("BENCH_ACCUM")
    if raw is not None:
        n = int(raw)
        if n < 1:
            raise SystemExit("BENCH_ACCUM must be ≥ 1")
        return n
    # Short runs: 1 opt step per iter (otherwise 10 iters ⇒ 80 full AdamW steps).
    return 1 if MAX_ITERS < SAMPLE_EVERY else ACCUM


class SpanProfiler:
    """Host-side spans with CUDA sync — same names as LeanProfiler in TrainGpt2.lean."""

    def __init__(self) -> None:
        self.enabled = PROFILE
        self.totals_ms: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self.events: list[dict] = []
        self.t0 = time.perf_counter()

    @contextmanager
    def span(self, name: str):
        if not self.enabled:
            yield
            return
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.totals_ms[name] += dt_ms
            self.counts[name] += 1
            start_us = int((t0 - self.t0) * 1e6)
            dur_us = int(dt_ms * 1000.0)
            self.events.append(
                {
                    "name": name,
                    "cat": "bench",
                    "ph": "X",
                    "ts": start_us,
                    "dur": max(dur_us, 1),
                    "pid": 1,
                    "tid": 1,
                }
            )

    def finish(self, out_dir: Path = OUT_DIR) -> None:
        if not self.enabled:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = sorted(self.totals_ms.items(), key=lambda kv: -kv[1])
        summary = {
            "backend": "pytorch",
            "max_iters": MAX_ITERS,
            "spans": [
                {
                    "name": name,
                    "total_ms": total,
                    "count": self.counts[name],
                    "mean_ms": total / max(self.counts[name], 1),
                }
                for name, total in rows
            ],
        }
        spans_path = out_dir / "pytorch-spans.json"
        trace_path = out_dir / "pytorch-trace.json"
        spans_path.write_text(json.dumps(summary, indent=2))
        trace_path.write_text(json.dumps({"traceEvents": self.events}))
        print("\n=== PyTorch span summary (CUDA-synced) ===")
        print(f"{'span':<24} {'count':>8} {'total_ms':>12} {'mean_ms':>12}")
        for row in summary["spans"]:
            print(
                f"{row['name']:<24} {row['count']:>8} {row['total_ms']:>12.2f} {row['mean_ms']:>12.2f}"
            )
        print(f"wrote {spans_path}")
        print(f"wrote {trace_path}")


PROF = SpanProfiler()


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(N_EMBD, dim=2)
        q = q.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
        k = k.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
        v = v.view(B, T, N_HEAD, HEAD_DIM).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * N_EMBD, N_EMBD),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB, N_EMBD)
        self.wpe = nn.Embedding(BLOCK, N_EMBD)
        self.h = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tie
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.h:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=MAX_NEW, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK:]
            with PROF.span("sample.forward"):
                logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx


def load_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "tinyshakespeare.txt"
    if not path.exists():
        print(f"downloading {path.name}...")
        urllib.request.urlretrieve(SHAKESPEARE, path)
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8).astype(np.int64)
    n = int(0.9 * len(data))
    return data[:n], data[n:]


def get_batch(data, device):
    ix = torch.randint(len(data) - BLOCK - 1, (BATCH,))
    x = torch.stack([torch.from_numpy(data[i : i + BLOCK]) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + BLOCK]) for i in ix])
    return x.to(device), y.to(device)


def decode(ids):
    return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")


def main():
    if not torch.cuda.is_available():
        raise SystemExit("need a CUDA GPU")
    device = torch.device("cuda")
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)

    with PROF.span("load.data"):
        train_data, _ = load_data()

    with PROF.span("init.model"):
        model = GPT().to(device)
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
        torch.cuda.synchronize()
    do_sample = _do_sample()
    run_accum = _accum()
    print(f"GPT-2 on {device}: {n_params / 1e6:.1f}M params | train {MAX_ITERS} iters | accum {run_accum}")
    if PROFILE:
        print("profiling ON (CUDA-synced spans → benchmark/out/pytorch-*.json)")
    if not do_sample:
        print("sampling OFF for short run (set BENCH_SAMPLE=1 to force)")

    ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)

    model.train()
    t_all0 = time.perf_counter()
    for it in range(MAX_ITERS + 1):
        if do_sample and it % SAMPLE_EVERY == 0:
            model.eval()
            with PROF.span("sample.generate"):
                prompt = torch.tensor([[ord(c) for c in "ROMEO:"]], dtype=torch.long, device=device)
                out = model.generate(prompt)
                torch.cuda.synchronize()
            print(f"\n=== iter {it} ===\n{decode(out[0].tolist())}\n")
            model.train()
            if it == MAX_ITERS:
                break
        elif it == MAX_ITERS:
            break

        if it < 50:
            lr = LR * (it + 1) / 50
        else:
            progress = (it - 50) / max(MAX_ITERS - 50, 1)
            lr = LR * 0.1 + 0.5 * (LR - LR * 0.1) * (1 + math.cos(math.pi * progress))
        for g in opt.param_groups:
            g["lr"] = lr

        t0 = time.perf_counter()
        with PROF.span("train.step"):
            opt.zero_grad(set_to_none=True)
            for _ in range(run_accum):
                with PROF.span("train.batch"):
                    x, y = get_batch(train_data, device)
                with ctx:
                    _, loss = model(x, y)
                    loss = loss / run_accum
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        print(f"iter {it:5d}  train {dt_ms:.2f} ms  ({run_accum}× step)  lr {lr:.2e}")

        if MAX_ITERS >= SAMPLE_EVERY and it % 10 == 0:
            with PROF.span("train.loss_eval"):
                loss_val = loss.item() * run_accum
                torch.cuda.synchronize()
            print(f"iter {it:5d}  loss {loss_val:.4f}")

    print(f"total wall {(time.perf_counter() - t_all0) * 1000.0:.2f} ms")
    PROF.finish()


if __name__ == "__main__":
    main()
