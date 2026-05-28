"""Data loaders mirroring TorchLean example datasets."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

from benchmark.config import (
    AUTO_MPG_CSV,
    CIFAR_X,
    CIFAR_Y,
    CNN_BATCH,
    CNN_N_ROWS,
    CNN_OUT_DIM,
    GPT2_BATCH,
    GPT2_PAD_ID,
    GPT2_PROMPT,
    GPT2_SEQ_LEN,
    GPT2_VOCAB,
    GPT2_WINDOWS,
    MLP_BATCH,
    MLP_IN_DIM,
    MLP_OUT_DIM,
    TINY_SHAKESPEARE,
)


def _require(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}\n{hint}")


def load_auto_mpg_batches(
    csv_path: Path = AUTO_MPG_CSV,
    batch_size: int = MLP_BATCH,
    seed: int = 0,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    _require(
        csv_path,
        "Run: python3 scripts/datasets/download_example_data.py --auto-mpg",
    )
    rows: list[tuple[list[float], list[float]]] = []
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            vals = [float(x) for x in row]
            rows.append((vals[:MLP_IN_DIM], vals[MLP_IN_DIM : MLP_IN_DIM + MLP_OUT_DIM]))

    rng = random.Random(seed)
    rng.shuffle(rows)

    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for start in range(0, len(rows) - batch_size + 1, batch_size):
        chunk = rows[start : start + batch_size]
        xs = torch.tensor([x for x, _ in chunk], dtype=torch.float32)
        ys = torch.tensor([y for _, y in chunk], dtype=torch.float32)
        batches.append((xs, ys))
    if not batches:
        raise RuntimeError("Auto MPG CSV is too small for the configured batch size.")
    return batches


def iter_auto_mpg_batches(
    csv_path: Path = AUTO_MPG_CSV,
    batch_size: int = MLP_BATCH,
    seed: int = 0,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    batches = load_auto_mpg_batches(csv_path, batch_size, seed)
    idx = 0
    while True:
        yield batches[idx % len(batches)]
        idx += 1


def load_cifar_batches(
    x_path: Path = CIFAR_X,
    y_path: Path = CIFAR_Y,
    n_rows: int = CNN_N_ROWS,
    batch_size: int = CNN_BATCH,
    seed: int = 0,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    _require(
        x_path,
        "Run: python3 scripts/datasets/download_example_data.py --cifar10",
    )
    _require(y_path, "Run: python3 scripts/datasets/download_example_data.py --cifar10")

    x = np.load(x_path)[:n_rows].astype(np.float32)
    y_idx = np.load(y_path)[:n_rows].astype(np.int64)
    order = np.arange(n_rows)
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    x = x[order]
    y_idx = y_idx[order]

    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for start in range(0, n_rows - batch_size + 1, batch_size):
        xb = torch.from_numpy(x[start : start + batch_size])
        yb = F.one_hot(
            torch.from_numpy(y_idx[start : start + batch_size]),
            num_classes=CNN_OUT_DIM,
        ).float()
        batches.append((xb, yb))
    if not batches:
        raise RuntimeError("CIFAR slice is too small for the configured batch size.")
    return batches


def iter_cifar_batches(
    x_path: Path = CIFAR_X,
    y_path: Path = CIFAR_Y,
    n_rows: int = CNN_N_ROWS,
    batch_size: int = CNN_BATCH,
    seed: int = 0,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    batches = load_cifar_batches(x_path, y_path, n_rows, batch_size, seed)
    idx = 0
    while True:
        yield batches[idx % len(batches)]
        idx += 1


def byte_encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def token_window(tokens: list[int], length: int, offset: int, pad_id: int = GPT2_PAD_ID) -> list[int]:
    return [tokens[offset + i] if offset + i < len(tokens) else pad_id for i in range(length)]


def find_prompt_offset(tokens: list[int], prompt_tokens: list[int]) -> int | None:
    if not prompt_tokens:
        return None
    plen = len(prompt_tokens)
    for i in range(len(tokens) - plen + 1):
        if tokens[i : i + plen] == prompt_tokens:
            return i
    return None


def prompt_aware_offsets(token_count: int, seq_len: int, windows: int, prompt_offset: int | None) -> list[int]:
    usable = max(1, token_count - seq_len - 1) if token_count > seq_len + 1 else 1
    if prompt_offset is None:
        return [((i * seq_len) % usable) for i in range(windows)]
    start = prompt_offset - windows // 4 if prompt_offset > windows // 4 else 0
    return [((start + i) % usable) for i in range(windows)]


def causal_lm_xy_one_hot(window: list[int], seq_len: int, vocab: int, pad_id: int = GPT2_PAD_ID):
    need = seq_len + 1
    padded = window + [pad_id] * max(0, need - len(window))
    padded = padded[:need]
    x = torch.zeros(seq_len, vocab, dtype=torch.float32)
    y = torch.zeros(seq_len, vocab, dtype=torch.float32)
    for t in range(seq_len):
        x[t, min(padded[t], vocab - 1)] = 1.0
        y[t, min(padded[t + 1], vocab - 1)] = 1.0
    return x, y


def build_gpt2_samples(
    text_path: Path = TINY_SHAKESPEARE,
    prompt: str = GPT2_PROMPT,
    windows: int = GPT2_WINDOWS,
    batch_size: int = GPT2_BATCH,
    seq_len: int = GPT2_SEQ_LEN,
    vocab: int = GPT2_VOCAB,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    _require(
        text_path,
        "Run: python3 scripts/datasets/download_example_data.py --tiny-shakespeare",
    )
    text = text_path.read_text(encoding="utf-8")
    toks = byte_encode(text)
    prompt_toks = byte_encode(prompt)
    prompt_offset = find_prompt_offset(toks, prompt_toks)
    offsets = prompt_aware_offsets(len(toks), seq_len, windows, prompt_offset)

    samples: list[tuple[torch.Tensor, torch.Tensor]] = []
    stride = seq_len // 2 + 1
    usable = max(1, len(toks) - (seq_len + 1))
    for off in offsets:
        rows_x: list[torch.Tensor] = []
        rows_y: list[torch.Tensor] = []
        for i in range(batch_size):
            off_i = (off + i * stride) % usable
            window = token_window(toks, seq_len + 1, off_i)
            x_row, y_row = causal_lm_xy_one_hot(window, seq_len, vocab)
            rows_x.append(x_row)
            rows_y.append(y_row)
        samples.append((torch.stack(rows_x), torch.stack(rows_y)))
    return samples


def iter_gpt2_samples(
    text_path: Path = TINY_SHAKESPEARE,
    **kwargs,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    samples = build_gpt2_samples(text_path=text_path, **kwargs)
    idx = 0
    while True:
        yield samples[idx % len(samples)]
        idx += 1
