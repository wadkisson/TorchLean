#!/usr/bin/env python3
"""From-scratch GPT-2 (~506M) in PyTorch: train on GPU, then sample.

Architecture matches the gpt2-500m preset used elsewhere in this repo:
  d_model=1024, n_layers=32, n_heads=16, head_dim=64, ffn=4096, vocab=50257

No HuggingFace / transformers dependency — just torch.

Examples:
  python3 benchmark/gpt2_pytorch.py --steps 100
  python3 benchmark/gpt2_pytorch.py --steps 20 --sample-tokens 64
  python3 benchmark/gpt2_pytorch.py --steps 0 --sample-only --checkpoint out.pt
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config (~506M parameters)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 50257
    seq_len: int = 128
    d_model: int = 1024
    n_layers: int = 32
    n_heads: int = 16
    ffn_hidden: int = 4096
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------------------
# Model (GPT-2 style, built from scratch)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=True)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        # Causal mask registered as buffer so it moves with .to(device)
        mask = torch.tril(torch.ones(cfg.seq_len, cfg.seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.seq_len, cfg.seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, T, H, D)
        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, H, T, D)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=True)
        self.proj = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=True)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GPT-2 uses GELU (tanh approximation in original; exact gelu is fine)
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2(nn.Module):
    """Decoder-only transformer matching GPT-2 layout (~506M at default cfg)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        # Untied LM head → ~506M params (tied would be ~455M). Matches gpt2-500m preset.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        if T > self.cfg.seq_len:
            raise ValueError(f"sequence length {T} > context {self.cfg.seq_len}")
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))  # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = 50,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.seq_len :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ---------------------------------------------------------------------------
# Train / sample
# ---------------------------------------------------------------------------

def make_batch(cfg: GPTConfig, batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic token stream (benchmark only — no real text corpus)."""
    data = torch.randint(0, cfg.vocab_size, (batch, cfg.seq_len + 1), device=device)
    return data[:, :-1], data[:, 1:]


def train(model: GPT2, cfg: GPTConfig, args: argparse.Namespace, device: torch.device) -> dict:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Warmup timing: one forward, one fwd+bwd
    x, y = make_batch(cfg, args.batch, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits, loss = model(x, y)
    if device.type == "cuda":
        torch.cuda.synchronize()
    forward_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    logits, loss = model(x, y)
    loss.backward()
    opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    fwd_bwd_ms = (time.perf_counter() - t0) * 1e3

    print(f"forward     = {forward_ms:.1f} ms")
    print(f"fwd+bwd     = {fwd_bwd_ms:.1f} ms  (loss {loss.item():.4f})")

    losses: list[float] = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        x, y = make_batch(cfg, args.batch, device)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        losses.append(loss.item())
        if step == 1 or step % max(1, args.log_every) == 0 or step == args.steps:
            print(f"  step {step}/{args.steps}: loss={loss.item():.4f}")
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_s = time.perf_counter() - wall0

    avg_step_ms = (wall_s / max(1, args.steps)) * 1e3 if args.steps else 0.0
    tokens_per_s = (args.batch * cfg.seq_len * args.steps) / wall_s if wall_s > 0 else 0.0
    print(f"train_wall  = {wall_s:.3f} s  for {args.steps} steps")
    print(f"avg_step    = {avg_step_ms:.1f} ms  ({tokens_per_s:,.0f} tokens/s)")

    return {
        "framework": "pytorch",
        "preset": "gpt2-500m",
        "device": str(device),
        "params": count_parameters(model),
        "batch": args.batch,
        "seq_len": cfg.seq_len,
        "forward_ms": forward_ms,
        "fwd_bwd_ms": fwd_bwd_ms,
        "train_wall_s": wall_s,
        "avg_step_ms": avg_step_ms,
        "tokens_per_s": tokens_per_s,
        "final_loss": losses[-1] if losses else None,
        "config": asdict(cfg),
    }


@torch.no_grad()
def sample(model: GPT2, cfg: GPTConfig, args: argparse.Namespace, device: torch.device) -> None:
    model.eval()
    # Random prompt of a few tokens (no tokenizer dependency)
    prompt_len = min(args.prompt_tokens, cfg.seq_len)
    idx = torch.randint(0, cfg.vocab_size, (1, prompt_len), device=device)
    print(f"\n[sample] prompt tokens ({prompt_len}): {idx[0].tolist()}")
    out = model.generate(
        idx,
        max_new_tokens=args.sample_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(f"[sample] generated {args.sample_tokens} new tokens:")
    print(out[0].tolist())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="From-scratch GPT-2 (~506M) train + sample")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample-tokens", type=int, default=32, help="tokens to generate after train")
    p.add_argument("--prompt-tokens", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--no-sample", action="store_true")
    p.add_argument("--sample-only", action="store_true")
    p.add_argument("--checkpoint", type=str, default="", help="save/load path for weights")
    p.add_argument("--json", action="store_true", help="print metrics JSON at the end")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)

    cfg = GPTConfig()
    model = GPT2(cfg).to(device)
    n_params = count_parameters(model)

    print("== PyTorch GPT-2 (from scratch) ==")
    print(f"device      = {device}")
    if device.type == "cuda":
        print(f"gpu         = {torch.cuda.get_device_name(device)}")
    print(
        f"d_model={cfg.d_model} n_layers={cfg.n_layers} n_heads={cfg.n_heads} "
        f"head_dim={cfg.head_dim} ffn_hidden={cfg.ffn_hidden} vocab={cfg.vocab_size} "
        f"batch={args.batch} seq_len={cfg.seq_len}"
    )
    print(f"parameters  = {n_params:,} ({n_params / 1e6:.2f}M)")

    if args.checkpoint and args.sample_only:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"loaded checkpoint {args.checkpoint}")

    metrics = None
    if not args.sample_only:
        metrics = train(model, cfg, args, device)
        if args.checkpoint:
            torch.save(model.state_dict(), args.checkpoint)
            print(f"wrote checkpoint {args.checkpoint}")

    if not args.no_sample:
        sample(model, cfg, args, device)

    if args.json and metrics is not None:
        print("JSON", json.dumps(metrics))


if __name__ == "__main__":
    main()
