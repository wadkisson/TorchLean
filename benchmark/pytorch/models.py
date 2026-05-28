"""
PyTorch models mirroring the TorchLean API example architectures.

- MLP: `NN.API.Models.Mlp.mlp1Relu` / `torchlean mlp`
- CNN: `NN.API.Models.Cnn.cnn` / `torchlean cnn`
- GPT-2 style LM: `NN.API.Models.Gpt2.causalTransformerOneHot` / `torchlean gpt2`
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark.config import (
    CNN_CONV_K,
    CNN_CONV_OUT_C,
    CNN_CONV_PADDING,
    CNN_CONV_STRIDE,
    CNN_IN_C,
    CNN_IN_H,
    CNN_IN_W,
    CNN_OUT_DIM,
    CNN_POOL_K,
    CNN_POOL_STRIDE,
    GPT2_D_MODEL,
    GPT2_FFN_HIDDEN,
    GPT2_LAYERS,
    GPT2_NUM_HEADS,
    GPT2_SEQ_LEN,
    GPT2_VOCAB,
    MLP_HID_DIM,
    MLP_IN_DIM,
    MLP_OUT_DIM,
)


def cnn_feat_size(
    in_h: int = CNN_IN_H,
    in_w: int = CNN_IN_W,
    out_c: int = CNN_CONV_OUT_C,
    k: int = CNN_CONV_K,
    stride: int = CNN_CONV_STRIDE,
    padding: int = CNN_CONV_PADDING,
    pool_k: int = CNN_POOL_K,
    pool_stride: int = CNN_POOL_STRIDE,
) -> int:
    out_h1 = (in_h + 2 * padding - k) // stride + 1
    out_w1 = (in_w + 2 * padding - k) // stride + 1
    out_h2 = (out_h1 - pool_k) // pool_stride + 1
    out_w2 = (out_w1 - pool_k) // pool_stride + 1
    return out_c * out_h2 * out_w2


class TorchLeanMlp(nn.Module):
    """Linear(in) -> ReLU -> Linear(out), matching `nn.models.mlp1Relu`."""

    def __init__(
        self,
        in_dim: int = MLP_IN_DIM,
        hid_dim: int = MLP_HID_DIM,
        out_dim: int = MLP_OUT_DIM,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class TorchLeanCnn(nn.Module):
    """Conv -> ReLU -> MaxPool -> Flatten -> Linear, matching `nn.models.cnn`."""

    def __init__(
        self,
        in_c: int = CNN_IN_C,
        in_h: int = CNN_IN_H,
        in_w: int = CNN_IN_W,
        out_dim: int = CNN_OUT_DIM,
        conv_out_c: int = CNN_CONV_OUT_C,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_c,
            conv_out_c,
            kernel_size=CNN_CONV_K,
            stride=CNN_CONV_STRIDE,
            padding=CNN_CONV_PADDING,
        )
        self.pool = nn.MaxPool2d(kernel_size=CNN_POOL_K, stride=CNN_POOL_STRIDE)
        feat = cnn_feat_size(in_h, in_w, conv_out_c)
        self.fc = nn.Linear(feat, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv(x)))
        return self.fc(x.flatten(1))


class MultiHeadAttention(nn.Module):
    """Bias-free Q/K/V/O projections, matching TorchLean MHA blocks."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        return self.out_proj(out)


class TransformerEncoderBlock(nn.Module):
    """
    Post-norm block matching `blocks.transformerEncoderBlockWithMask`:
      residual(attn) -> norm1 -> residual(ffn) -> norm2
    with GELU in the FFN.
    """

    def __init__(self, d_model: int, num_heads: int, ffn_hidden: int) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.norm1(x + self.attn(x, mask))
        x = self.norm2(x + self.ffn(x))
        return x


class TorchLeanGpt2(nn.Module):
    """
    Embedding -> learned positional embedding -> Transformer stack -> LayerNorm -> LM head.

    Matches `nn.models.causalTransformerOneHot` with one-hot token inputs.
    """

    def __init__(
        self,
        vocab: int = GPT2_VOCAB,
        seq_len: int = GPT2_SEQ_LEN,
        d_model: int = GPT2_D_MODEL,
        num_heads: int = GPT2_NUM_HEADS,
        ffn_hidden: int = GPT2_FFN_HIDDEN,
        layers: int = GPT2_LAYERS,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.token_embedding = nn.Embedding(vocab, d_model)
        self.pos_embedding = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, num_heads, ffn_hidden)
                for _ in range(layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=True)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)).unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

    def embed_one_hot(self, x_one_hot: torch.Tensor) -> torch.Tensor:
        # One-hot rows select embedding rows, equivalent to an embedding lookup.
        return torch.matmul(x_one_hot, self.token_embedding.weight)

    def forward(self, x_one_hot: torch.Tensor) -> torch.Tensor:
        b, s, v = x_one_hot.shape
        positions = torch.arange(s, device=x_one_hot.device).unsqueeze(0).expand(b, -1)
        x = self.embed_one_hot(x_one_hot) + self.pos_embedding(positions)
        mask = self.causal_mask[:, :, :s, :s]
        for block in self.blocks:
            x = block(x, mask)
        x = self.final_norm(x)
        return self.lm_head(x)


def cross_entropy_one_hot_mean(logits: torch.Tensor, target_one_hot: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy on one-hot targets, matching `crossEntropyOneHot` with `.mean`."""
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_one_hot * log_probs).sum() / target_one_hot.numel()
