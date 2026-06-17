/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.API.Public.NN.FunctionalBatch

/-!
# Public Vision Layers

This file provides named-field, PyTorch-style layer records for common image operators. The API
keeps user-facing configuration explicit while lowering to TorchLean's typed channel-first tensor
operations.
-/

@[expose] public section

namespace NN
namespace API
namespace nn
namespace pure

/--
Named-field Conv2d configuration (CHW layout).

This is the public, PyTorch-like entry point for convolution in TorchLean.
PyTorch analogue: `torch.nn.Conv2d`.
See `https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html`.
-/
structure Conv2d where
  /-- Output channels. -/
  outC : Nat
  /-- Kernel height. -/
  kH : Nat
  /-- Kernel width. -/
  kW : Nat
  /-- Stride (shared for height/width). -/
  stride : Nat := 1
  /-- Zero-padding (shared for height/width). -/
  padding : Nat := 0
  /-- Seed for deterministic kernel initialization. -/
  seedK : Nat := 0
  /-- Seed for deterministic bias initialization. -/
  seedB : Nat := 0
  /-- Initialization scheme for the kernel weights. -/
  kInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .uniform (-0.1) 0.1

@[inherit_doc Conv2d]
abbrev Conv := Conv2d

/--
2D convolution over a CHW tensor, using explicit well-formedness proofs.
-/
def conv2dCHWWith {inC inH inW : Nat} (cfg : Conv2d)
    (hInC : inC ≠ 0) (hKH : cfg.kH ≠ 0) (hKW : cfg.kW ≠ 0) :
    Sequential
      (NN.Tensor.Shape.Image inC inH inW)
      (NN.Tensor.Shape.Image cfg.outC
        ((inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1)
        ((inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1)) :=
  TorchLean.Layers.conv2d inC cfg.outC cfg.kH cfg.kW cfg.stride cfg.padding inH inW
    (hInC := hInC) (hKH := hKH) (hKW := hKW)
    (seedK := cfg.seedK) (seedB := cfg.seedB) (kInit := cfg.kInit)

/--
2D convolution over a CHW tensor, with a PyTorch-like named-field spec.

This hides the Nat-side proof arguments via the `NeZero` typeclass.
-/
def conv2dCHW {inC inH inW : Nat} (cfg : Conv2d) [NeZero inC] [NeZero cfg.kH] [NeZero cfg.kW] :
    Sequential
      (NN.Tensor.Shape.Image inC inH inW)
      (NN.Tensor.Shape.Image cfg.outC
        ((inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1)
        ((inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1) ) :=
  conv2dCHWWith (inC := inC) (inH := inH) (inW := inW) cfg (NeZero.ne _) (NeZero.ne _) (NeZero.ne _)

/-- 2D convolution over a batched image tensor (shape `N×C×H×W`, like PyTorch). -/
def conv2d {n inC inH inW : Nat} (cfg : Conv2d) [NeZero inC] [NeZero cfg.kH] [NeZero cfg.kW] :
    Sequential
      (NN.Tensor.Shape.Images n inC inH inW)
      (NN.Tensor.Shape.Images n cfg.outC
        ((inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1)
        ((inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1)) :=
  batchDim0 n (conv2dCHW (inC := inC) (inH := inH) (inW := inW) cfg)

@[inherit_doc conv2dCHWWith]
def convCHWWith := @conv2dCHWWith

@[inherit_doc conv2dCHW]
def convCHW := @conv2dCHW

/--
Convolution over batched CHW images, using the PyTorch-style `Conv2d` config record.

Shorthand for `conv2d`.
-/
def conv {n inC inH inW : Nat} (cfg : Conv) [NeZero inC] [NeZero cfg.kH] [NeZero cfg.kW] :=
  conv2d (n := n) (inC := inC) (inH := inH) (inW := inW) cfg

/--
MaxPool2d configuration for CHW inputs.

PyTorch analogue: `torch.nn.MaxPool2d`.
See `https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html`.
-/
structure MaxPool2d where
  /-- Kernel height. -/
  kH : Nat
  /-- Kernel width. -/
  kW : Nat
  /-- Stride (shared for height/width). -/
  stride : Nat := 2

@[inherit_doc MaxPool2d]
abbrev MaxPool := MaxPool2d

/-- MaxPool2d with explicit nonzero kernel proofs. -/
def maxPool2dWith {inC inH inW : Nat} (cfg : MaxPool2d) (hKH : cfg.kH ≠ 0) (hKW : cfg.kW ≠ 0) :
    Sequential
      (NN.Tensor.Shape.Image inC inH inW)
      (NN.Tensor.Shape.Image inC
        ((inH - cfg.kH) / cfg.stride + 1)
        ((inW - cfg.kW) / cfg.stride + 1)) :=
  TorchLean.Layers.maxPool2d cfg.kH cfg.kW inH inW inC cfg.stride (hKH := hKH) (hKW := hKW)

/-- MaxPool2d over CHW inputs using `NeZero` to hide nonzero kernel proofs. -/
def maxPool2dCHW {inC inH inW : Nat} (cfg : MaxPool2d) [NeZero cfg.kH] [NeZero cfg.kW] :
    Sequential
      (NN.Tensor.Shape.Image inC inH inW)
      (NN.Tensor.Shape.Image inC
        ((inH - cfg.kH) / cfg.stride + 1)
        ((inW - cfg.kW) / cfg.stride + 1)) :=
  maxPool2dWith (inC := inC) (inH := inH) (inW := inW) cfg (NeZero.ne _) (NeZero.ne _)

/-- MaxPool2d using `NeZero` to hide nonzero kernel proofs. -/
def maxPool2d {n inC inH inW : Nat} (cfg : MaxPool2d) [NeZero cfg.kH] [NeZero cfg.kW] :
    Sequential
      (NN.Tensor.Shape.Images n inC inH inW)
      (NN.Tensor.Shape.Images n inC
        ((inH - cfg.kH) / cfg.stride + 1)
        ((inW - cfg.kW) / cfg.stride + 1)) :=
  batchDim0 n (maxPool2dCHW (inC := inC) (inH := inH) (inW := inW) cfg)

/-- Shorthand for `maxPool2dWith` (PyTorch-style). -/
def maxPoolWith := @maxPool2dWith

/-- Shorthand for `maxPool2dCHW` (PyTorch-style). -/
def maxPoolCHW := @maxPool2dCHW

/--
Max pooling over batched CHW images, using the PyTorch-style `MaxPool2d` config record.

Shorthand for `maxPool2d`.
-/
def maxPool {n inC inH inW : Nat} (cfg : MaxPool) [NeZero cfg.kH] [NeZero cfg.kW] :=
  maxPool2d (n := n) (inC := inC) (inH := inH) (inW := inW) cfg

/--
AvgPool2d configuration for CHW inputs.

PyTorch analogue: `torch.nn.AvgPool2d`.
See `https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html`.
-/
structure AvgPool2d where
  /-- Kernel height. -/
  kH : Nat
  /-- Kernel width. -/
  kW : Nat
  /-- Stride (shared for height/width). -/
  stride : Nat := 2

@[inherit_doc AvgPool2d]
abbrev AvgPool := AvgPool2d

/-- AvgPool2d with explicit nonzero kernel proofs. -/
def avgPool2dWith {inC inH inW : Nat} (cfg : AvgPool2d) (hKH : cfg.kH ≠ 0) (hKW : cfg.kW ≠ 0) :
    Sequential
      (NN.Tensor.Shape.Image inC inH inW)
      (NN.Tensor.Shape.Image inC
        ((inH - cfg.kH) / cfg.stride + 1)
        ((inW - cfg.kW) / cfg.stride + 1)) :=
  TorchLean.Layers.avgPool2d cfg.kH cfg.kW inH inW inC cfg.stride (hKH := hKH) (hKW := hKW)

/-- AvgPool2d over CHW inputs using `NeZero` to hide nonzero kernel proofs. -/
def avgPool2dCHW {inC inH inW : Nat} (cfg : AvgPool2d) [NeZero cfg.kH] [NeZero cfg.kW] :
    Sequential
      (NN.Tensor.Shape.Image inC inH inW)
      (NN.Tensor.Shape.Image inC
        ((inH - cfg.kH) / cfg.stride + 1)
        ((inW - cfg.kW) / cfg.stride + 1)) :=
  avgPool2dWith (inC := inC) (inH := inH) (inW := inW) cfg (NeZero.ne _) (NeZero.ne _)

/-- AvgPool2d over batched NCHW inputs (shape `N×C×H×W`, like PyTorch). -/
def avgPool2d {n inC inH inW : Nat} (cfg : AvgPool2d) [NeZero cfg.kH] [NeZero cfg.kW] :
    Sequential
      (NN.Tensor.Shape.Images n inC inH inW)
      (NN.Tensor.Shape.Images n inC
        ((inH - cfg.kH) / cfg.stride + 1)
        ((inW - cfg.kW) / cfg.stride + 1)) :=
  batchDim0 n (avgPool2dCHW (inC := inC) (inH := inH) (inW := inW) cfg)

/-- Shorthand for `avgPool2dWith` (PyTorch-style). -/
def avgPoolWith := @avgPool2dWith

/-- Shorthand for `avgPool2dCHW` (PyTorch-style). -/
def avgPoolCHW := @avgPool2dCHW

/--
Average pooling over batched CHW images, using the PyTorch-style `AvgPool2d` config record.

Shorthand for `avgPool2d`.
-/
def avgPool {n inC inH inW : Nat} (cfg : AvgPool) [NeZero cfg.kH] [NeZero cfg.kW] :=
  avgPool2d (n := n) (inC := inC) (inH := inH) (inW := inW) cfg

/--
Global average pooling over a CHW tensor.

PyTorch analogue: `torch.nn.AdaptiveAvgPool2d((1, 1))` followed by flattening.
-/
def globalAvgPoolCHW := TorchLean.Layers.globalAvgPoolCHW

/-- Global average pooling over an NCHW tensor (preserves the batch dimension). -/
def globalAvgPoolNCHW := TorchLean.Layers.globalAvgPoolNCHW

/--
LayerNorm configuration for batched `(batch x seqLen x embedDim)` tensors.

PyTorch analogue: `torch.nn.LayerNorm`.
See `https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html`.
-/
structure LayerNorm where
  /-- Seed for deterministic initialization of `gamma` (scale). -/
  seedGamma : Nat := 0
  /-- Seed for deterministic initialization of `beta` (shift). -/
  seedBeta : Nat := 0

/--
Layer normalization over `(batch × seqLen × embedDim)` tensors, with explicit positivity proofs.

This matches the common Transformer usage: normalize each token’s `embedDim`-vector independently,
with learnable scale/shift parameters `gamma` and `beta`.

PyTorch analogue: `torch.nn.LayerNorm(embedDim)` applied to a tensor of shape
`(batch, seqLen, embedDim)`.

Call `nn.layerNorm` when `NeZero` can discharge the positivity proofs automatically.
-/
def layerNormWith {batch seqLen embedDim : Nat} (cfg : LayerNorm)
    (hSeq : seqLen > 0) (hEmbed : embedDim > 0) :
    Sequential (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim))
      (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim)) :=
  TorchLean.Layers.layerNorm (batch := batch) (seqLen := seqLen) (embedDim := embedDim)
    (hSeq := hSeq) (hEmbed := hEmbed)
    (seedGamma := cfg.seedGamma) (seedBeta := cfg.seedBeta)

/--
Layer normalization over `(batch × seqLen × embedDim)` tensors.

This normalizes each `embedDim`-vector (per batch element, per sequence position), and applies
learned affine parameters `gamma` and `beta`.

PyTorch analogue: `torch.nn.LayerNorm(embedDim)` on a tensor shaped `(batch, seqLen, embedDim)`.

Implementation note:
TorchLean uses `NeZero` to ensure `seqLen` and `embedDim` are positive, avoiding degenerate shapes.
-/
def layerNorm {batch seqLen embedDim : Nat} (cfg : LayerNorm := {})
    [NeZero seqLen] [NeZero embedDim] :
    Sequential (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim))
      (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim)) :=
  layerNormWith (batch := batch) (seqLen := seqLen) (embedDim := embedDim) cfg
    (Nat.pos_of_ne_zero (NeZero.ne (n := seqLen)))
    (Nat.pos_of_ne_zero (NeZero.ne (n := embedDim)))

/--
RMSNorm configuration for batched `(batch x seqLen x embedDim)` tensors.

This is a common alternative to LayerNorm in modern transformer architectures.
-/
structure RMSNorm where
  /-- Seed for deterministic initialization of `gamma` (scale). -/
  seedGamma : Nat := 0

/--
RMS normalization over `(batch × seqLen × embedDim)` tensors, with explicit positivity proofs.

This is like LayerNorm but without mean subtraction: we scale by the root-mean-square over the
`embedDim` axis, and apply a learned scale `gamma`.

PyTorch analogue: many libraries provide an `RMSNorm(embedDim)` module; conceptually it is applied
to tensors shaped `(batch, seqLen, embedDim)`.

Call `nn.rmsNorm` when `NeZero` can discharge the positivity proofs automatically.
-/
def rmsNormWith {batch seqLen embedDim : Nat} (cfg : RMSNorm)
    (hSeq : seqLen > 0) (hEmbed : embedDim > 0) :
    Sequential (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim))
      (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim)) :=
  TorchLean.Layers.rmsNorm (batch := batch) (seqLen := seqLen) (embedDim := embedDim)
    (hSeq := hSeq) (hEmbed := hEmbed)
    (seedGamma := cfg.seedGamma)

/--
RMS normalization over `(batch × seqLen × embedDim)` tensors.

This normalizes by the root-mean-square over the `embedDim` axis (per batch element, per position),
then applies a learned scale `gamma`.

Implementation note:
TorchLean uses `NeZero` to ensure `seqLen` and `embedDim` are positive, avoiding degenerate shapes.
-/
def rmsNorm {batch seqLen embedDim : Nat} (cfg : RMSNorm := {})
    [NeZero seqLen] [NeZero embedDim] :
    Sequential (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim))
      (.dim batch (NN.Tensor.Shape.Mat seqLen embedDim)) :=
  rmsNormWith (batch := batch) (seqLen := seqLen) (embedDim := embedDim) cfg
    (Nat.pos_of_ne_zero (NeZero.ne (n := seqLen)))
    (Nat.pos_of_ne_zero (NeZero.ne (n := embedDim)))

/--
BatchNorm2d configuration (learned scale/shift).

PyTorch analogue: `torch.nn.BatchNorm2d`.
See `https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html`.
-/
structure BatchNorm2d where
  /-- Seed for deterministic initialization of `gamma` (scale). -/
  seedGamma : Nat := 0
  /-- Seed for deterministic initialization of `beta` (shift). -/
  seedBeta : Nat := 0

/-- BatchNorm2d over NCHW inputs (train/eval is handled by `Seq` mode). -/
def batchNorm2dNCHWWith {n c h w : Nat} (cfg : BatchNorm2d)
    (hN : n > 0) (hC : c > 0) (hH : h > 0) (hW : w > 0) :
    Sequential (NN.Tensor.Shape.Images n c h w) (NN.Tensor.Shape.Images n c h w) :=
  TorchLean.Layers.batchNorm2dNCHW (n := n) (c := c) (h := h) (w := w)
    (hN := hN) (hC := hC) (hH := hH) (hW := hW)
    (seedGamma := cfg.seedGamma) (seedBeta := cfg.seedBeta)

/--
BatchNorm2d over NCHW inputs, using `NeZero` to hide the positivity proofs.

PyTorch analogue: `torch.nn.BatchNorm2d`.
See `https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html`.
-/
def batchNorm2d {n c h w : Nat} (cfg : BatchNorm2d := {})
    [NeZero n] [NeZero c] [NeZero h] [NeZero w] :
    Sequential (NN.Tensor.Shape.Images n c h w) (NN.Tensor.Shape.Images n c h w) :=
  batchNorm2dNCHWWith (n := n) (c := c) (h := h) (w := w) cfg
    (Nat.pos_of_ne_zero (NeZero.ne (n := n)))
    (Nat.pos_of_ne_zero (NeZero.ne (n := c)))
    (Nat.pos_of_ne_zero (NeZero.ne (n := h)))
    (Nat.pos_of_ne_zero (NeZero.ne (n := w)))

/--
InstanceNorm2d configuration (learned scale/shift).

PyTorch analogue: `torch.nn.InstanceNorm2d`.
See `https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html`.
-/
structure InstanceNorm2d where
  /-- Seed for deterministic initialization of `gamma` (scale). -/
  seedGamma : Nat := 0
  /-- Seed for deterministic initialization of `beta` (shift). -/
  seedBeta : Nat := 0

/--
InstanceNorm2d over NCHW inputs, using explicit positivity proofs.

PyTorch analogue: `torch.nn.InstanceNorm2d`.
See `https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html`.
-/
def instanceNorm2dWith {n c h w : Nat} (cfg : InstanceNorm2d)
    (hN : n > 0) (hC : c > 0) (hH : h > 0) (hW : w > 0) :
    Sequential (NN.Tensor.Shape.Images n c h w) (NN.Tensor.Shape.Images n c h w) :=
  TorchLean.Layers.instanceNorm2dNCHW (n := n) (c := c) (h := h) (w := w)
    (hN := hN) (hC := hC) (hH := hH) (hW := hW)
    (seedGamma := cfg.seedGamma) (seedBeta := cfg.seedBeta)

/--
InstanceNorm2d over NCHW inputs, using `NeZero` to hide the positivity proofs.

PyTorch analogue: `torch.nn.InstanceNorm2d`.
See `https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html`.
-/
def instanceNorm2d {n c h w : Nat} (cfg : InstanceNorm2d := {})
    [NeZero n] [NeZero c] [NeZero h] [NeZero w] :
    Sequential (NN.Tensor.Shape.Images n c h w) (NN.Tensor.Shape.Images n c h w) :=
  instanceNorm2dWith (n := n) (c := c) (h := h) (w := w) cfg
    (Nat.pos_of_ne_zero (NeZero.ne (n := n)))
    (Nat.pos_of_ne_zero (NeZero.ne (n := c)))
    (Nat.pos_of_ne_zero (NeZero.ne (n := h)))
    (Nat.pos_of_ne_zero (NeZero.ne (n := w)))

/--
GroupNorm over NCHW inputs.

PyTorch analogue: `torch.nn.GroupNorm`.
See `https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html`.
-/
def groupNorm2dNCHW (n c h w groups : Nat) {hN : n > 0} {hC : c > 0} {hH : h > 0} {hW : w > 0}
    {hG : groups > 0} (hGE : c ≥ groups) (hDiv : c % groups = 0)
    (seedGamma seedBeta : Nat := 0) :
    Sequential (NN.Tensor.Shape.Images n c h w) (NN.Tensor.Shape.Images n c h w) :=
  TorchLean.Layers.groupNorm2dNCHW (n := n) (c := c) (h := h) (w := w) (groups := groups)
    (hN := hN) (hC := hC) (hH := hH) (hW := hW) (hG := hG)
    (hGE := hGE) (hDiv := hDiv) (seedGamma := seedGamma) (seedBeta := seedBeta)

/--
Multi-head self-attention configuration.

PyTorch analogue: `torch.nn.MultiheadAttention` (conceptually).
See `https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html`.
-/
structure MultiheadAttention where
  /-- Number of attention heads. -/
  numHeads : Nat
  /-- Per-head embedding dimension. -/
  headDim : Nat
  /-- Base seed for deterministic parameter initialization. -/
  seedW : Nat := 0

/--
Multi-head self-attention with an explicit nonzero sequence length proof.

If `mask` is provided, it is a boolean attention mask of shape `(n × n)` (e.g. causal masking).
-/
def multiheadAttentionWith {batch n dModel : Nat} (cfg : MultiheadAttention) (hN : n ≠ 0)
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    Sequential (.dim batch (NN.Tensor.Shape.Mat n dModel)) (.dim batch (NN.Tensor.Shape.Mat n dModel))
      :=
  TorchLean.Layers.attention (batch := batch) (n := n) (dModel := dModel)
    (numHeads := cfg.numHeads) (headDim := cfg.headDim)
    (hN := hN) (seedW := cfg.seedW) (mask := mask)

/--
Multi-head self-attention using `NeZero` to hide the nonzero sequence length proof.

If `mask` is provided, it is a boolean attention mask of shape `(n × n)` (e.g. causal masking).
-/
def multiheadAttention {batch n dModel : Nat} (cfg : MultiheadAttention) [NeZero n]
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    Sequential (.dim batch (NN.Tensor.Shape.Mat n dModel)) (.dim batch (NN.Tensor.Shape.Mat n dModel))
      :=
  multiheadAttentionWith (batch := batch) (n := n) (dModel := dModel) cfg (NeZero.ne (n := n))
    (mask := mask)
