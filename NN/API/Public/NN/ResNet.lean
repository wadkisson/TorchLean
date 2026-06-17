/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.API.Public.NN.Blocks

/-!
# Public ResNet Builders

This file exposes the ResNet block and model builders in the high-level `nn.pure.blocks` API. The
definitions keep the usual ResNet shape arithmetic visible in Lean, including stride-2 downsampling
and the residual shortcut path.
-/

@[expose] public section

namespace NN
namespace API
namespace nn
namespace pure
namespace blocks

/--
Canonical stride-2 spatial downsampling formula used by ResNet blocks.

`down2 h = (h - 1) / 2 + 1 = ceil(h / 2)`.

This matches the output-size formula for common stride-2 layers used in ResNet downsampling
(e.g. `3×3` conv with padding `1`, or `1×1` conv with padding `0`).
-/
abbrev down2 (h : Nat) : Nat := (h - 1) / 2 + 1

/-- `down2` is always positive (used to discharge `NeZero` goals). -/
lemma down2_pos (h : Nat) : down2 h > 0 := by
  simp [down2]

/--
Shape arithmetic fact: `3×3` conv with stride `1` and padding `1` preserves a positive spatial
  size.

This matches the standard conv output formula used by `conv2dCHW`.
-/
lemma conv3_same_out_eq {h : Nat} (hh : h > 0) : ((h + 2 * 1 - 3) / 1 + 1) = h := by
  cases h with
  | zero => cases (Nat.lt_irrefl 0 hh)
  | succ _n => simp

/--
Shape arithmetic fact: `1×1` conv with stride `1` and padding `0` preserves a positive spatial
  size.
-/
lemma conv1_same_out_eq {h : Nat} (hh : h > 0) : ((h + 2 * 0 - 1) / 1 + 1) = h := by
  cases h with
  | zero => cases (Nat.lt_irrefl 0 hh)
  | succ _n => simp

/--
ResNet `3×3` convolution with padding `1`, stride `1` (shape-preserving), over CHW images.
-/
def conv3x3Same {inC outC h w : Nat}
    [NeZero inC] [NeZero h] [NeZero w]
    (seedK seedB : Nat := 0)
    (kInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .uniform (-0.1) 0.1) :
    Sequential (NN.Tensor.Shape.Image inC h w) (NN.Tensor.Shape.Image outC h w) := by
  let raw :=
    conv2dCHW (inC := inC) (inH := h) (inW := w)
      { outC := outC, kH := 3, kW := 3, stride := 1, padding := 1
      , seedK := seedK, seedB := seedB, kInit := kInit }
  have hh : h > 0 := Nat.pos_of_ne_zero (NeZero.ne (n := h))
  have hw : w > 0 := Nat.pos_of_ne_zero (NeZero.ne (n := w))
  have hH : ((h + 2 * 1 - 3) / 1 + 1) = h := conv3_same_out_eq hh
  have hW : ((w + 2 * 1 - 3) / 1 + 1) = w := conv3_same_out_eq hw
  have hShape :
      NN.Tensor.Shape.Image outC ((h + 2 * 1 - 3) / 1 + 1) ((w + 2 * 1 - 3) / 1 + 1) =
        NN.Tensor.Shape.Image outC h w := by
    simpa [NN.Tensor.Shape.Image] using And.intro hH hW
  exact
    Eq.ndrec (motive := fun τ => Sequential (NN.Tensor.Shape.Image inC h w) τ) raw hShape

/--
ResNet `3×3` convolution with padding `1`, stride `2` (spatial downsampling via `down2`),
  over CHW images.
-/
def conv3x3Down {inC outC h w : Nat} [NeZero inC]
    (seedK seedB : Nat := 0)
    (kInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .uniform (-0.1) 0.1) :
    Sequential (NN.Tensor.Shape.Image inC h w) (NN.Tensor.Shape.Image outC (down2 h) (down2 w)) :=
      by
  let raw :=
    conv2dCHW (inC := inC) (inH := h) (inW := w)
      { outC := outC, kH := 3, kW := 3, stride := 2, padding := 1
      , seedK := seedK, seedB := seedB, kInit := kInit }
  have hShape :
      NN.Tensor.Shape.CHW outC ((h + 2 * 1 - 3) / 2 + 1) ((w + 2 * 1 - 3) / 2 + 1) =
        NN.Tensor.Shape.CHW outC (down2 h) (down2 w) := by
    simp [NN.Tensor.Shape.CHW, down2, Nat.add_comm]
  exact
    Eq.ndrec (motive := fun τ => Sequential (NN.Tensor.Shape.Image inC h w) τ) raw hShape

/-- ResNet `1×1` convolution with stride `1` (shape-preserving), over CHW images. -/
def conv1x1Same {inC outC h w : Nat}
    [NeZero inC] [NeZero h] [NeZero w]
    (seedK seedB : Nat := 0)
    (kInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .uniform (-0.1) 0.1) :
    Sequential (NN.Tensor.Shape.Image inC h w) (NN.Tensor.Shape.Image outC h w) := by
  let raw :=
    conv2dCHW (inC := inC) (inH := h) (inW := w)
      { outC := outC, kH := 1, kW := 1, stride := 1, padding := 0
      , seedK := seedK, seedB := seedB, kInit := kInit }
  have hh : h > 0 := Nat.pos_of_ne_zero (NeZero.ne (n := h))
  have hw : w > 0 := Nat.pos_of_ne_zero (NeZero.ne (n := w))
  have hH : ((h + 2 * 0 - 1) / 1 + 1) = h := conv1_same_out_eq hh
  have hW : ((w + 2 * 0 - 1) / 1 + 1) = w := conv1_same_out_eq hw
  have hShape :
      NN.Tensor.Shape.Image outC ((h + 2 * 0 - 1) / 1 + 1) ((w + 2 * 0 - 1) / 1 + 1) =
        NN.Tensor.Shape.Image outC h w := by
    simpa [NN.Tensor.Shape.Image] using And.intro hH hW
  exact
    Eq.ndrec (motive := fun τ => Sequential (NN.Tensor.Shape.Image inC h w) τ) raw hShape

/-- ResNet `1×1` convolution with stride `2` (spatial downsampling via `down2`), over CHW
  images. -/
def conv1x1Down {inC outC h w : Nat} [NeZero inC]
    (seedK seedB : Nat := 0)
    (kInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .uniform (-0.1) 0.1) :
    Sequential (NN.Tensor.Shape.Image inC h w) (NN.Tensor.Shape.Image outC (down2 h) (down2 w)) :=
      by
  let raw :=
    conv2dCHW (inC := inC) (inH := h) (inW := w)
      { outC := outC, kH := 1, kW := 1, stride := 2, padding := 0
      , seedK := seedK, seedB := seedB, kInit := kInit }
  have hShape :
      NN.Tensor.Shape.Image outC ((h + 2 * 0 - 1) / 2 + 1) ((w + 2 * 0 - 1) / 2 + 1) =
        NN.Tensor.Shape.Image outC (down2 h) (down2 w) := by
    simp [NN.Tensor.Shape.Image, down2, Nat.add_comm]
  exact
    Eq.ndrec (motive := fun τ => Sequential (NN.Tensor.Shape.Image inC h w) τ) raw hShape

/-- ResNet `3×3` convolution over batched images (`NCHW`-style), preserving spatial size. -/
def conv3x3SameImages {n inC outC h w : Nat}
    [NeZero n] [NeZero inC] [NeZero h] [NeZero w]
    (seedK seedB : Nat := 0)
    (kInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .uniform (-0.1) 0.1) :
    Sequential (NN.Tensor.Shape.Images n inC h w) (NN.Tensor.Shape.Images n outC h w) := by
  let raw :=
    conv2d (n := n) (inC := inC) (inH := h) (inW := w)
      { outC := outC, kH := 3, kW := 3, stride := 1, padding := 1
      , seedK := seedK, seedB := seedB, kInit := kInit }
  have hh : h > 0 := Nat.pos_of_ne_zero (NeZero.ne (n := h))
  have hw : w > 0 := Nat.pos_of_ne_zero (NeZero.ne (n := w))
  have hH : ((h + 2 * 1 - 3) / 1 + 1) = h := conv3_same_out_eq hh
  have hW : ((w + 2 * 1 - 3) / 1 + 1) = w := conv3_same_out_eq hw
  have hShape :
      NN.Tensor.Shape.Images n outC ((h + 2 * 1 - 3) / 1 + 1) ((w + 2 * 1 - 3) / 1 + 1) =
        NN.Tensor.Shape.Images n outC h w := by
    simpa [NN.Tensor.Shape.Images] using And.intro hH hW
  exact
    Eq.ndrec (motive := fun τ => Sequential (NN.Tensor.Shape.Images n inC h w) τ) raw hShape

/-- ResNet `3×3` convolution over batched images (`NCHW`-style), downsampling via `down2`.
  -/
def conv3x3DownImages {n inC outC h w : Nat}
    [NeZero n] [NeZero inC]
    (seedK seedB : Nat := 0)
    (kInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .uniform (-0.1) 0.1) :
    Sequential (NN.Tensor.Shape.Images n inC h w) (NN.Tensor.Shape.Images n outC (down2 h) (down2
      w)) := by
  let raw :=
    conv2d (n := n) (inC := inC) (inH := h) (inW := w)
      { outC := outC, kH := 3, kW := 3, stride := 2, padding := 1
      , seedK := seedK, seedB := seedB, kInit := kInit }
  have hShape :
      NN.Tensor.Shape.Images n outC ((h + 2 * 1 - 3) / 2 + 1) ((w + 2 * 1 - 3) / 2 + 1) =
        NN.Tensor.Shape.Images n outC (down2 h) (down2 w) := by
    simp [NN.Tensor.Shape.Images, down2, Nat.add_comm]
  exact
    Eq.ndrec (motive := fun τ => Sequential (NN.Tensor.Shape.Images n inC h w) τ) raw hShape

/-- ResNet `1×1` convolution over batched images (`NCHW`-style), preserving spatial size. -/
def conv1x1SameImages {n inC outC h w : Nat}
    [NeZero n] [NeZero inC] [NeZero h] [NeZero w]
    (seedK seedB : Nat := 0)
    (kInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .uniform (-0.1) 0.1) :
    Sequential (NN.Tensor.Shape.Images n inC h w) (NN.Tensor.Shape.Images n outC h w) := by
  let raw :=
    conv2d (n := n) (inC := inC) (inH := h) (inW := w)
      { outC := outC, kH := 1, kW := 1, stride := 1, padding := 0
      , seedK := seedK, seedB := seedB, kInit := kInit }
  have hh : h > 0 := Nat.pos_of_ne_zero (NeZero.ne (n := h))
  have hw : w > 0 := Nat.pos_of_ne_zero (NeZero.ne (n := w))
  have hH : ((h + 2 * 0 - 1) / 1 + 1) = h := conv1_same_out_eq hh
  have hW : ((w + 2 * 0 - 1) / 1 + 1) = w := conv1_same_out_eq hw
  have hShape :
      NN.Tensor.Shape.Images n outC ((h + 2 * 0 - 1) / 1 + 1) ((w + 2 * 0 - 1) / 1 + 1) =
        NN.Tensor.Shape.Images n outC h w := by
    simpa [NN.Tensor.Shape.Images] using And.intro hH hW
  exact
    Eq.ndrec (motive := fun τ => Sequential (NN.Tensor.Shape.Images n inC h w) τ) raw hShape

/-- ResNet `1×1` convolution over batched images (`NCHW`-style), downsampling via `down2`.
  -/
def conv1x1DownImages {n inC outC h w : Nat}
    [NeZero n] [NeZero inC]
    (seedK seedB : Nat := 0)
    (kInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .uniform (-0.1) 0.1) :
    Sequential (NN.Tensor.Shape.Images n inC h w) (NN.Tensor.Shape.Images n outC (down2 h) (down2
      w)) := by
  let raw :=
    conv2d (n := n) (inC := inC) (inH := h) (inW := w)
      { outC := outC, kH := 1, kW := 1, stride := 2, padding := 0
      , seedK := seedK, seedB := seedB, kInit := kInit }
  have hShape :
      NN.Tensor.Shape.Images n outC ((h + 2 * 0 - 1) / 2 + 1) ((w + 2 * 0 - 1) / 2 + 1) =
        NN.Tensor.Shape.Images n outC (down2 h) (down2 w) := by
    simp [NN.Tensor.Shape.Images, down2, Nat.add_comm]
  exact
    Eq.ndrec (motive := fun τ => Sequential (NN.Tensor.Shape.Images n inC h w) τ) raw hShape

/--
ResNet-style "basic block" configuration (CHW layout).

PyTorch reference (conceptual):
`torchvision.models.resnet.BasicBlock` (see `https://pytorch.org/vision/stable/models/resnet.html`).
-/
structure ResNetBasicBlock where
  /-- Number of output channels produced by the block. -/
  outC : Nat
  /-- If true, use stride-2 downsampling + projection shortcut; otherwise preserve spatial dims. -/
  downsample : Bool := false
  /-- Activation used inside the block (and after the residual addition). -/
  activation : Activation := .relu
  /-- Base seed used to derive deterministic per-layer seeds inside the block. -/
  seedBase : Nat := 0

/--
ResNet-style "basic block" configuration (CHW layout).

This public building block follows the standard ResNet basic-block pattern:
`conv3x3 -> BN -> act -> conv3x3 -> BN` with a residual/skip connection.

PyTorch references (for the conceptual shape):
- Torchvision ResNet: `https://pytorch.org/vision/stable/models/resnet.html`
-/
def resnetBasicBlockCHW {inC h w : Nat} (cfg : ResNetBasicBlock)
    [NeZero inC] [NeZero h] [NeZero w] [NeZero cfg.outC] :
    Sequential
      (NN.Tensor.Shape.Image inC h w)
      (NN.Tensor.Shape.Image cfg.outC
        (if cfg.downsample then down2 h else h)
        (if cfg.downsample then down2 w else w)) := by
  classical
  -- Seed layout: conv/bn/conv/bn/(proj conv/bn) in a fixed order.
  let seedConv1K := cfg.seedBase + 0
  let seedConv1B := cfg.seedBase + 1
  let seedBN1G := cfg.seedBase + 2
  let seedBN1B := cfg.seedBase + 3
  let seedBN1M := cfg.seedBase + 4
  let seedBN1V := cfg.seedBase + 5
  let seedConv2K := cfg.seedBase + 6
  let seedConv2B := cfg.seedBase + 7
  let seedBN2G := cfg.seedBase + 8
  let seedBN2B := cfg.seedBase + 9
  let seedBN2M := cfg.seedBase + 10
  let seedBN2V := cfg.seedBase + 11
  let seedProjK := cfg.seedBase + 12
  let seedProjB := cfg.seedBase + 13
  let seedProjBNG := cfg.seedBase + 14
  let seedProjBNB := cfg.seedBase + 15
  let seedProjBNM := cfg.seedBase + 16
  let seedProjBNV := cfg.seedBase + 17

  if hDown : cfg.downsample = true then
    -- Stride-2 downsample.
    let h' := down2 h
    let w' := down2 w
    have hh' : h' > 0 := down2_pos h
    have hw' : w' > 0 := down2_pos w

    let conv1 : Sequential (NN.Tensor.Shape.Image inC h w) (NN.Tensor.Shape.Image cfg.outC h' w') :=
      conv3x3Down (inC := inC) (outC := cfg.outC) (h := h) (w := w)
        (seedK := seedConv1K) (seedB := seedConv1B)
    let bn1 : Sequential (NN.Tensor.Shape.Image cfg.outC h' w') (NN.Tensor.Shape.Image cfg.outC h'
      w') :=
      TorchLean.Layers.batchNormCHW cfg.outC h' w'
        (hC := Nat.pos_of_ne_zero (NeZero.ne (n := cfg.outC)))
        (hH := hh') (hW := hw')
        (seedGamma := seedBN1G) (seedBeta := seedBN1B) (seedMean := seedBN1M) (seedVar := seedBN1V)
    let relu1 : Sequential (NN.Tensor.Shape.Image cfg.outC h' w') (NN.Tensor.Shape.Image cfg.outC h'
      w') :=
      activation (s := NN.Tensor.Shape.Image cfg.outC h' w') cfg.activation
    let conv2 : Sequential (NN.Tensor.Shape.Image cfg.outC h' w') (NN.Tensor.Shape.Image cfg.outC h'
      w') :=
      conv3x3Same (inC := cfg.outC) (outC := cfg.outC) (h := h') (w := w')
        (seedK := seedConv2K) (seedB := seedConv2B)
    let bn2 : Sequential (NN.Tensor.Shape.Image cfg.outC h' w') (NN.Tensor.Shape.Image cfg.outC h'
      w') :=
      TorchLean.Layers.batchNormCHW cfg.outC h' w'
        (hC := Nat.pos_of_ne_zero (NeZero.ne (n := cfg.outC)))
        (hH := hh') (hW := hw')
        (seedGamma := seedBN2G) (seedBeta := seedBN2B) (seedMean := seedBN2M) (seedVar := seedBN2V)
    let main := seq! conv1, bn1, relu1, conv2, bn2

    let projConv : Sequential (NN.Tensor.Shape.Image inC h w) (NN.Tensor.Shape.Image cfg.outC h' w')
      :=
      conv1x1Down (inC := inC) (outC := cfg.outC) (h := h) (w := w)
        (seedK := seedProjK) (seedB := seedProjB)
    let projBN : Sequential (NN.Tensor.Shape.Image cfg.outC h' w') (NN.Tensor.Shape.Image cfg.outC
      h' w') :=
      TorchLean.Layers.batchNormCHW cfg.outC h' w'
        (hC := Nat.pos_of_ne_zero (NeZero.ne (n := cfg.outC)))
        (hH := hh') (hW := hw')
        (seedGamma := seedProjBNG) (seedBeta := seedProjBNB) (seedMean := seedProjBNM) (seedVar :=
          seedProjBNV)
    let skip := seq! projConv, projBN

    let summed : Sequential (NN.Tensor.Shape.Image inC h w) (NN.Tensor.Shape.Image cfg.outC h' w')
      :=
      addBranches main skip
    let outAct : Sequential (NN.Tensor.Shape.Image cfg.outC h' w') (NN.Tensor.Shape.Image cfg.outC
      h' w') :=
      activation (s := NN.Tensor.Shape.Image cfg.outC h' w') cfg.activation

    -- The `if` in the return type reduces via `hDown`.
    simpa [hDown] using (seq! summed, outAct)
  else
    -- Stride-1 (no spatial downsample).
    have hh : h > 0 := Nat.pos_of_ne_zero (NeZero.ne (n := h))
    have hw : w > 0 := Nat.pos_of_ne_zero (NeZero.ne (n := w))

    let conv1 : Sequential (NN.Tensor.Shape.Image inC h w) (NN.Tensor.Shape.Image cfg.outC h w) :=
      conv3x3Same (inC := inC) (outC := cfg.outC) (h := h) (w := w)
        (seedK := seedConv1K) (seedB := seedConv1B)
    let bn1 : Sequential (NN.Tensor.Shape.Image cfg.outC h w) (NN.Tensor.Shape.Image cfg.outC h w)
      :=
      TorchLean.Layers.batchNormCHW cfg.outC h w
        (hC := Nat.pos_of_ne_zero (NeZero.ne (n := cfg.outC)))
        (hH := hh) (hW := hw)
        (seedGamma := seedBN1G) (seedBeta := seedBN1B) (seedMean := seedBN1M) (seedVar := seedBN1V)
    let relu1 : Sequential (NN.Tensor.Shape.Image cfg.outC h w) (NN.Tensor.Shape.Image cfg.outC h w)
      :=
      activation (s := NN.Tensor.Shape.Image cfg.outC h w) cfg.activation
    let conv2 : Sequential (NN.Tensor.Shape.Image cfg.outC h w) (NN.Tensor.Shape.Image cfg.outC h w)
      :=
      conv3x3Same (inC := cfg.outC) (outC := cfg.outC) (h := h) (w := w)
        (seedK := seedConv2K) (seedB := seedConv2B)
    let bn2 : Sequential (NN.Tensor.Shape.Image cfg.outC h w) (NN.Tensor.Shape.Image cfg.outC h w)
      :=
      TorchLean.Layers.batchNormCHW cfg.outC h w
        (hC := Nat.pos_of_ne_zero (NeZero.ne (n := cfg.outC)))
        (hH := hh) (hW := hw)
        (seedGamma := seedBN2G) (seedBeta := seedBN2B) (seedMean := seedBN2M) (seedVar := seedBN2V)
    let main := seq! conv1, bn1, relu1, conv2, bn2

    let skip : Sequential (NN.Tensor.Shape.Image inC h w) (NN.Tensor.Shape.Image cfg.outC h w) :=
      if hEq : cfg.outC = inC then
        -- Identity shortcut.
        have hShape : NN.Tensor.Shape.Image inC h w = NN.Tensor.Shape.Image cfg.outC h w := by
          simp [hEq]
        Eq.ndrec (motive := fun τ => Sequential (NN.Tensor.Shape.Image inC h w) τ)
          (_root_.Runtime.Autograd.TorchLean.NN.Seq.id (NN.Tensor.Shape.Image inC h w)) hShape
      else
        -- Projection shortcut (1x1 + BN).
        let projConv : Sequential (NN.Tensor.Shape.Image inC h w) (NN.Tensor.Shape.Image cfg.outC h
          w) :=
          conv1x1Same (inC := inC) (outC := cfg.outC) (h := h) (w := w)
            (seedK := seedProjK) (seedB := seedProjB)
        let projBN : Sequential (NN.Tensor.Shape.Image cfg.outC h w) (NN.Tensor.Shape.Image cfg.outC
          h w) :=
          TorchLean.Layers.batchNormCHW cfg.outC h w
            (hC := Nat.pos_of_ne_zero (NeZero.ne (n := cfg.outC)))
            (hH := hh) (hW := hw)
            (seedGamma := seedProjBNG) (seedBeta := seedProjBNB) (seedMean := seedProjBNM) (seedVar
              := seedProjBNV)
        seq! projConv, projBN

    let summed : Sequential (NN.Tensor.Shape.Image inC h w) (NN.Tensor.Shape.Image cfg.outC h w) :=
      addBranches main skip
    let outAct : Sequential (NN.Tensor.Shape.Image cfg.outC h w) (NN.Tensor.Shape.Image cfg.outC h
      w) :=
      activation (s := NN.Tensor.Shape.Image cfg.outC h w) cfg.activation

    have hDown' : cfg.downsample = false := by
      cases hds : cfg.downsample with
      | false => rfl
      | true => cases (hDown hds)
    simpa [hDown'] using (seq! summed, outAct)

/-- ResNet-18 style BasicBlock over batched image tensors (`N×C×H×W`). -/
def resnetBasicBlock {n inC h w : Nat} (cfg : ResNetBasicBlock)
    [NeZero n] [NeZero inC] [NeZero h] [NeZero w] [NeZero cfg.outC] :
    Sequential
      (NN.Tensor.Shape.Images n inC h w)
      (NN.Tensor.Shape.Images n cfg.outC
        (if cfg.downsample then down2 h else h)
        (if cfg.downsample then down2 w else w)) := by
  classical
  have hn : n > 0 := Nat.pos_of_ne_zero (NeZero.ne (n := n))
  -- Seed layout: conv/bn/conv/bn/(proj conv/bn) in a fixed order.
  let seedConv1K := cfg.seedBase + 0
  let seedConv1B := cfg.seedBase + 1
  let seedBN1G := cfg.seedBase + 2
  let seedBN1B := cfg.seedBase + 3
  let seedBN1M := cfg.seedBase + 4
  let seedBN1V := cfg.seedBase + 5
  let seedConv2K := cfg.seedBase + 6
  let seedConv2B := cfg.seedBase + 7
  let seedBN2G := cfg.seedBase + 8
  let seedBN2B := cfg.seedBase + 9
  let seedBN2M := cfg.seedBase + 10
  let seedBN2V := cfg.seedBase + 11
  let seedProjK := cfg.seedBase + 12
  let seedProjB := cfg.seedBase + 13
  let seedProjBNG := cfg.seedBase + 14
  let seedProjBNB := cfg.seedBase + 15
  let seedProjBNM := cfg.seedBase + 16
  let seedProjBNV := cfg.seedBase + 17

  if hDown : cfg.downsample = true then
    -- Stride-2 downsample.
    let h' := down2 h
    let w' := down2 w
    have hh' : h' > 0 := down2_pos h
    have hw' : w' > 0 := down2_pos w
    have hH' : h' ≠ 0 := Nat.ne_of_gt hh'
    have hW' : w' ≠ 0 := Nat.ne_of_gt hw'
    letI : NeZero h' := ⟨hH'⟩
    letI : NeZero w' := ⟨hW'⟩

    let conv1 : Sequential (NN.Tensor.Shape.Images n inC h w) (NN.Tensor.Shape.Images n cfg.outC h'
      w') :=
      conv3x3DownImages (n := n) (inC := inC) (outC := cfg.outC) (h := h) (w := w)
        (seedK := seedConv1K) (seedB := seedConv1B)
    let bn1 : Sequential (NN.Tensor.Shape.Images n cfg.outC h' w') (NN.Tensor.Shape.Images n
      cfg.outC h' w') :=
      TorchLean.Layers.batchNorm2dNCHW n cfg.outC h' w'
        (hN := hn)
        (hC := Nat.pos_of_ne_zero (NeZero.ne (n := cfg.outC)))
        (hH := hh') (hW := hw')
        (seedGamma := seedBN1G) (seedBeta := seedBN1B) (seedMean := seedBN1M) (seedVar := seedBN1V)
    let relu1 : Sequential (NN.Tensor.Shape.Images n cfg.outC h' w') (NN.Tensor.Shape.Images n
      cfg.outC h' w') :=
      activation (s := NN.Tensor.Shape.Images n cfg.outC h' w') cfg.activation
    let conv2 : Sequential (NN.Tensor.Shape.Images n cfg.outC h' w') (NN.Tensor.Shape.Images n
      cfg.outC h' w') :=
      conv3x3SameImages (n := n) (inC := cfg.outC) (outC := cfg.outC) (h := h') (w := w')
        (seedK := seedConv2K) (seedB := seedConv2B)
    let bn2 : Sequential (NN.Tensor.Shape.Images n cfg.outC h' w') (NN.Tensor.Shape.Images n
      cfg.outC h' w') :=
      TorchLean.Layers.batchNorm2dNCHW n cfg.outC h' w'
        (hN := hn)
        (hC := Nat.pos_of_ne_zero (NeZero.ne (n := cfg.outC)))
        (hH := hh') (hW := hw')
        (seedGamma := seedBN2G) (seedBeta := seedBN2B) (seedMean := seedBN2M) (seedVar := seedBN2V)
    let main := seq! conv1, bn1, relu1, conv2, bn2

    let projConv : Sequential (NN.Tensor.Shape.Images n inC h w) (NN.Tensor.Shape.Images n cfg.outC
      h' w') :=
      conv1x1DownImages (n := n) (inC := inC) (outC := cfg.outC) (h := h) (w := w)
        (seedK := seedProjK) (seedB := seedProjB)
    let projBN : Sequential (NN.Tensor.Shape.Images n cfg.outC h' w') (NN.Tensor.Shape.Images n
      cfg.outC h' w') :=
      TorchLean.Layers.batchNorm2dNCHW n cfg.outC h' w'
        (hN := hn)
        (hC := Nat.pos_of_ne_zero (NeZero.ne (n := cfg.outC)))
        (hH := hh') (hW := hw')
        (seedGamma := seedProjBNG) (seedBeta := seedProjBNB) (seedMean := seedProjBNM) (seedVar :=
          seedProjBNV)
    let skip := seq! projConv, projBN

    let summed : Sequential (NN.Tensor.Shape.Images n inC h w) (NN.Tensor.Shape.Images n cfg.outC h'
      w') :=
      addBranches main skip
    let outAct : Sequential (NN.Tensor.Shape.Images n cfg.outC h' w') (NN.Tensor.Shape.Images n
      cfg.outC h' w') :=
      activation (s := NN.Tensor.Shape.Images n cfg.outC h' w') cfg.activation

    -- The `if` in the return type reduces via `hDown`.
    simpa [hDown] using (seq! summed, outAct)
  else
    -- Stride-1 (no spatial downsample).
    have hh : h > 0 := Nat.pos_of_ne_zero (NeZero.ne (n := h))
    have hw : w > 0 := Nat.pos_of_ne_zero (NeZero.ne (n := w))

    let conv1 : Sequential (NN.Tensor.Shape.Images n inC h w) (NN.Tensor.Shape.Images n cfg.outC h
      w) :=
      conv3x3SameImages (n := n) (inC := inC) (outC := cfg.outC) (h := h) (w := w)
        (seedK := seedConv1K) (seedB := seedConv1B)
    let bn1 : Sequential (NN.Tensor.Shape.Images n cfg.outC h w) (NN.Tensor.Shape.Images n cfg.outC
      h w) :=
      TorchLean.Layers.batchNorm2dNCHW n cfg.outC h w
        (hN := hn)
        (hC := Nat.pos_of_ne_zero (NeZero.ne (n := cfg.outC)))
        (hH := hh) (hW := hw)
        (seedGamma := seedBN1G) (seedBeta := seedBN1B) (seedMean := seedBN1M) (seedVar := seedBN1V)
    let relu1 : Sequential (NN.Tensor.Shape.Images n cfg.outC h w) (NN.Tensor.Shape.Images n
      cfg.outC h w) :=
      activation (s := NN.Tensor.Shape.Images n cfg.outC h w) cfg.activation
    let conv2 : Sequential (NN.Tensor.Shape.Images n cfg.outC h w) (NN.Tensor.Shape.Images n
      cfg.outC h w) :=
      conv3x3SameImages (n := n) (inC := cfg.outC) (outC := cfg.outC) (h := h) (w := w)
        (seedK := seedConv2K) (seedB := seedConv2B)
    let bn2 : Sequential (NN.Tensor.Shape.Images n cfg.outC h w) (NN.Tensor.Shape.Images n cfg.outC
      h w) :=
      TorchLean.Layers.batchNorm2dNCHW n cfg.outC h w
        (hN := hn)
        (hC := Nat.pos_of_ne_zero (NeZero.ne (n := cfg.outC)))
        (hH := hh) (hW := hw)
        (seedGamma := seedBN2G) (seedBeta := seedBN2B) (seedMean := seedBN2M) (seedVar := seedBN2V)
    let main := seq! conv1, bn1, relu1, conv2, bn2

    let skip : Sequential (NN.Tensor.Shape.Images n inC h w) (NN.Tensor.Shape.Images n cfg.outC h w)
      :=
      if hEq : cfg.outC = inC then
        -- Identity shortcut.
        have hShape : NN.Tensor.Shape.Images n inC h w = NN.Tensor.Shape.Images n cfg.outC h w := by
          simp [hEq]
        Eq.ndrec (motive := fun τ => Sequential (NN.Tensor.Shape.Images n inC h w) τ)
          (_root_.Runtime.Autograd.TorchLean.NN.Seq.id (NN.Tensor.Shape.Images n inC h w)) hShape
      else
        -- Projection shortcut (1x1 + BN).
        let projConv : Sequential (NN.Tensor.Shape.Images n inC h w) (NN.Tensor.Shape.Images n
          cfg.outC h w) :=
          conv1x1SameImages (n := n) (inC := inC) (outC := cfg.outC) (h := h) (w := w)
            (seedK := seedProjK) (seedB := seedProjB)
        let projBN : Sequential (NN.Tensor.Shape.Images n cfg.outC h w) (NN.Tensor.Shape.Images n
          cfg.outC h w) :=
          TorchLean.Layers.batchNorm2dNCHW n cfg.outC h w
            (hN := hn)
            (hC := Nat.pos_of_ne_zero (NeZero.ne (n := cfg.outC)))
            (hH := hh) (hW := hw)
            (seedGamma := seedProjBNG) (seedBeta := seedProjBNB) (seedMean := seedProjBNM) (seedVar
              := seedProjBNV)
        seq! projConv, projBN

    let summed : Sequential (NN.Tensor.Shape.Images n inC h w) (NN.Tensor.Shape.Images n cfg.outC h
      w) :=
      addBranches main skip
    let outAct : Sequential (NN.Tensor.Shape.Images n cfg.outC h w) (NN.Tensor.Shape.Images n
      cfg.outC h w) :=
      activation (s := NN.Tensor.Shape.Images n cfg.outC h w) cfg.activation

    have hDown' : cfg.downsample = false := by
      cases hds : cfg.downsample with
      | false => rfl
      | true => cases (hDown hds)
    simpa [hDown'] using (seq! summed, outAct)
