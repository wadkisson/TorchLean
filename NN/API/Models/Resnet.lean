/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public

/-!
# ResNet Model Helpers (API)

This module provides a ResNet-style classifier constructor used by runnable examples.

The architecture is small enough to keep the Lean shape arguments readable while still following the
standard ResNet pattern:
- a 3×3 conv stem + BatchNorm + ReLU
- three `resnetBasicBlock`s (one downsampling)
- global average pooling
- linear classifier head

The positivity proofs below are Lean's way of recording that pooling axes are nonempty. The public
constructor is `nn.models.resnet cfg`; executable examples can decide separately whether this
heavier residual path belongs in their runtime budget.
-/

@[expose] public section

namespace NN
namespace API

open Spec Tensor

namespace nn
namespace models

/-- Configuration for a ResNet-style image classifier. -/
structure ResnetConfig where
  batch : Nat
  inC : Nat
  inH : Nat
  inW : Nat
  stemC : Nat
  numClasses : Nat
deriving Repr

/-- Channel count of the second ResNet stage. -/
def ResnetConfig.stage2C (cfg : ResnetConfig) : Nat :=
  cfg.stemC * 2

/-- Spatial height after the downsampling block. -/
def ResnetConfig.downsampledHeight (cfg : ResnetConfig) : Nat :=
  nn.blocks.strideTwoOutput cfg.inH

/-- Spatial width after the downsampling block. -/
def ResnetConfig.downsampledWidth (cfg : ResnetConfig) : Nat :=
  nn.blocks.strideTwoOutput cfg.inW

/-- Batched image input shape for the ResNet helper. -/
abbrev resnetInShape (cfg : ResnetConfig) : Shape :=
  NN.Tensor.Shape.NCHW cfg.batch cfg.inC cfg.inH cfg.inW

/-- Batched classifier-logit output shape for the ResNet helper. -/
abbrev resnetOutShape (cfg : ResnetConfig) : Shape :=
  NN.Tensor.Shape.Mat cfg.batch cfg.numClasses

/--
Build a ResNet-style image classifier.

Architecture:
`conv3x3 stem -> batch norm -> ReLU -> basic block -> downsampling basic block -> basic block
-> global average pool -> linear head`.

The explicit proof arguments are optional defaults. For concrete model configs they are solved by
`by decide`; keeping them here hides the proof arguments from runnable examples.
-/
def resnet (cfg : ResnetConfig)
    (h_batch : cfg.batch ≠ 0 := by decide)
    (h_inC : cfg.inC ≠ 0 := by decide)
    (h_inH : cfg.inH ≠ 0 := by decide)
    (h_inW : cfg.inW ≠ 0 := by decide)
    (h_stemC : cfg.stemC ≠ 0 := by decide)
    (h_stage2C : cfg.stage2C ≠ 0 := by decide) :
    nn.M (nn.Sequential (resnetInShape cfg) (resnetOutShape cfg)) := do
  letI : NeZero cfg.batch := ⟨h_batch⟩
  letI : NeZero cfg.inC := ⟨h_inC⟩
  letI : NeZero cfg.inH := ⟨h_inH⟩
  letI : NeZero cfg.inW := ⟨h_inW⟩
  letI : NeZero cfg.stemC := ⟨h_stemC⟩
  letI : NeZero cfg.stage2C := ⟨h_stage2C⟩

  let hb : cfg.batch > 0 := Nat.pos_of_ne_zero (NeZero.ne (n := cfg.batch))
  let hc : cfg.stage2C > 0 := Nat.pos_of_ne_zero (NeZero.ne (n := cfg.stage2C))
  let hDownsampledHeight : cfg.downsampledHeight > 0 := by
    simp [ResnetConfig.downsampledHeight]
  let hDownsampledWidth : cfg.downsampledWidth > 0 := by
    simp [ResnetConfig.downsampledWidth]

  let pool : nn.Sequential (NN.Tensor.Shape.Images cfg.batch cfg.stage2C cfg.downsampledHeight cfg.downsampledWidth)
      (NN.Tensor.Shape.Mat cfg.batch cfg.stage2C) :=
    nn.globalAvgPoolNCHW cfg.batch cfg.stage2C cfg.downsampledHeight cfg.downsampledWidth
      (hN := hb) (hC := hc) (hH := hDownsampledHeight) (hW := hDownsampledWidth)

  nn.Sequential![
    -- Use the ResNet helper conv so the output shape is definitionally `H×W` (not a conv-formula
    -- expression), while still allocating seeds from the `nn.M` seed stream.
    withSeedPair (fun seedK seedB =>
      _root_.NN.API.nn.pure.blocks.conv3x3SameImages (n := cfg.batch) (inC := cfg.inC) (outC := cfg.stemC)
        (h := cfg.inH) (w := cfg.inW) (seedK := seedK) (seedB := seedB)
        (kInit := .uniform (-0.1) 0.1)),
    nn.batchNorm,
    ReLU,
    nn.resnetBasicBlock
      { outC := cfg.stemC, downsample := false, activation := .relu },
    nn.resnetBasicBlock
      { outC := cfg.stage2C, downsample := true, activation := .relu },
    nn.resnetBasicBlock
      { outC := cfg.stage2C, downsample := false, activation := .relu },
    nn.lift pool,
    Linear cfg.stage2C cfg.numClasses (pfx := NN.Tensor.Shape.Vec cfg.batch)
  ]

end models
end nn

end API
end NN
