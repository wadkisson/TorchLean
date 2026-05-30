/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.SelfSupervised
public import NN.API.Models.Generative
public import NN.API.Models.Vit

/-!
# Self-Supervised Model Constructors

Most SSL machinery belongs in `NN.API.ssl`: masks, tensor-to-training-sample transforms, and
objective-facing helpers should work with any compatible model.

This file keeps architecture-level conveniences. The compact MAE constructor below is useful for
examples, but the SSL idea itself is not tied to this model.
-/

@[expose] public section

namespace NN
namespace API

open Spec Tensor

namespace nn
namespace models

/-! ## ViT-MAE -/

/--
Configuration for a compact ViT-MAE image reconstructor.

The input/output contract is MAE-style:
- input: a masked image tensor, `N×C×H×W`;
- output: a flattened reconstruction vector, `N×reconDim`.

`reconDim` can be the full image size (`C*H*W`) or a prefix for faster experiments.
-/
structure VitMaeConfig where
  batch : Nat
  inC : Nat
  inH : Nat
  inW : Nat
  patchH : Nat
  patchW : Nat
  stride : Nat
  padding : Nat := 0
  dModel : Nat
  reconDim : Nat
  numHeads : Nat
  headDim : Nat
  ffnHidden : Nat
deriving Repr

/-- Convert a ViT-MAE configuration into the classifier-style ViT config used by the encoder. -/
def VitMaeConfig.toVitConfig (cfg : VitMaeConfig) : VitConfig :=
  { batch := cfg.batch
    inC := cfg.inC
    inH := cfg.inH
    inW := cfg.inW
    patchH := cfg.patchH
    patchW := cfg.patchW
    stride := cfg.stride
    padding := cfg.padding
    dModel := cfg.dModel
    outDim := cfg.reconDim
    numHeads := cfg.numHeads
    headDim := cfg.headDim
    ffnHidden := cfg.ffnHidden }

/-- Batched masked-image input shape for the ViT-MAE helper. -/
abbrev vitMaeInShape (cfg : VitMaeConfig) : Shape :=
  NN.Tensor.Shape.Images cfg.batch cfg.inC cfg.inH cfg.inW

/-- Batched reconstruction-vector output shape for the ViT-MAE helper. -/
abbrev vitMaeOutShape (cfg : VitMaeConfig) : Shape :=
  NN.Tensor.Shape.Mat cfg.batch cfg.reconDim

/-- Number of patch tokens produced by the ViT-MAE patch embedding. -/
def VitMaeConfig.seqLen (cfg : VitMaeConfig) : Nat :=
  cfg.toVitConfig.seqLen

/-- Flattened encoded-token representation size before the MAE decoder head. -/
def VitMaeConfig.flatDim (cfg : VitMaeConfig) : Nat :=
  cfg.toVitConfig.flatDim

/--
Compact ViT-MAE image reconstructor.

This is a real image/patch transformer path:
1. patch embedding by strided convolution,
2. tokenization to `N×numPatches×dModel`,
3. one transformer encoder block,
4. a linear pixel decoder from encoded patch tokens to a reconstruction vector.

The masking objective is provided by `NN.API.ssl.imagePatchMaeSample`, so any image model with this
input/output shape can use the same SSL training sample.
-/
def vitMaskedAutoencoder (cfg : VitMaeConfig)
    (h_inC : cfg.inC ≠ 0 := by decide)
    (h_patchH : cfg.patchH ≠ 0 := by decide)
    (h_patchW : cfg.patchW ≠ 0 := by decide)
    (h_seqLen : cfg.seqLen ≠ 0 := by decide)
    (h_dModel : cfg.dModel ≠ 0 := by decide) :
    nn.M (nn.Sequential (vitMaeInShape cfg) (vitMaeOutShape cfg)) :=
  let vitCfg := cfg.toVitConfig
  letI : NeZero cfg.inC := ⟨h_inC⟩
  letI : NeZero cfg.patchH := ⟨h_patchH⟩
  letI : NeZero cfg.patchW := ⟨h_patchW⟩
  letI : NeZero vitCfg.seqLen := ⟨h_seqLen⟩
  letI : NeZero cfg.dModel := ⟨h_dModel⟩
  letI : NeZero vitCfg.dModel := ⟨by simpa [VitMaeConfig.toVitConfig] using h_dModel⟩
  nn.sequential![
    nn.conv { outC := cfg.dModel, kH := cfg.patchH, kW := cfg.patchW, stride := cfg.stride, padding := cfg.padding },
    nn.lift (nn.of (nchwToTokens vitCfg)),
    nn.transformerEncoderBlock
      { numHeads := cfg.numHeads
        headDim := cfg.headDim
        ffnHidden := cfg.ffnHidden
        activation := .gelu
        dropout? := none },
    nn.flattenBatch,
    nn.linear cfg.flatDim cfg.reconDim (pfx := NN.Tensor.Shape.Vec cfg.batch)
  ]

/--
Compact vector masked autoencoder.

Architecturally this reuses the vector autoencoder body; the self-supervised part is in
`NN.API.ssl.vectorMaeSample` or `NN.API.ssl.tensorPrefixMaeSample`, which mask the input while
keeping the original tensor content as the target.
-/
def vectorMaskedAutoencoder (cfg : VectorGenerativeConfig) :
    nn.M (nn.Sequential (vectorDataShape cfg) (vectorDataShape cfg)) :=
  vectorAutoencoder cfg

end models
end nn

end API
end NN
