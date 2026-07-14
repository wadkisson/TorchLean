/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Core
public import NN.API.Rand
public import NN.API.TorchLean.ParamIO
public import NN.API.TorchLean.Schedulers
public import NN.GraphSpec.Models.TorchLean
public import NN.Runtime.Autograd.TorchLean
public import NN.Runtime.Autograd.Train.Dataset
public import NN.Runtime.RL
public import NN.Spec.RL.MDP
public import NN.Spec.RL.MarkovMDP
public import NN.Spec.RL.FiniteStochasticMDP
public import NN.API.Runtime.Core

import Mathlib.Algebra.Order.Algebra
import NN.Spec.Autograd.AutogradSpec

@[expose] public section

namespace NN
namespace API
namespace TorchLean

/-!
# Runtime Layer Helpers

Sequential layer constructors over the lower-level TorchLean runtime API. These keep the direct
runtime API available while the higher-level `NN.API.Public.nn` namespace provides named-field configs.
-/

namespace Layers

/-!
### Sequential Layer Helpers

`Runtime.Autograd.TorchLean.NN` exposes *layers* (`LayerDef σ τ`) and *sequential models*
(`Seq σ τ`). This namespace provides direct `Seq` constructors and common derived shapes such as
`flattenLinear`.

For the documented user API with named-field configs and blocks, see
`NN.API.Public` under `API.nn`.
-/

/-- Lift a single layer into a sequential model. -/
def of {σ τ : Spec.Shape} (layer : API.TorchLean.LayerCore.LayerDef σ τ) :
    API.TorchLean.LayerCore.Seq σ τ :=
  API.TorchLean.LayerCore.singleLayer layer

namespace Internal

/-- Checked reshape layer used by generic runtime adapters. -/
def reshapeLayer (source target : Spec.Shape)
    (sameSize : Spec.Shape.size source = Spec.Shape.size target) :
    API.TorchLean.LayerCore.LayerDef source target :=
  { kind := "Reshape"
    paramShapes := []
    initParams := .nil
    paramRequiresGrad := []
    forward := fun _ {α} _ _ => fun {m} _ _ => fun x =>
      API.TorchLean.reshape (m := m) (α := α) (s₁ := source) (s₂ := target) x sameSize }

/-- Checked reshape as a one-layer sequential model. -/
def reshapeSeq (source target : Spec.Shape)
    (sameSize : Spec.Shape.size source = Spec.Shape.size target) :
    API.TorchLean.LayerCore.Seq source target :=
  of (reshapeLayer source target sameSize)

end Internal

/-- Linear layer over vectors (returns a 1-layer `Seq`). -/
def linear (inDim outDim : Nat) (seedW seedB : Nat := 0) :
    API.TorchLean.LayerCore.Seq (.dim inDim .scalar) (.dim outDim .scalar) :=
  of <| API.TorchLean.LayerCore.linear inDim outDim seedW seedB

/-- Pointwise ReLU activation, preserving the input shape. -/
def relu {s : Spec.Shape} : API.TorchLean.LayerCore.Seq s s :=
  of <| API.TorchLean.LayerCore.relu (s := s)

/-- Elementwise SiLU/Swish. -/
def silu {s : Spec.Shape} : API.TorchLean.LayerCore.Seq s s :=
  of <| API.TorchLean.LayerCore.silu (s := s)

/-- Pointwise GELU activation, preserving the input shape. -/
def gelu {s : Spec.Shape} : API.TorchLean.LayerCore.Seq s s :=
  of <| API.TorchLean.LayerCore.gelu (s := s)

/-- Pointwise logistic sigmoid activation, preserving the input shape. -/
def sigmoid {s : Spec.Shape} : API.TorchLean.LayerCore.Seq s s :=
  of <| API.TorchLean.LayerCore.sigmoid (s := s)

/-- Pointwise hyperbolic tangent activation, preserving the input shape. -/
def tanh {s : Spec.Shape} : API.TorchLean.LayerCore.Seq s s :=
  of <| API.TorchLean.LayerCore.tanh (s := s)

/-- Softmax over the flattened tensor entries for the current runtime layer convention. -/
def softmax {s : Spec.Shape} : API.TorchLean.LayerCore.Seq s s :=
  of <| API.TorchLean.LayerCore.softmax (s := s)

/-- Pointwise square map, preserving the input shape. -/
def square {s : Spec.Shape} : API.TorchLean.LayerCore.Seq s s :=
  of <| API.TorchLean.LayerCore.square (s := s)

/-- Reduce-sum to a scalar. -/
def sum {s : Spec.Shape} : API.TorchLean.LayerCore.Seq s Spec.Shape.scalar :=
  of <| API.TorchLean.LayerCore.sum (s := s)

/-- Flatten any input shape into a 1D vector of length `Spec.Shape.size s`. -/
def flatten {s : Spec.Shape} : API.TorchLean.LayerCore.Seq s (.dim (Spec.Shape.size s) .scalar) :=
  of <| API.TorchLean.LayerCore.flatten (s := s)

/-- Dropout layer that is active in training mode and identity in eval mode. -/
def dropout {s : Spec.Shape} (p : Float) (seed : Nat := 0) : API.TorchLean.LayerCore.Seq s s :=
  of <| API.TorchLean.LayerCore.dropout (s := s) p seed

/-- `Flatten -> Linear` head, with the input dimension computed from the input shape. -/
def flattenLinear {s : Spec.Shape} (outDim : Nat) (seedW seedB : Nat := 0) :
    API.TorchLean.LayerCore.Seq s (.dim outDim .scalar) :=
  (flatten (s := s)) >>> (linear (Spec.Shape.size s) outDim seedW seedB)

/--
Sequence-wise layer normalization.

PyTorch analogy: `torch.nn.LayerNorm(embedDim)` applied to each position in a sequence.
-/
def layerNorm (batch seqLen embedDim : Nat)
    {hSeq : seqLen > 0} {hEmbed : embedDim > 0}
    (seedGamma seedBeta : Nat := 0) :
    API.TorchLean.LayerCore.Seq (.dim batch (.dim seqLen (.dim embedDim .scalar)))
      (.dim batch (.dim seqLen (.dim embedDim .scalar))) :=
  of <| API.TorchLean.LayerCore.layerNorm (batch := batch)
    (seqLen := seqLen) (embedDim := embedDim)
    (h_seq_pos := hSeq) (h_embed_pos := hEmbed)
    (seedGamma := seedGamma) (seedBeta := seedBeta)

/--
Sequence-wise RMS normalization.

PyTorch analogy: an `RMSNorm`-style layer over `(seqLen × embedDim)` tensors.
-/
def rmsNorm (batch seqLen embedDim : Nat)
    {hSeq : seqLen > 0} {hEmbed : embedDim > 0}
    (seedGamma : Nat := 0) :
    API.TorchLean.LayerCore.Seq (.dim batch (.dim seqLen (.dim embedDim .scalar)))
      (.dim batch (.dim seqLen (.dim embedDim .scalar))) :=
  of <| API.TorchLean.LayerCore.rmsNorm (batch := batch)
    (seqLen := seqLen) (embedDim := embedDim)
    (h_seq_pos := hSeq) (h_embed_pos := hEmbed)
    (seedGamma := seedGamma)

/-- Instance normalization over a flattened `(leading, channels, spatial)` representation. -/
def instanceNormChannelFirst (leadingSize channels spatialSize : Nat)
    {hLeading : leadingSize > 0} {hChannels : channels > 0} {hSpatial : spatialSize > 0}
    (seedGamma seedBeta : Nat := 0) :
    API.TorchLean.LayerCore.Seq
      (.dim leadingSize (.dim channels (.dim spatialSize .scalar)))
      (.dim leadingSize (.dim channels (.dim spatialSize .scalar))) :=
  let source : Spec.Shape := .dim leadingSize (.dim channels (.dim spatialSize .scalar))
  let kernelShape : Spec.Shape :=
    .dim leadingSize (.dim channels (.dim spatialSize (.dim 1 .scalar)))
  API.TorchLean.LayerCore.Seq.comp
    (Internal.reshapeSeq source kernelShape (by simp [source, kernelShape, Spec.Shape.size]))
    (API.TorchLean.LayerCore.Seq.comp
      (of (API.TorchLean.LayerCore.instanceNorm2dNchw
        (n := leadingSize) (c := channels) (h := spatialSize) (w := 1)
        (h_n_pos := hLeading) (h_c_pos := hChannels) (h_h_pos := hSpatial)
        (h_w_pos := by decide) (seedGamma := seedGamma) (seedBeta := seedBeta)))
      (Internal.reshapeSeq kernelShape source (by simp [source, kernelShape, Spec.Shape.size])))

/-- Group normalization over a flattened `(leading, channels, spatial)` representation. -/
def groupNormChannelFirst (leadingSize channels spatialSize groups : Nat)
    {hLeading : leadingSize > 0} {hChannels : channels > 0} {hSpatial : spatialSize > 0}
    {hGroups : groups > 0} (hGroupsLe : channels ≥ groups) (hDiv : channels % groups = 0)
    (seedGamma seedBeta : Nat := 0) :
    API.TorchLean.LayerCore.Seq
      (.dim leadingSize (.dim channels (.dim spatialSize .scalar)))
      (.dim leadingSize (.dim channels (.dim spatialSize .scalar))) :=
  let source : Spec.Shape := .dim leadingSize (.dim channels (.dim spatialSize .scalar))
  let kernelShape : Spec.Shape :=
    .dim leadingSize (.dim channels (.dim spatialSize (.dim 1 .scalar)))
  API.TorchLean.LayerCore.Seq.comp
    (Internal.reshapeSeq source kernelShape (by simp [source, kernelShape, Spec.Shape.size]))
    (API.TorchLean.LayerCore.Seq.comp
      (of (API.TorchLean.LayerCore.groupNorm2dNchw
        (n := leadingSize) (c := channels) (h := spatialSize) (w := 1) (groups := groups)
        (h_n_pos := hLeading) (h_c_pos := hChannels) (h_h_pos := hSpatial)
        (h_w_pos := by decide) (h_g_pos := hGroups) hGroupsLe hDiv
        (seedGamma := seedGamma) (seedBeta := seedBeta)))
      (Internal.reshapeSeq kernelShape source (by simp [source, kernelShape, Spec.Shape.size])))

/-- Batch normalization over a flattened `(leading, channels, spatial)` representation. -/
def batchNormChannelFirst (leadingSize channels spatialSize : Nat)
    {hLeading : leadingSize > 0} {hChannels : channels > 0} {hSpatial : spatialSize > 0}
    (seedGamma seedBeta seedMean seedVar : Nat := 0) :
    API.TorchLean.LayerCore.Seq
      (.dim leadingSize (.dim channels (.dim spatialSize .scalar)))
      (.dim leadingSize (.dim channels (.dim spatialSize .scalar))) :=
  let source : Spec.Shape := .dim leadingSize (.dim channels (.dim spatialSize .scalar))
  let kernelShape : Spec.Shape :=
    .dim leadingSize (.dim channels (.dim spatialSize (.dim 1 .scalar)))
  API.TorchLean.LayerCore.Seq.comp
    (Internal.reshapeSeq source kernelShape (by simp [source, kernelShape, Spec.Shape.size]))
    (API.TorchLean.LayerCore.Seq.comp
      (of (API.TorchLean.LayerCore.batchNorm2dNchwMode
        (n := leadingSize) (c := channels) (h := spatialSize) (w := 1)
        (h_n_pos := hLeading) (h_c_pos := hChannels) (h_h_pos := hSpatial)
        (h_w_pos := by decide) (seedGamma := seedGamma) (seedBeta := seedBeta)
        (seedMean := seedMean) (seedVar := seedVar)))
      (Internal.reshapeSeq kernelShape source (by simp [source, kernelShape, Spec.Shape.size])))

/--
Multi-head self-attention over sequence embeddings.

PyTorch analogy: `torch.nn.MultiheadAttention(embed_dim=dModel, num_heads=numHeads)` in self-
attention mode, with explicit `n × dModel` shapes.
-/
def attention (batch n dModel numHeads headDim : Nat)
    {hN : n ≠ 0}
    (seedW : Nat := 0)
    (mask : Option (_root_.Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    API.TorchLean.LayerCore.Seq (.dim batch (.dim n (.dim dModel .scalar)))
      (.dim batch (.dim n (.dim dModel .scalar))) :=
  of <| API.TorchLean.LayerCore.multiHeadAttention (batch := batch)
    (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
    (h1 := hN) (seedW := seedW) (mask := mask)

end Layers

end TorchLean
end API
end NN
