/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Capsule
public import NN.Backend.OpCatalog

/-!
# Reference and Portable Backend Capsules

Portable capsules for paths that do not require CUDA or LibTorch.

These are not meant to win benchmarks. They keep TorchLean runnable on CPU-only machines and give
the planner a clear fallback vocabulary for reference/spec-aligned execution.

Catalog ops (`OpCatalog.*`) are stamped through the builders below. Specials with custom summaries
stay explicit.
-/

@[expose] public section

namespace NN
namespace Backend
namespace Reference

def referenceCapsule
    (name : String) (op : BackendOp) (specName valueSummary vjpSummary : String)
    (vjpMode : VJPMode := .torchLeanTape) : KernelCapsule :=
  { name
    op
    provider := .reference
    device := .cpu
    specName
    trustLevel := .checked
    supportsForward := true
    vjpMode
    shapeContract :=
      { claim := .shapeSafety op
        summary := "Shapes are checked in Lean before entering the portable runtime path."
        evidence := .runtimeGuard "portable runtime shape checks" }
    layoutContract :=
      { claim := .layoutCompatibility op .canonicalTensor
        summary := "Portable paths use TorchLean's canonical tensor representation."
        evidence := .runtimeGuard "typed tensor layout" }
    valueContract :=
      { claim := .valueRefinement op specName
        summary := valueSummary
        evidence := .testSuite "NN.Tests.Runtime.Floats.Suite" }
    vjpContract :=
      { claim := .vjpRefinement op specName vjpMode
        summary := vjpSummary
        evidence := .testSuite "NN.Tests.Runtime.Floats.Suite" }
    notes := "Reference/portable capsules are the cross-platform fallback, not the scaling path." }

def referencePointwiseCapsule (op : BackendOp) : KernelCapsule :=
  referenceCapsule
    s!"reference.{op.name}"
    op
    s!"Spec.{op.name}"
    s!"Reference `{op.name}` follows the pointwise tensor contract."
    s!"TorchLean tape supplies the `{op.name}` VJP where differentiable."

def referenceReductionCapsule (op : BackendOp) : KernelCapsule :=
  referenceCapsule
    s!"reference.{op.name}"
    op
    s!"IR.{op.name} / Spec reduction contract"
    s!"Reference `{op.name}` follows the explicit reduction shape contract."
    s!"TorchLean tape supplies the `{op.name}` adjoint where differentiable."

def referenceViewCapsule (op : BackendOp) : KernelCapsule :=
  referenceCapsule
    s!"reference.{op.name}"
    op
    s!"IR.{op.name} shape/layout contract"
    s!"Reference `{op.name}` follows the explicit shape/layout contract."
    s!"TorchLean tape supplies the `{op.name}` adjoint where differentiable."

def referenceForwardOnlyCapsule (op : BackendOp) (specName valueSummary : String) : KernelCapsule :=
  referenceCapsule
    s!"reference.{op.name}"
    op
    specName
    valueSummary
    s!"Reference `{op.name}` is a forward-only capsule with no registered VJP."
    .none

def referenceConvPoolCapsule (op : BackendOp) : KernelCapsule :=
  referenceCapsule
    s!"reference.{op.name}"
    op
    s!"Spec.{op.name} / channel-first convolution-pooling contract"
    s!"Reference `{op.name}` follows the channel-first runtime contract."
    s!"TorchLean tape supplies the `{op.name}` VJP where differentiable."

/-- Reference ReLU activation (kept named for backend tests). -/
def relu : KernelCapsule :=
  referenceCapsule
    "reference.relu" .relu "Spec.relu / pointwise max(x, 0)"
    "Reference ReLU follows pointwise tensor semantics."
    "TorchLean tape supplies the VJP."

def gelu : KernelCapsule :=
  referenceCapsule
    "reference.gelu" .gelu "Spec.gelu"
    "Reference GELU follows the documented runtime approximation contract."
    "TorchLean tape supplies the VJP."

def softmax : KernelCapsule :=
  referenceCapsule
    "reference.softmax" .softmax "Spec.softmax"
    "Reference softmax follows the row/axis normalization contract."
    "TorchLean tape supplies the VJP."

def randUniform : KernelCapsule :=
  referenceForwardOnlyCapsule .randUniform "IR.randUniform"
    "Reference deterministic random-uniform tensors follow the seeded spec/runtime contract."

def bernoulliMask : KernelCapsule :=
  referenceForwardOnlyCapsule .bernoulliMask "IR.bernoulliMask"
    "Reference deterministic Bernoulli masks follow the seeded spec/runtime contract."

def matmul : KernelCapsule :=
  referenceCapsule
    "reference.matmul" .matmul "Spec.matmul / IR.matmul"
    "Portable matmul follows the spec-level matrix product contract."
    "TorchLean tape supplies the VJP."

def bmm : KernelCapsule :=
  referenceCapsule
    "reference.bmm" .bmm "Spec.bmm / batched matrix product"
    "Reference batched matrix multiplication follows the spec-level contract."
    "TorchLean tape supplies the VJP."

def linear : KernelCapsule :=
  referenceCapsule
    "reference.linear" .linear "Spec.linear"
    "Reference linear follows the matvec/matmul plus bias contract."
    "TorchLean tape supplies the VJP."

def mseLoss : KernelCapsule :=
  referenceCapsule
    "reference.mse_loss" .mseLoss "Spec.mseLoss"
    "Reference MSE follows the mean squared residual contract."
    "TorchLean tape supplies the VJP."

def layerNorm : KernelCapsule :=
  referenceCapsule
    "reference.layer_norm" .layerNorm "Spec.layerNorm"
    "Reference LayerNorm follows the per-row normalization contract."
    "TorchLean tape supplies the VJP."

def batchNorm : KernelCapsule :=
  referenceCapsule
    "reference.batch_norm" .batchNorm "Spec.batchNorm"
    "Reference BatchNorm follows the channel-first normalization contract."
    "TorchLean tape supplies the VJP."

def batchNormChannelFirst : KernelCapsule :=
  referenceCapsule
    "reference.batchnorm_channel_first" .batchNormChannelFirst
    "Spec.batchNorm / channel-first runtime contract"
    "Reference channel-first BatchNorm follows the image/channel normalization contract."
    "TorchLean tape supplies the VJP."

def conv2d : KernelCapsule :=
  referenceCapsule
    "reference.conv2d" .conv2d "Spec.conv2d"
    "Reference Conv2D follows the channel-first runtime contract."
    "TorchLean tape supplies the VJP."

def maxPool : KernelCapsule :=
  referenceCapsule
    "reference.max_pool" .maxPool "Spec.maxPool"
    "Reference max-pooling follows the channel-first window contract."
    "TorchLean tape supplies the VJP."

def avgPool : KernelCapsule :=
  referenceCapsule
    "reference.avg_pool" .avgPool "Spec.avgPool"
    "Reference average-pooling follows the channel-first window contract."
    "TorchLean tape supplies the VJP."

/-- Reference attention path using the composed TorchLean expression. -/
def attention : KernelCapsule :=
  { name := "reference.attention"
    op := .scaledDotProductAttention
    provider := .reference
    device := .cpu
    specName := "Spec.scaledDotProductAttention"
    trustLevel := .checked
    supportsForward := true
    vjpMode := .torchLeanTape
    shapeContract := ContractDescriptor.guarded (.shapeSafety .scaledDotProductAttention)
      "Q/K/V and mask shapes are checked by the typed tensor layer."
      "typed attention shapes"
    layoutContract := ContractDescriptor.guarded
      (.layoutCompatibility .scaledDotProductAttention .canonicalTensor)
      "Reference attention uses TorchLean tensor semantics rather than a foreign layout."
      "typed tensor layout"
    valueContract := ContractDescriptor.tested
      (.valueRefinement .scaledDotProductAttention "Spec.scaledDotProductAttention")
      "Composed reference attention uses hard-mask zero-numerator semantics."
      "NN.Tests.Runtime.Floats.Suite"
    vjpContract := ContractDescriptor.tested
      (.vjpRefinement .scaledDotProductAttention "Spec.scaledDotProductAttentionBackward"
        .torchLeanTape)
      "TorchLean tape supplies the composed VJP."
      "NN.Tests.Runtime.Floats.Suite"
    notes := "This is the CPU/reference attention contract." }

/-- Cross-platform reference capsules. -/
def capsules : List KernelCapsule :=
  [ matmul, bmm, linear, mseLoss, relu, gelu, softmax
  , randUniform, bernoulliMask
  , layerNorm, batchNorm, batchNormChannelFirst
  , conv2d, maxPool, avgPool, attention ]
  ++ (OpCatalog.pointwiseOps ++ OpCatalog.referenceExtraPointwiseOps).map referencePointwiseCapsule
  ++ OpCatalog.reductionOps.map referenceReductionCapsule
  ++ OpCatalog.viewOps.map referenceViewCapsule
  ++ OpCatalog.convPoolOps.map referenceConvPoolCapsule

end Reference
end Backend
end NN
