/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Capsule

/-!
# Reference and Portable Backend Capsules

Portable capsules for paths that do not require CUDA or LibTorch.

These are not meant to win benchmarks. They keep TorchLean runnable on CPU-only machines and give
the planner a clear fallback vocabulary for reference/spec-aligned execution.
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

/-- Reference ReLU activation. -/
def relu : KernelCapsule :=
  referenceCapsule
    "reference.relu"
    .relu
    "Spec.relu / pointwise max(x, 0)"
    "Reference ReLU follows pointwise tensor semantics."
    "TorchLean tape supplies the VJP."

/-- Reference GELU activation. -/
def gelu : KernelCapsule :=
  referenceCapsule
    "reference.gelu"
    .gelu
    "Spec.gelu"
    "Reference GELU follows the documented runtime approximation contract."
    "TorchLean tape supplies the VJP."

def add : KernelCapsule := referencePointwiseCapsule .add
def sub : KernelCapsule := referencePointwiseCapsule .sub
def mul : KernelCapsule := referencePointwiseCapsule .mul
def scale : KernelCapsule := referencePointwiseCapsule .scale
def abs : KernelCapsule := referencePointwiseCapsule .abs
def sqrt : KernelCapsule := referencePointwiseCapsule .sqrt
def clamp : KernelCapsule := referencePointwiseCapsule .clamp
def max : KernelCapsule := referencePointwiseCapsule .max
def min : KernelCapsule := referencePointwiseCapsule .min
def sigmoid : KernelCapsule := referencePointwiseCapsule .sigmoid
def tanh : KernelCapsule := referencePointwiseCapsule .tanh
def softplus : KernelCapsule := referencePointwiseCapsule .softplus
def exp : KernelCapsule := referencePointwiseCapsule .exp
def log : KernelCapsule := referencePointwiseCapsule .log
def sin : KernelCapsule := referencePointwiseCapsule .sin
def cos : KernelCapsule := referencePointwiseCapsule .cos
def inv : KernelCapsule := referencePointwiseCapsule .inv
def safeLog : KernelCapsule := referencePointwiseCapsule .safeLog
def logSoftmax : KernelCapsule := referencePointwiseCapsule .logSoftmax

/-- Reference softmax path. -/
def softmax : KernelCapsule :=
  referenceCapsule
    "reference.softmax"
    .softmax
    "Spec.softmax"
    "Reference softmax follows the row/axis normalization contract."
    "TorchLean tape supplies the VJP."

def sum : KernelCapsule := referenceReductionCapsule .sum
def reduceSum : KernelCapsule := referenceReductionCapsule .reduceSum
def reduceMean : KernelCapsule := referenceReductionCapsule .reduceMean

def flatten : KernelCapsule := referenceViewCapsule .flatten
def reshape : KernelCapsule := referenceViewCapsule .reshape
def permute : KernelCapsule := referenceViewCapsule .permute
def transpose2d : KernelCapsule := referenceViewCapsule .transpose2d
def swapAdjacentAtDepth : KernelCapsule := referenceViewCapsule .swapAdjacentAtDepth
def transpose3dFirstToLast : KernelCapsule := referenceViewCapsule .transpose3dFirstToLast
def transpose3dLastToFirst : KernelCapsule := referenceViewCapsule .transpose3dLastToFirst
def transpose3dLastTwo : KernelCapsule := referenceViewCapsule .transpose3dLastTwo
def broadcastTo : KernelCapsule := referenceViewCapsule .broadcastTo
def concatVectors : KernelCapsule := referenceViewCapsule .concatVectors
def concatLeadingAxis : KernelCapsule := referenceViewCapsule .concatLeadingAxis
def sliceLeadingAxisRange : KernelCapsule := referenceViewCapsule .sliceLeadingAxisRange
def gatherScalar : KernelCapsule := referenceViewCapsule .gatherScalar
def gatherScalarNat : KernelCapsule := referenceViewCapsule .gatherScalarNat
def gatherRow : KernelCapsule := referenceViewCapsule .gatherRow
def gatherVecNat : KernelCapsule := referenceViewCapsule .gatherVecNat
def gatherRowsNat : KernelCapsule := referenceViewCapsule .gatherRowsNat
def scatterAddVec : KernelCapsule := referenceViewCapsule .scatterAddVec
def scatterAddRow : KernelCapsule := referenceViewCapsule .scatterAddRow

def randUniform : KernelCapsule :=
  referenceForwardOnlyCapsule
    .randUniform
    "IR.randUniform"
    "Reference deterministic random-uniform tensors follow the seeded spec/runtime contract."

def bernoulliMask : KernelCapsule :=
  referenceForwardOnlyCapsule
    .bernoulliMask
    "IR.bernoulliMask"
    "Reference deterministic Bernoulli masks follow the seeded spec/runtime contract."

/-- Reference matmul path. -/
def matmul : KernelCapsule :=
  referenceCapsule
    "reference.matmul"
    .matmul
    "Spec.matmul / IR.matmul"
    "Portable matmul follows the spec-level matrix product contract."
    "TorchLean tape supplies the VJP."

/-- Reference batched matrix multiplication. -/
def bmm : KernelCapsule :=
  referenceCapsule
    "reference.bmm"
    .bmm
    "Spec.bmm / batched matrix product"
    "Reference batched matrix multiplication follows the spec-level contract."
    "TorchLean tape supplies the VJP."

/-- Reference linear layer. -/
def linear : KernelCapsule :=
  referenceCapsule
    "reference.linear"
    .linear
    "Spec.linear"
    "Reference linear follows the matvec/matmul plus bias contract."
    "TorchLean tape supplies the VJP."

/-- Reference mean-squared-error loss. -/
def mseLoss : KernelCapsule :=
  referenceCapsule
    "reference.mse_loss"
    .mseLoss
    "Spec.mseLoss"
    "Reference MSE follows the mean squared residual contract."
    "TorchLean tape supplies the VJP."

/-- Reference layer normalization. -/
def layerNorm : KernelCapsule :=
  referenceCapsule
    "reference.layer_norm"
    .layerNorm
    "Spec.layerNorm"
    "Reference LayerNorm follows the per-row normalization contract."
    "TorchLean tape supplies the VJP."

/-- Reference batch normalization. -/
def batchNorm : KernelCapsule :=
  referenceCapsule
    "reference.batch_norm"
    .batchNorm
    "Spec.batchNorm"
    "Reference BatchNorm follows the channel-first normalization contract."
    "TorchLean tape supplies the VJP."

/-- Reference channel-first BatchNorm runtime op. -/
def batchNormChannelFirst : KernelCapsule :=
  referenceCapsule
    "reference.batchnorm_channel_first"
    .batchNormChannelFirst
    "Spec.batchNorm / channel-first runtime contract"
    "Reference channel-first BatchNorm follows the image/channel normalization contract."
    "TorchLean tape supplies the VJP."

/-- Reference generic channel-first convolution. -/
def conv : KernelCapsule := referenceConvPoolCapsule .conv

/-- Reference 2D convolution. -/
def conv2d : KernelCapsule :=
  referenceCapsule
    "reference.conv2d"
    .conv2d
    "Spec.conv2d"
    "Reference Conv2D follows the channel-first runtime contract."
    "TorchLean tape supplies the VJP."

/-- Reference generic channel-first transpose convolution. -/
def convTranspose : KernelCapsule := referenceConvPoolCapsule .convTranspose

/-- Reference 2D transpose convolution. -/
def convTranspose2d : KernelCapsule := referenceConvPoolCapsule .convTranspose2d

/-- Reference max pooling. -/
def maxPool : KernelCapsule :=
  referenceCapsule
    "reference.max_pool"
    .maxPool
    "Spec.maxPool"
    "Reference max-pooling follows the channel-first window contract."
    "TorchLean tape supplies the VJP."

/-- Reference 2D max pooling. -/
def maxPool2d : KernelCapsule := referenceConvPoolCapsule .maxPool2d

/-- Reference padded 2D max pooling. -/
def maxPool2dPad : KernelCapsule := referenceConvPoolCapsule .maxPool2dPad

/-- Reference smooth max pooling. -/
def smoothMaxPool : KernelCapsule := referenceConvPoolCapsule .smoothMaxPool

/-- Reference smooth 2D max pooling. -/
def smoothMaxPool2d : KernelCapsule := referenceConvPoolCapsule .smoothMaxPool2d

/-- Reference average pooling. -/
def avgPool : KernelCapsule :=
  referenceCapsule
    "reference.avg_pool"
    .avgPool
    "Spec.avgPool"
    "Reference average-pooling follows the channel-first window contract."
    "TorchLean tape supplies the VJP."

/-- Reference 2D average pooling. -/
def avgPool2d : KernelCapsule := referenceConvPoolCapsule .avgPool2d

/-- Reference padded 2D average pooling. -/
def avgPool2dPad : KernelCapsule := referenceConvPoolCapsule .avgPool2dPad

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
  [ matmul
  , bmm
  , linear
  , mseLoss
  , relu
  , gelu
  , add
  , sub
  , mul
  , scale
  , abs
  , sqrt
  , clamp
  , max
  , min
  , sigmoid
  , tanh
  , softplus
  , exp
  , log
  , sin
  , cos
  , inv
  , safeLog
  , logSoftmax
  , softmax
  , sum
  , reduceSum
  , reduceMean
  , flatten
  , reshape
  , permute
  , transpose2d
  , swapAdjacentAtDepth
  , transpose3dFirstToLast
  , transpose3dLastToFirst
  , transpose3dLastTwo
  , broadcastTo
  , concatVectors
  , concatLeadingAxis
  , sliceLeadingAxisRange
  , gatherScalar
  , gatherScalarNat
  , gatherRow
  , gatherVecNat
  , gatherRowsNat
  , scatterAddVec
  , scatterAddRow
  , randUniform
  , bernoulliMask
  , layerNorm
  , batchNorm
  , batchNormChannelFirst
  , conv
  , conv2d
  , convTranspose
  , convTranspose2d
  , maxPool
  , maxPool2d
  , maxPool2dPad
  , smoothMaxPool
  , smoothMaxPool2d
  , avgPool
  , avgPool2d
  , avgPool2dPad
  , attention
  ]

end Reference
end Backend
end NN
