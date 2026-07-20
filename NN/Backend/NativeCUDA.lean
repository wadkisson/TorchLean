/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Capsule

/-!
# Native CUDA Backend Capsules

Capsule metadata for TorchLean's native CUDA runtime provider.

These capsules describe kernels that currently live under `csrc/cuda/**` and are exposed
to Lean through `NN.Runtime.Autograd.Engine.Cuda.*`. The C/CUDA source still owns the implementation;
this module gives the planner a typed, inspectable contract layer for those implementation choices.
-/

@[expose] public section

namespace NN
namespace Backend
namespace NativeCUDA

def nativeCapsule
    (name : String) (op : BackendOp) (specName valueSummary vjpSummary : String)
    (vjpMode : VJPMode := .backendVJP) : KernelCapsule :=
  { name
    op
    provider := .nativeCuda
    device := .cuda
    trustLevel := .checked
    supportsForward := true
    vjpMode
    shapeContract :=
      { claim := .shapeSafety op
        summary := "Inputs and outputs are checked against explicit UInt32 dimensions."
        evidence := .runtimeGuard "CUDA FFI size/rank checks at the Lean/native boundary" }
    layoutContract :=
      { claim := .layoutCompatibility op .flatRowMajor
        summary := "CUDA buffers are contiguous flat float32 buffers."
        evidence := .runtimeGuard "flat row-major Cuda.Buffer layout checks" }
    valueContract :=
      { claim := .valueRefinement op specName
        summary := valueSummary
        evidence := .testSuite "NN.Tests.Runtime.Cuda.Suite" }
    vjpContract :=
      { claim := .vjpRefinement op specName vjpMode
        summary := vjpSummary
        evidence := .testSuite "NN.Tests.Runtime.Cuda.Suite" }
    numericalPolicy :=
      { rounding := .nearestEven
        subnormals := .implementationDefined
        contraction := .implementationDefined
        reduction := .notApplicable }
    notes := "Native CUDA code is an FFI boundary; the capsule records the contract TorchLean checks." }

def nativePointwiseCapsule (op : BackendOp) : KernelCapsule :=
  nativeCapsule
    s!"native_cuda.{op.name}"
    op
    s!"Spec.{op.name}"
    s!"Native CUDA `{op.name}` follows the pointwise tensor contract."
    s!"Native CUDA `{op.name}` VJP is checked through runtime autograd tests."

def nativeReductionCapsule (op : BackendOp) : KernelCapsule :=
  { nativeCapsule
    s!"native_cuda.{op.name}"
    op
    s!"IR.{op.name} / Spec reduction contract"
    s!"Native CUDA `{op.name}` follows the explicit reduction shape contract."
    s!"Native CUDA `{op.name}` adjoint is checked through runtime gradient tests." with
    numericalPolicy.reduction := .implementationDefined }

/-- Native CUDA kernel with an accumulation whose tree/order is selected by the implementation.

This covers matrix products, affine layers, convolutions, losses, and average pooling. CUDA may use
parallel trees, fused multiply-add, cuBLAS/cuDNN algorithms, or architecture-specific schedules;
the capsule therefore records the reduction as implementation-defined instead of pretending it is
the reference left fold. -/
def nativeAccumulationCapsule (name : String) (op : BackendOp) (specName valueSummary vjpSummary :
    String) (vjpMode : VJPMode := .backendVJP) : KernelCapsule :=
  { nativeCapsule name op specName valueSummary vjpSummary vjpMode with
    numericalPolicy.reduction := .implementationDefined }

def nativeViewCapsule (op : BackendOp) : KernelCapsule :=
  nativeCapsule
    s!"native_cuda.{op.name}"
    op
    s!"IR.{op.name} shape/layout contract"
    s!"Native CUDA `{op.name}` follows the explicit shape/layout contract."
    s!"Native CUDA `{op.name}` adjoint is checked through runtime gradient tests."

def nativeForwardOnlyCapsule (op : BackendOp) (specName valueSummary : String) : KernelCapsule :=
  nativeCapsule
    s!"native_cuda.{op.name}"
    op
    specName
    valueSummary
    s!"Native CUDA `{op.name}` is a forward-only capsule with no registered VJP."
    .none

def nativeConvPoolCapsule (op : BackendOp) : KernelCapsule :=
  nativeCapsule
    s!"native_cuda.{op.name}"
    op
    s!"Spec.{op.name} / channel-first convolution-pooling contract"
    s!"Native CUDA `{op.name}` follows the channel-first runtime contract."
    s!"Native CUDA `{op.name}` VJP is checked by CUDA runtime coverage."

/-- Native window selection whose traversal and tie winner are chosen by the CUDA implementation. -/
def nativeSelectionCapsule (op : BackendOp) : KernelCapsule :=
  { nativeConvPoolCapsule op with
    numericalPolicy.reduction := .implementationDefined }

/-- Native CUDA batched/matrix multiplication, backed by CUDA/cuBLAS paths. -/
def matmul : KernelCapsule :=
  nativeAccumulationCapsule
    "native_cuda.matmul"
    .matmul
    "Spec.matmul / IR.matmul"
    "Matrix products agree with the row-major runtime contract."
    "Backward products are checked through autograd/runtime parity."

/-- Native CUDA ReLU activation. -/
def relu : KernelCapsule :=
  nativeCapsule
    "native_cuda.relu"
    .relu
    "Spec.relu / pointwise max(x, 0)"
    "ReLU forward follows the pointwise activation contract."
    "ReLU VJP is checked through runtime autograd tests."

/-- Native CUDA GELU activation. -/
def gelu : KernelCapsule :=
  nativeCapsule
    "native_cuda.gelu"
    .gelu
    "Spec.gelu / pointwise Gaussian error linear unit"
    "GELU forward follows the documented runtime approximation contract."
    "GELU VJP is checked through runtime autograd tests."

def add : KernelCapsule := nativePointwiseCapsule .add
def sub : KernelCapsule := nativePointwiseCapsule .sub
def mul : KernelCapsule := nativePointwiseCapsule .mul
def scale : KernelCapsule := nativePointwiseCapsule .scale
def abs : KernelCapsule := nativePointwiseCapsule .abs
def sqrt : KernelCapsule := nativePointwiseCapsule .sqrt
def clamp : KernelCapsule := nativePointwiseCapsule .clamp
def max : KernelCapsule := nativePointwiseCapsule .max
def min : KernelCapsule := nativePointwiseCapsule .min
def sigmoid : KernelCapsule := nativePointwiseCapsule .sigmoid
def tanh : KernelCapsule := nativePointwiseCapsule .tanh
def softplus : KernelCapsule := nativePointwiseCapsule .softplus
def exp : KernelCapsule := nativePointwiseCapsule .exp
def log : KernelCapsule := nativePointwiseCapsule .log
def inv : KernelCapsule := nativePointwiseCapsule .inv
def safeLog : KernelCapsule := nativePointwiseCapsule .safeLog
def logSoftmax : KernelCapsule :=
  nativeAccumulationCapsule
    "native_cuda.log_softmax"
    .logSoftmax
    "Spec.logSoftmax"
    "Log-softmax kernels follow the stable row/axis normalization contract."
    "Log-softmax VJPs are checked through runtime autograd tests."

/-- Native CUDA row/axis softmax kernels. -/
def softmax : KernelCapsule :=
  nativeAccumulationCapsule
    "native_cuda.softmax"
    .softmax
    "Spec.softmax"
    "Softmax kernels follow the row/axis normalization contract."
    "Softmax VJPs are checked through runtime autograd tests."

def reduceSum : KernelCapsule := nativeReductionCapsule .reduceSum
def reduceMean : KernelCapsule := nativeReductionCapsule .reduceMean

def reshape : KernelCapsule := nativeViewCapsule .reshape
def permute : KernelCapsule := nativeViewCapsule .permute
def broadcast : KernelCapsule := nativeViewCapsule .broadcast
def concat : KernelCapsule := nativeViewCapsule .concat
def slice : KernelCapsule := nativeViewCapsule .slice
def gather : KernelCapsule := nativeViewCapsule .gather
def scatterAdd : KernelCapsule := nativeViewCapsule .scatterAdd

def randUniform : KernelCapsule :=
  nativeForwardOnlyCapsule
    .randUniform
    "IR.randUniform"
    "Native CUDA deterministic random-uniform buffers follow the seeded runtime contract."

def bernoulliMask : KernelCapsule :=
  nativeForwardOnlyCapsule
    .bernoulliMask
    "IR.bernoulliMask"
    "Native CUDA deterministic Bernoulli masks follow the seeded runtime contract."

/-- Native CUDA layer normalization. -/
def layerNorm : KernelCapsule :=
  nativeAccumulationCapsule
    "native_cuda.layer_norm"
    .layerNorm
    "Spec.layerNorm"
    "LayerNorm follows the per-row normalization contract."
    "LayerNorm VJP is checked by CUDA runtime coverage."

/-- Native CUDA batch normalization. -/
def batchNorm : KernelCapsule :=
  nativeAccumulationCapsule
    "native_cuda.batch_norm"
    .batchNorm
    "Spec.batchNorm"
    "BatchNorm follows the channel-first normalization contract."
    "BatchNorm VJP is checked by CUDA runtime coverage."

/-- Native CUDA generic channel-first convolution. -/
def conv : KernelCapsule :=
  nativeAccumulationCapsule
    "native_cuda.conv"
    .conv
    "Spec.conv"
    "Convolution follows the generic channel-first runtime contract."
    "Convolution VJP is checked by CUDA runtime coverage."

/-- Native CUDA generic channel-first transpose convolution. -/
def convTranspose : KernelCapsule :=
  nativeAccumulationCapsule
    "native_cuda.conv_transpose"
    .convTranspose
    "Spec.convTranspose"
    "Transpose convolution follows the generic channel-first runtime contract."
    "Transpose-convolution VJP is checked by CUDA runtime coverage."

/-- Native CUDA max pooling. -/
def maxPool : KernelCapsule :=
  nativeSelectionCapsule .maxPool

/-- Native CUDA smooth max pooling. -/
def smoothMaxPool : KernelCapsule :=
  nativeAccumulationCapsule
    "native_cuda.smooth_max_pool"
    .smoothMaxPool
    "Spec.smoothMaxPool"
    "Smooth max pooling follows the generic window contract."
    "Smooth-max-pooling VJP is checked by CUDA runtime coverage."

/-- Native CUDA average pooling. -/
def avgPool : KernelCapsule :=
  nativeAccumulationCapsule
    "native_cuda.avg_pool"
    .avgPool
    "Spec.avgPool"
    "Average-pooling follows the channel-first window contract."
    "Average-pooling VJP is checked by CUDA runtime coverage."

/-- Native CUDA linear layer. -/
def linear : KernelCapsule :=
  nativeAccumulationCapsule
    "native_cuda.linear"
    .linear
    "Spec.linear"
    "Linear layer kernels follow the matvec/matmul plus bias contract."
    "Linear VJP is checked by CUDA runtime coverage."

/-- Native CUDA mean-squared-error loss. -/
def mseLoss : KernelCapsule :=
  nativeAccumulationCapsule
    "native_cuda.mse_loss"
    .mseLoss
    "Spec.mseLoss"
    "MSE loss follows the mean squared residual contract."
    "MSE VJP is checked by CUDA runtime coverage."

/-- Native CUDA FFT/FNO kernels. -/
def fftFno : KernelCapsule :=
  nativeAccumulationCapsule
    "native_cuda.fft_fno"
    .fftFno
    "packed rFFT and FNO spectral-convolution contracts"
    "Packed rFFT/irFFT and spectral convolution follow the documented half-spectrum contract."
    "Spectral-convolution VJPs are checked against finite differences."

/-- Native CUDA selective scan kernels. -/
def selectiveScan : KernelCapsule :=
  nativeAccumulationCapsule
    "native_cuda.selective_scan"
    .selectiveScan
    "diagonal selective-scan recurrence contract"
    "Selective scan forward follows the diagonal recurrence contract."
    "No generic VJP capsule is registered yet."
    .none

/-- Native CUDA capsules, excluding attention which has a dedicated semantic split. -/
def capsules : List KernelCapsule :=
  [ matmul
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
  , inv
  , safeLog
  , logSoftmax
  , softmax
  , reduceSum
  , reduceMean
  , reshape
  , permute
  , broadcast
  , concat
  , slice
  , gather
  , scatterAdd
  , randUniform
  , bernoulliMask
  , layerNorm
  , batchNorm
  , conv
  , convTranspose
  , maxPool
  , smoothMaxPool
  , avgPool
  , fftFno
  , selectiveScan
  ]

end NativeCUDA
end Backend
end NN
