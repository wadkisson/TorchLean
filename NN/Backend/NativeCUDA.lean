/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Capsule
public import NN.Backend.OpCatalog

/-!
# Native CUDA Backend Capsules

Capsule metadata for TorchLean's native CUDA runtime provider.

These capsules describe kernels that currently live under `csrc/cuda/**` and are exposed
to Lean through `NN.Runtime.Autograd.Engine.Cuda.*`. The C/CUDA source still owns the implementation;
this module gives the planner a typed, inspectable contract layer for those implementation choices.

Catalog ops (`OpCatalog.pointwiseOps` / `reductionOps` / `viewOps` / `convPoolOps`) are stamped
through the builders below. Specials with custom summaries stay explicit.
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
    specName
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
    notes := "Native CUDA code is an FFI boundary; the capsule records the contract TorchLean checks." }

def nativePointwiseCapsule (op : BackendOp) : KernelCapsule :=
  nativeCapsule
    s!"native_cuda.{op.name}"
    op
    s!"Spec.{op.name}"
    s!"Native CUDA `{op.name}` follows the pointwise tensor contract."
    s!"Native CUDA `{op.name}` VJP is checked through runtime autograd tests."

def nativeReductionCapsule (op : BackendOp) : KernelCapsule :=
  nativeCapsule
    s!"native_cuda.{op.name}"
    op
    s!"IR.{op.name} / Spec reduction contract"
    s!"Native CUDA `{op.name}` follows the explicit reduction shape contract."
    s!"Native CUDA `{op.name}` adjoint is checked through runtime gradient tests."

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

/-- Native CUDA batched/matrix multiplication, backed by CUDA/cuBLAS paths. -/
def matmul : KernelCapsule :=
  nativeCapsule
    "native_cuda.matmul" .matmul "Spec.matmul / IR.matmul"
    "Matrix products agree with the row-major runtime contract."
    "Backward products are checked through autograd/runtime parity."

def bmm : KernelCapsule :=
  nativeCapsule
    "native_cuda.bmm" .bmm "Spec.bmm / batched matrix product"
    "Batched matrix products agree with the row-major runtime contract."
    "Batched matrix-product VJPs are checked through autograd/runtime parity."

def relu : KernelCapsule :=
  nativeCapsule
    "native_cuda.relu" .relu "Spec.relu / pointwise max(x, 0)"
    "ReLU forward follows the pointwise activation contract."
    "ReLU VJP is checked through runtime autograd tests."

def gelu : KernelCapsule :=
  nativeCapsule
    "native_cuda.gelu" .gelu "Spec.gelu / pointwise Gaussian error linear unit"
    "GELU forward follows the documented runtime approximation contract."
    "GELU VJP is checked through runtime autograd tests."

def softmax : KernelCapsule :=
  nativeCapsule
    "native_cuda.softmax" .softmax "Spec.softmax"
    "Softmax kernels follow the row/axis normalization contract."
    "Softmax VJPs are checked through runtime autograd tests."

def randUniform : KernelCapsule :=
  nativeForwardOnlyCapsule .randUniform "IR.randUniform"
    "Native CUDA deterministic random-uniform buffers follow the seeded runtime contract."

def bernoulliMask : KernelCapsule :=
  nativeForwardOnlyCapsule .bernoulliMask "IR.bernoulliMask"
    "Native CUDA deterministic Bernoulli masks follow the seeded runtime contract."

def layerNorm : KernelCapsule :=
  nativeCapsule
    "native_cuda.layer_norm" .layerNorm "Spec.layerNorm"
    "LayerNorm follows the per-row normalization contract."
    "LayerNorm VJP is checked by CUDA runtime coverage."

def batchNorm : KernelCapsule :=
  nativeCapsule
    "native_cuda.batch_norm" .batchNorm "Spec.batchNorm"
    "BatchNorm follows the channel-first normalization contract."
    "BatchNorm VJP is checked by CUDA runtime coverage."

def batchNormChannelFirst : KernelCapsule :=
  nativeCapsule
    "native_cuda.batchnorm_channel_first" .batchNormChannelFirst
    "Spec.batchNorm / channel-first runtime contract"
    "Channel-first BatchNorm follows the image/channel normalization contract."
    "Channel-first BatchNorm VJP is checked by CUDA runtime coverage."

def conv2d : KernelCapsule :=
  nativeCapsule
    "native_cuda.conv2d" .conv2d "Spec.conv2d"
    "Conv2D follows the NCHW/channel-first runtime contract."
    "Conv2D VJP is checked by CUDA runtime coverage."

def maxPool : KernelCapsule :=
  nativeCapsule
    "native_cuda.max_pool" .maxPool "Spec.maxPool"
    "Max-pooling follows the channel-first window contract."
    "Max-pooling VJP is checked by CUDA runtime coverage."

def avgPool : KernelCapsule :=
  nativeCapsule
    "native_cuda.avg_pool" .avgPool "Spec.avgPool"
    "Average-pooling follows the channel-first window contract."
    "Average-pooling VJP is checked by CUDA runtime coverage."

def linear : KernelCapsule :=
  nativeCapsule
    "native_cuda.linear" .linear "Spec.linear"
    "Linear layer kernels follow the matvec/matmul plus bias contract."
    "Linear VJP is checked by CUDA runtime coverage."

def mseLoss : KernelCapsule :=
  nativeCapsule
    "native_cuda.mse_loss" .mseLoss "Spec.mseLoss"
    "MSE loss follows the mean squared residual contract."
    "MSE VJP is checked by CUDA runtime coverage."

def fftFno : KernelCapsule :=
  nativeCapsule
    "native_cuda.fft_fno" .fftFno "packed rFFT and FNO spectral-convolution contracts"
    "Packed rFFT/irFFT and spectral convolution follow the documented half-spectrum contract."
    "Spectral-convolution VJPs are checked against finite differences."

def selectiveScan : KernelCapsule :=
  nativeCapsule
    "native_cuda.selective_scan" .selectiveScan "diagonal selective-scan recurrence contract"
    "Selective scan forward follows the diagonal recurrence contract."
    "No generic VJP capsule is registered yet."
    .none

/-- Native CUDA capsules, excluding attention which has a dedicated semantic split. -/
def capsules : List KernelCapsule :=
  [ matmul, bmm, linear, mseLoss, relu, gelu, softmax
  , randUniform, bernoulliMask
  , layerNorm, batchNorm, batchNormChannelFirst
  , conv2d, maxPool, avgPool, fftFno, selectiveScan ]
  ++ OpCatalog.pointwiseOps.map nativePointwiseCapsule
  ++ OpCatalog.reductionOps.map nativeReductionCapsule
  ++ OpCatalog.viewOps.map nativeViewCapsule
  ++ OpCatalog.convPoolOps.map nativeConvPoolCapsule

end NativeCUDA
end Backend
end NN
