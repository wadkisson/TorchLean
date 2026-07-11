/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# Backend Types

Small vocabulary for backend selection and trust boundaries.

TorchLean owns the spec, graph, and proof-facing contracts. Backends are execution providers for
parts of that graph: a Lean reference path, the TorchLean runtime, native CUDA kernels, LibTorch,
ATen, cuBLAS/cuDNN/cuFFT, TPU/XLA, AWS Neuron/Trainium, or future platform-specific providers. This
file deliberately contains only data. It should stay cheap to import from specs, runtime wrappers,
docs generators, and tests.
-/

@[expose] public section

namespace NN
namespace Backend

/-- Hardware or execution target visible to the planner. -/
inductive Device where
  | cpu
  | cuda
  | rocm
  | metal
  | wasm
  | tpu
  | trainium
  | custom
  | external
  deriving DecidableEq, Repr

namespace Device

/-- Stable spelling used in profile names, reports, and CLI bridges. -/
def cliName : Device → String
  | .cpu => "cpu"
  | .cuda => "cuda"
  | .rocm => "rocm"
  | .metal => "metal"
  | .wasm => "wasm"
  | .tpu => "tpu"
  | .trainium => "trainium"
  | .custom => "custom"
  | .external => "external"

end Device

/-- Concrete provider family used to execute a kernel capsule. -/
inductive Provider where
  | reference
  | torchLean
  | nativeCuda
  | libTorch
  | aten
  | mps
  | webGpu
  | cuBLAS
  | cuDNN
  | cuFFT
  | xla
  | neuron
  | customChip
  | external
  deriving DecidableEq, Repr

/-- Stable operation vocabulary used by backend capsules, graph planning, and runtime guards.

This is deliberately a closed vocabulary. New backend-visible operations should be added here and
then wired through the IR adapter and capsule registry. Runtime tape/debug labels may still be
strings, but the backend planner should not accept arbitrary stringly-typed operation names.
-/
inductive BackendOp where
  | randUniform
  | bernoulliMask
  | add
  | sub
  | mul
  | scale
  | abs
  | sqrt
  | clamp
  | max
  | min
  | relu
  | gelu
  | sigmoid
  | tanh
  | softplus
  | exp
  | log
  | sin
  | cos
  | inv
  | safeLog
  | logSoftmax
  | softmax
  | sum
  | reduceSum
  | reduceMean
  | flatten
  | reshape
  | permute
  | transpose2d
  | swapAdjacentAtDepth
  | transpose3dFirstToLast
  | transpose3dLastToFirst
  | transpose3dLastTwo
  | broadcastTo
  | concatVectors
  | concatLeadingAxis
  | sliceLeadingAxisRange
  | gatherScalar
  | gatherScalarNat
  | gatherRow
  | gatherVecNat
  | gatherRowsNat
  | scatterAddVec
  | scatterAddRow
  | matmul
  | bmm
  | linear
  | mseLoss
  | layerNorm
  | batchNorm
  | batchNormChannelFirst
  | conv
  | conv2d
  | convTranspose
  | convTranspose2d
  | maxPool
  | maxPool2d
  | maxPool2dPad
  | smoothMaxPool
  | smoothMaxPool2d
  | avgPool
  | avgPool2d
  | avgPool2dPad
  | fftFno
  | selectiveScan
  | scaledDotProductAttention
  deriving DecidableEq, BEq, Repr

namespace BackendOp

/-- Stable spelling used in reports, capsule names, and CLI diagnostics. -/
def name : BackendOp → String
  | .randUniform => "rand_uniform"
  | .bernoulliMask => "bernoulli_mask"
  | .add => "add"
  | .sub => "sub"
  | .mul => "mul"
  | .scale => "scale"
  | .abs => "abs"
  | .sqrt => "sqrt"
  | .clamp => "clamp"
  | .max => "max"
  | .min => "min"
  | .relu => "relu"
  | .gelu => "gelu"
  | .sigmoid => "sigmoid"
  | .tanh => "tanh"
  | .softplus => "softplus"
  | .exp => "exp"
  | .log => "log"
  | .sin => "sin"
  | .cos => "cos"
  | .inv => "inv"
  | .safeLog => "safe_log"
  | .logSoftmax => "log_softmax"
  | .softmax => "softmax"
  | .sum => "sum"
  | .reduceSum => "reduce_sum"
  | .reduceMean => "reduce_mean"
  | .flatten => "flatten"
  | .reshape => "reshape"
  | .permute => "permute"
  | .transpose2d => "transpose2d"
  | .swapAdjacentAtDepth => "swapAdjacentAtDepth"
  | .transpose3dFirstToLast => "transpose3d_first_to_last"
  | .transpose3dLastToFirst => "transpose3d_last_to_first"
  | .transpose3dLastTwo => "transpose3d_last_two"
  | .broadcastTo => "broadcastTo"
  | .concatVectors => "concat_vectors"
  | .concatLeadingAxis => "concat_leading_axis"
  | .sliceLeadingAxisRange => "slice_leading_axis_range"
  | .gatherScalar => "gather_scalar"
  | .gatherScalarNat => "gather_scalar_nat"
  | .gatherRow => "gather_row"
  | .gatherVecNat => "gather_vec_nat"
  | .gatherRowsNat => "gather_rows_nat"
  | .scatterAddVec => "scatter_add_vec"
  | .scatterAddRow => "scatter_add_row"
  | .matmul => "matmul"
  | .bmm => "bmm"
  | .linear => "linear"
  | .mseLoss => "mse_loss"
  | .layerNorm => "layer_norm"
  | .batchNorm => "batch_norm"
  | .batchNormChannelFirst => "batchnorm_channel_first"
  | .conv => "conv"
  | .conv2d => "conv2d"
  | .convTranspose => "conv_transpose"
  | .convTranspose2d => "conv_transpose2d"
  | .maxPool => "max_pool"
  | .maxPool2d => "max_pool2d"
  | .maxPool2dPad => "max_pool2d_pad"
  | .smoothMaxPool => "smooth_max_pool"
  | .smoothMaxPool2d => "smooth_max_pool2d"
  | .avgPool => "avg_pool"
  | .avgPool2d => "avg_pool2d"
  | .avgPool2dPad => "avg_pool2d_pad"
  | .fftFno => "fft_fno"
  | .selectiveScan => "selective_scan"
  | .scaledDotProductAttention => "scaled_dot_product_attention"

instance : ToString BackendOp where
  toString op := op.name

/-- Whether training through this operation requires a registered local VJP.

Random sources create values but are not themselves differentiated. Every other backend-visible
operation must provide a compatible VJP whenever gradient tracking is requested.
-/
def requiresVJP : BackendOp → Bool
  | .randUniform | .bernoulliMask => false
  | _ => true

end BackendOp

/--
How much TorchLean knows about an implementation.

`trustedExternal` is allowed, but it is intentionally loud: the contract names the boundary instead
of silently treating an industrial kernel as though Lean had verified its source.
-/
inductive TrustLevel where
  | verified
  | checked
  | fuzzed
  | trustedExternal
  deriving DecidableEq, Repr

/-- User- or CI-selected policy for which backend capsules may be used. -/
inductive TrustPolicy where
  | verifiedOnly
  | checked
  | fuzzedOk
  | allowTrustedExternal
  deriving DecidableEq, Repr

namespace TrustPolicy

/-- Whether a policy admits a capsule at the given trust level. -/
def accepts : TrustPolicy -> TrustLevel -> Bool
  | .verifiedOnly, .verified => true
  | .verifiedOnly, _ => false
  | .checked, .verified => true
  | .checked, .checked => true
  | .checked, _ => false
  | .fuzzedOk, .trustedExternal => false
  | .fuzzedOk, _ => true
  | .allowTrustedExternal, _ => true

end TrustPolicy

/-- How a backend capsule treats gradients. -/
inductive VJPMode where
  | none
  | torchLeanTape
  | backendVJP
  | externalAutograd
  deriving DecidableEq, Repr

/-- Backend preference requested by a runtime configuration. -/
inductive BackendPreference where
  | auto
  | prefer (provider : Provider)
  | only (provider : Provider)
  deriving DecidableEq, Repr

/-- Runtime-level backend configuration. Public APIs can wrap this with friendlier defaults. -/
structure ExecutionConfig where
  device : Device := .cpu
  backend : BackendPreference := .auto
  trustPolicy : TrustPolicy := .checked
  vjpMode : VJPMode := .torchLeanTape
  deriving Repr

end Backend
end NN
