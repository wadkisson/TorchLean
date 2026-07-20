/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Capsule

/-!
# Attention Backend Capsules

Attention is the first place where the backend-contract distinction matters in practice.

The proof-facing FlashAttention spec and every registered runtime provider use hard-mask semantics:
blocked entries have exactly zero softmax numerator. Additive attention biases are a separate
operation and are never used to encode a boolean mask.
-/

@[expose] public section

namespace NN
namespace Backend
namespace Attention

/-- Shared op tag for scaled dot-product attention capsules. -/
def scaledDotProductOp : BackendOp := .scaledDotProductAttention

/-- Composed TorchLean attention path: slower, but aligned with the hard-mask spec. -/
def torchLeanComposed : KernelCapsule :=
  { name := "torchlean.composed_attention"
    op := scaledDotProductOp
    provider := .torchLean
    device := .cuda
    trustLevel := .checked
    supportsForward := true
    vjpMode := .torchLeanTape
    shapeContract :=
      ContractDescriptor.guarded (.shapeSafety scaledDotProductOp)
        "Q/K/V have shape (heads, n, headDim); optional mask broadcasts to (heads, n, n)."
        "Runtime.Autograd.Cuda.requireValue plus checked UInt32 dimensions"
    layoutContract :=
      ContractDescriptor.guarded (.layoutCompatibility scaledDotProductOp .flatRowMajor)
        "Flat row-major buffers; head axis is the batch axis for bmm."
        "Buffer.swapAdjacentAtDepth and bmm shape checks"
    valueContract :=
      ContractDescriptor.tested
        (.valueRefinement scaledDotProductOp "Spec.scaledDotProductAttention")
        "Composed bmm, hard-masked row softmax, and bmm."
        "NN.Tests.Runtime.Cuda.Attention"
    vjpContract :=
      ContractDescriptor.tested
        (.vjpRefinement scaledDotProductOp "Spec.scaledDotProductAttentionBackward" .torchLeanTape)
        "TorchLean tape VJP through the composed attention expression."
        "Runtime autograd attention tests"
    numericalPolicy :=
      { rounding := .nearestEven
        subnormals := .implementationDefined
        contraction := .implementationDefined
        reduction := .implementationDefined }
    notes := "This is the proof-aligned fallback path: masked entries have zero softmax numerator." }

/-- Native fused attention path: faster than the composed fallback, still a CUDA runtime boundary. -/
def nativeFlashAttention : KernelCapsule :=
  { name := "native_cuda.flash_attention"
    op := scaledDotProductOp
    provider := .nativeCuda
    device := .cuda
    trustLevel := .checked
    supportsForward := true
    vjpMode := .backendVJP
    shapeContract :=
      ContractDescriptor.guarded (.shapeSafety scaledDotProductOp)
        "Q/K/V have shape (heads, n, headDim); optional mask broadcasts to (heads, n, n)."
        "torchlean_cuda_buffer_flash_attention_* size checks"
    layoutContract :=
      ContractDescriptor.guarded (.layoutCompatibility scaledDotProductOp .flatRowMajor)
        "Flat row-major buffers; head axis is the batch axis."
        "Cuda.Buffer shape/size FFI checks"
    valueContract :=
      ContractDescriptor.tested (.valueRefinement scaledDotProductOp "Spec.flashAttention")
        "Direct CUDA fused attention with hard-mask zero-numerator semantics."
        "NN.Tests.Runtime.Cuda.Attention"
    vjpContract :=
      ContractDescriptor.tested
        (.vjpRefinement scaledDotProductOp "Spec.flashAttentionBackward" .backendVJP)
        "CUDA VJP kernels return dQ, dK, and dV for the fused operator."
        "NN.Tests.Runtime.Cuda.Attention"
    numericalPolicy :=
      { rounding := .nearestEven
        subnormals := .implementationDefined
        contraction := .fused
        reduction := .implementationDefined }
    notes := "This is the default fused CUDA capsule until a planner explicitly selects LibTorch." }

/-- LibTorch SDPA forward provider while TorchLean keeps the graph/tape boundary. -/
def libTorchSDPAForward : KernelCapsule :=
  { name := "libtorch.sdpa_forward"
    op := scaledDotProductOp
    provider := .libTorch
    device := .cuda
    trustLevel := .trustedExternal
    supportsForward := true
    vjpMode := .torchLeanTape
    shapeContract :=
      ContractDescriptor.guarded (.shapeSafety scaledDotProductOp)
        "Q/K/V are checked as (heads, n, headDim); optional mask as (heads, n, n)."
        "torchlean_libtorch_sdpa_fwd size checks"
    layoutContract :=
      ContractDescriptor.guarded
        (.layoutCompatibility scaledDotProductOp .libTorchCudaView)
        "TorchLean CUDA buffers are wrapped by LibTorch from_blob and copied back contiguous."
        "contiguous CUDA tensor views"
        [.nativeSymbol
          { path := "csrc/cuda/kernels/torchlean_libtorch_sdpa.cpp"
            symbol := "torchlean_libtorch_sdpa_fwd"
            buildTarget? := some "torchlean_libtorch_sdpa_so" }]
    valueContract :=
      ContractDescriptor.trusted
        (.valueRefinement scaledDotProductOp "scaled_dot_product_attention with hard boolean masking")
        "LibTorch scaled_dot_product_attention forward."
        "LibTorch/CUDA SDPA implementation"
    vjpContract :=
      ContractDescriptor.guarded
        (.vjpRefinement scaledDotProductOp "TorchLean attention tape VJP" .torchLeanTape)
        "TorchLean records the node and keeps the backward boundary inside TorchLean."
        "backend profile requires vjpMode=torchLeanTape"
    numericalPolicy :=
      { rounding := .implementationDefined
        subnormals := .implementationDefined
        contraction := .implementationDefined
        reduction := .implementationDefined }
    notes :=
      "LibTorch provides the forward value; TorchLean records the node and evaluates its local VJP." }

/-- Built-in attention capsules in default planner order. Optional external providers register
their own capsules in their provider modules. -/
def capsules : List KernelCapsule :=
  [nativeFlashAttention, torchLeanComposed]

end Attention
end Backend
end NN
