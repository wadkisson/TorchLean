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
    specName := "Spec.scaledDotProductAttention / Spec.flashAttention"
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
    notes := "This is the proof-aligned fallback path: masked entries have zero softmax numerator." }

/-- Native fused attention path: faster than the composed fallback, still a CUDA runtime boundary. -/
def nativeFlashAttention : KernelCapsule :=
  { name := "native_cuda.flash_attention"
    op := scaledDotProductOp
    provider := .nativeCuda
    device := .cuda
    specName := "Spec.flashAttention"
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
    notes := "Default fused CUDA capsule when LibTorch is not enabled." }

/-- LibTorch SDPA forward provider while TorchLean keeps the graph/tape boundary. -/
def libTorchSDPAForward : KernelCapsule :=
  { name := "libtorch.sdpa_forward"
    op := scaledDotProductOp
    provider := .libTorch
    device := .cuda
    specName := "scaled_dot_product_attention with hard boolean masking"
    trustLevel := .trustedExternal
    supportsForward := true
    vjpMode := .torchLeanTape
    runtimeSupport := .eager
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
    notes :=
      "LibTorch forward + TorchLean composed attention VJP on the CUDA tape." }

/-- LibTorch SDPA provider where LibTorch also owns the local autograd VJP. -/
def libTorchSDPAAutograd : KernelCapsule :=
  { name := "libtorch.sdpa_autograd"
    op := scaledDotProductOp
    provider := .libTorch
    device := .cuda
    specName := "scaled_dot_product_attention forward and VJP with hard boolean masking"
    trustLevel := .trustedExternal
    supportsForward := true
    vjpMode := .externalAutograd
    runtimeSupport := .eager
    shapeContract :=
      ContractDescriptor.guarded (.shapeSafety scaledDotProductOp)
        "Q/K/V and dOut are checked as (heads, n, headDim); optional mask as (heads, n, n)."
        "torchlean_libtorch_sdpa_bwd size checks"
    layoutContract :=
      ContractDescriptor.guarded
        (.layoutCompatibility scaledDotProductOp .libTorchCudaView)
        "TorchLean CUDA buffers are wrapped by LibTorch from_blob and copied back contiguous."
        "contiguous CUDA tensor views"
        [.nativeSymbol
          { path := "csrc/cuda/kernels/torchlean_libtorch_sdpa.cpp"
            symbol := "torchlean_libtorch_sdpa_bwd"
            buildTarget? := some "torchlean_libtorch_sdpa_so" }]
    valueContract :=
      ContractDescriptor.trusted
        (.valueRefinement scaledDotProductOp "scaled_dot_product_attention with hard boolean masking")
        "LibTorch scaled_dot_product_attention forward."
        "LibTorch/CUDA SDPA implementation"
    vjpContract :=
      ContractDescriptor.trusted
        (.vjpRefinement scaledDotProductOp "LibTorch scaled_dot_product_attention VJP" .externalAutograd)
        "LibTorch autograd VJP for scaled_dot_product_attention."
        "LibTorch autograd for SDPA"
    notes :=
      "Preferred training path when LibTorch is linked: PyTorch SDPA owns forward and local VJP." }

/-- Candidate capsules in default CUDA planner order (no LibTorch). -/
def cudaCandidates : List KernelCapsule :=
  [nativeFlashAttention, torchLeanComposed]

/-- Candidate capsules when the optional LibTorch backend is enabled by policy/config. -/
def cudaCandidatesWithLibTorch : List KernelCapsule :=
  [libTorchSDPAAutograd, libTorchSDPAForward, nativeFlashAttention, torchLeanComposed]

/-- Concrete CUDA attention implementation selected from a capsule. -/
inductive CudaAttentionImpl where
  /-- Native fused FlashAttention kernels. -/
  | nativeFlash
  /-- LibTorch SDPA forward + LibTorch autograd VJP. -/
  | libTorchAutograd
  /-- LibTorch SDPA forward + TorchLean composed VJP. -/
  | libTorchForward
  /-- Fully composed TorchLean `bmm -> softmax -> bmm` path. -/
  | composed
deriving DecidableEq, Repr

/--
Runtime implementation selector for CUDA attention.
-/
def cudaAttentionImpl (c : KernelCapsule) : Except String CudaAttentionImpl :=
  if c.op != scaledDotProductOp then
    .error s!"backend capsule {c.name} does not implement {scaledDotProductOp.name}"
  else
    match c.provider with
    | .nativeCuda => .ok .nativeFlash
    | .torchLean | .reference => .ok .composed
    | .libTorch =>
        match c.vjpMode with
        | .externalAutograd => .ok .libTorchAutograd
        | .torchLeanTape | .backendVJP | .none => .ok .libTorchForward
    | p =>
        .error s!"backend provider {reprStr p} is not wired for CUDA attention execution"

end Attention
end Backend
end NN
