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

/-- Build a CUDA attention capsule from the fields that actually vary across providers. -/
def mkCudaAttentionCapsule
    (name : String) (provider : Provider) (specName : String)
    (trustLevel : TrustLevel) (vjpMode : VJPMode)
    (shapeContract layoutContract valueContract vjpContract : ContractDescriptor)
    (notes : String) (runtimeSupport : RuntimeSupport := .eager) : KernelCapsule :=
  { name, op := scaledDotProductOp, provider, device := .cuda, specName, trustLevel
    supportsForward := true, vjpMode, runtimeSupport
    shapeContract, layoutContract, valueContract, vjpContract, notes }

/-- Composed TorchLean attention path: slower, but aligned with the hard-mask spec. -/
def torchLeanComposed : KernelCapsule :=
  mkCudaAttentionCapsule
    "torchlean.composed_attention" .torchLean
    "Spec.scaledDotProductAttention / Spec.flashAttention"
    .checked .torchLeanTape
    (ContractDescriptor.guarded (.shapeSafety scaledDotProductOp)
      "Q/K/V have shape (heads, n, headDim); optional mask broadcasts to (heads, n, n)."
      "Runtime.Autograd.Cuda.requireValue plus checked UInt32 dimensions")
    (ContractDescriptor.guarded (.layoutCompatibility scaledDotProductOp .flatRowMajor)
      "Flat row-major buffers; head axis is the batch axis for bmm."
      "Buffer.swapAdjacentAtDepth and bmm shape checks")
    (ContractDescriptor.tested
      (.valueRefinement scaledDotProductOp "Spec.scaledDotProductAttention")
      "Composed bmm, hard-masked row softmax, and bmm."
      "NN.Tests.Runtime.Cuda.Attention")
    (ContractDescriptor.tested
      (.vjpRefinement scaledDotProductOp "Spec.scaledDotProductAttentionBackward" .torchLeanTape)
      "TorchLean tape VJP through the composed attention expression."
      "Runtime autograd attention tests")
    "This is the proof-aligned fallback path: masked entries have zero softmax numerator."

/-- Native fused attention path: faster than the composed fallback, still a CUDA runtime boundary. -/
def nativeFlashAttention : KernelCapsule :=
  mkCudaAttentionCapsule
    "native_cuda.flash_attention" .nativeCuda "Spec.flashAttention"
    .checked .backendVJP
    (ContractDescriptor.guarded (.shapeSafety scaledDotProductOp)
      "Q/K/V have shape (heads, n, headDim); optional mask broadcasts to (heads, n, n)."
      "torchlean_cuda_buffer_flash_attention_* size checks")
    (ContractDescriptor.guarded (.layoutCompatibility scaledDotProductOp .flatRowMajor)
      "Flat row-major buffers; head axis is the batch axis."
      "Cuda.Buffer shape/size FFI checks")
    (ContractDescriptor.tested (.valueRefinement scaledDotProductOp "Spec.flashAttention")
      "Direct CUDA fused attention with hard-mask zero-numerator semantics."
      "NN.Tests.Runtime.Cuda.Attention")
    (ContractDescriptor.tested
      (.vjpRefinement scaledDotProductOp "Spec.flashAttentionBackward" .backendVJP)
      "CUDA VJP kernels return dQ, dK, and dV for the fused operator."
      "NN.Tests.Runtime.Cuda.Attention")
    "Default fused CUDA capsule when LibTorch is not enabled."

/-- LibTorch SDPA forward provider while TorchLean keeps the graph/tape boundary. -/
def libTorchSDPAForward : KernelCapsule :=
  mkCudaAttentionCapsule
    "libtorch.sdpa_forward" .libTorch
    "scaled_dot_product_attention with hard boolean masking"
    .trustedExternal .torchLeanTape
    (ContractDescriptor.guarded (.shapeSafety scaledDotProductOp)
      "Q/K/V are checked as (heads, n, headDim); optional mask as (heads, n, n)."
      "torchlean_libtorch_sdpa_fwd size checks")
    (ContractDescriptor.guarded (.layoutCompatibility scaledDotProductOp .libTorchCudaView)
      "TorchLean CUDA buffers are wrapped by LibTorch from_blob and copied back contiguous."
      "contiguous CUDA tensor views"
      [.nativeSymbol
        { path := "csrc/cuda/kernels/torchlean_libtorch_sdpa.cpp"
          symbol := "torchlean_libtorch_sdpa_fwd"
          buildTarget? := some "torchlean_libtorch_sdpa_so" }])
    (ContractDescriptor.trusted
      (.valueRefinement scaledDotProductOp "scaled_dot_product_attention with hard boolean masking")
      "LibTorch scaled_dot_product_attention forward."
      "LibTorch/CUDA SDPA implementation")
    (ContractDescriptor.guarded
      (.vjpRefinement scaledDotProductOp "TorchLean attention tape VJP" .torchLeanTape)
      "TorchLean records the node and keeps the backward boundary inside TorchLean."
      "backend profile requires vjpMode=torchLeanTape")
    "LibTorch forward + TorchLean composed attention VJP on the CUDA tape."

/-- LibTorch SDPA provider where LibTorch also owns the local autograd VJP. -/
def libTorchSDPAAutograd : KernelCapsule :=
  mkCudaAttentionCapsule
    "libtorch.sdpa_autograd" .libTorch
    "scaled_dot_product_attention forward and VJP with hard boolean masking"
    .trustedExternal .externalAutograd
    (ContractDescriptor.guarded (.shapeSafety scaledDotProductOp)
      "Q/K/V and dOut are checked as (heads, n, headDim); optional mask as (heads, n, n)."
      "torchlean_libtorch_sdpa_bwd size checks")
    (ContractDescriptor.guarded (.layoutCompatibility scaledDotProductOp .libTorchCudaView)
      "TorchLean CUDA buffers are wrapped by LibTorch from_blob and copied back contiguous."
      "contiguous CUDA tensor views"
      [.nativeSymbol
        { path := "csrc/cuda/kernels/torchlean_libtorch_sdpa.cpp"
          symbol := "torchlean_libtorch_sdpa_bwd"
          buildTarget? := some "torchlean_libtorch_sdpa_so" }])
    (ContractDescriptor.trusted
      (.valueRefinement scaledDotProductOp "scaled_dot_product_attention with hard boolean masking")
      "LibTorch scaled_dot_product_attention forward."
      "LibTorch/CUDA SDPA implementation")
    (ContractDescriptor.trusted
      (.vjpRefinement scaledDotProductOp "LibTorch scaled_dot_product_attention VJP" .externalAutograd)
      "LibTorch autograd VJP for scaled_dot_product_attention."
      "LibTorch autograd for SDPA")
    "Preferred training path when LibTorch is linked: PyTorch SDPA owns forward and local VJP."

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

/-- Runtime implementation selector for CUDA attention. -/
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
