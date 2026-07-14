/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Session

/-!
# Eager Tensor Operations

PyTorch-style tensor operations backed by the eager CPU/CUDA tapes. These wrappers record runtime
nodes, dispatch CUDA kernels when requested, and preserve the typed `TensorRef` surface.
-/


@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec
open Tensor
open Proofs.Autograd.Algebra

namespace Internal

namespace EagerSession

/-!
## Tensor ops (eager tape wrappers)

The following definitions are the eager front-end for `Runtime.Autograd.Tape.*` primitives. Each one:
- reads the current tape from `s.tape`,
- appends a new node/leaf via a `Tape.*` constructor,
- writes the updated tape back, and
- returns a fresh `TensorRef` pointing to the new node id.

PyTorch comparison: this is the standard eager autograd mechanism (a dynamic tape of ops).
-/

/--
Dispatch an eager op with optional CUDA support.

When `Options.device = .cuda`, any op whose CUDA implementation returns `none` will throw.

TorchLean's CUDA eager mode has no per-op CPU fallback: either the op is supported by CUDA, or it
errors immediately.
-/
def dispatchCudaOpt {α β : Type} (s : EagerSession α) (op : NN.Backend.BackendOp)
    (cpu : IO β) (cuda : IO (Option β)) : IO β := do
  let _ ← s.selectedCapsule op
  if Options.device s.opts == .cuda then
    let r? ← cuda
    match r? with
    | some r => pure r
    | none =>
        throw <| IO.userError s!"torch: cuda: `{op.name}` is unsupported by the eager CUDA backend"
  else
    cpu

/--
Require that the selected backend contract is the native CUDA capsule used by this eager op.

This guard is intentionally runtime-side. The planner may know about LibTorch, Metal, ROCm, or
reference capsules, but these eager branches below call TorchLean's native CUDA tape directly. If a
profile selects another provider, failing here is better than silently running a different backend.
-/
def requireNativeCudaCapsule {α : Type} (s : EagerSession α) (op : NN.Backend.BackendOp) : IO Unit := do
  let _ ← s.selectedCapsule op
  unless s.opts.usesCuda do
    throw <| IO.userError s!"torch: native CUDA capsule requested for CPU op `{op.name}`"

/--
Validate one float-encoded token id and return the corresponding `Nat`.

This is intentionally stricter than `Float.floor`: language-model targets are discrete labels, so a
fractional value is almost certainly a bad dataset or adapter boundary. Rejecting it here prevents a
quiet change of class label before the embedding or cross-entropy code sees the id.
-/
def natOfTokenFloat (i : Nat) (x : Float) : IO Nat := do
  if x.isNaN || x.isInf then
    throw <| IO.userError s!"torch: token id at index {i} is not finite"
  else if x < 0.0 then
    throw <| IO.userError s!"torch: token id at index {i} is negative: {x}"
  else
    let y := Float.floor x
    if y != x then
      throw <| IO.userError s!"torch: token id at index {i} is not an integer: {x}"
    else
      let n := y.toUInt64.toNat
      if Float.ofNat n == x then
        pure n
      else
        throw <| IO.userError s!"torch: token id at index {i} is outside the supported Nat range: {x}"

end EagerSession

end Internal
end Torch
end Autograd
end Runtime
