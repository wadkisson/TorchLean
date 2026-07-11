/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Planner
public import NN.Backend.Target
public import NN.Backend.Attention
public import NN.Backend.NativeCUDA
public import NN.Backend.Reference
public import NN.Backend.LibTorch

/-!
# Backend Registry

Registry of backend capsules known to TorchLean's planner.

The default registry keeps TorchLean/native providers ahead of external providers. Optional external
registries, such as LibTorch, are appended only when the caller explicitly enables that backend and
chooses a trust policy that admits it.
-/

@[expose] public section

namespace NN
namespace Backend
namespace Registry

/-- Default capsules: cross-platform reference paths plus native CUDA paths. -/
def defaultCapsules : List KernelCapsule :=
  Attention.cudaCandidates ++ NativeCUDA.capsules ++ Reference.capsules

/-- Default capsules plus optional LibTorch providers. -/
def withLibTorchCapsules : List KernelCapsule :=
  Attention.cudaCandidatesWithLibTorch ++ NativeCUDA.capsules ++ Reference.capsules

/-- Select the registry used by an execution configuration. -/
def capsules (enableLibTorch : Bool := false) : List KernelCapsule :=
  if enableLibTorch then withLibTorchCapsules else defaultCapsules

end Registry
end Backend
end NN
