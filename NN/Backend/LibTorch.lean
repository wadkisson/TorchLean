/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Attention

/-!
# LibTorch Backend Capsules

Optional LibTorch-backed capsules.

These capsules are intentionally separated from the default registry. Enabling them means accepting
an external implementation boundary. Its SDPA bridge uses TorchLean's hard boolean-mask
convention. Forward-only and external-autograd
paths are separate capsules so the planner can name the exact trust boundary.
-/

@[expose] public section

namespace NN
namespace Backend
namespace LibTorch

/-- Optional LibTorch capsules currently known to the planner. -/
def capsules : List KernelCapsule :=
  [Attention.libTorchSDPAForward, Attention.libTorchSDPAAutograd]

end LibTorch
end Backend
end NN
