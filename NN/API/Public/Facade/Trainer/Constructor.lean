/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.Trainer.Core

/-!
# TorchLean Trainer Construction

Public constructors for the unified trainer handle.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

universe u

/--
Values that can be used as public trainer models.

This class lets `Trainer.new` accept either a seedable model builder or an already-built checked
model:

```lean
Trainer.new modelBuilder ...
Trainer.new alreadyBuiltModel ...
```

The seed is consumed only by the builder case. Already-built models pass through unchanged.
-/
class ToModel (model : Type u) (σ τ : outParam Shape) where
  /-- Materialize the model, using the seed only when the value still needs initialization. -/
  build : Nat → model → nn.Sequential σ τ

instance {σ τ : Shape} : ToModel (nn.Sequential σ τ) σ τ where
  build _ model := model

instance {σ τ : Shape} : ToModel (nn.M (nn.Sequential σ τ)) σ τ where
  build seed model := nn.run seed model

/-- Build one unified public trainer from a sequential model or seedable model builder. -/
def new {model : Type u} {σ τ : Shape} [ToModel model σ τ] (m : model)
    (cfg : Config σ τ := {}) : Handle σ τ :=
  let built := ToModel.build cfg.seed m
  { model := built
    task := cfg.task
    runtime :=
      { optimizer := cfg.optimizer
        dtype := cfg.dtype
        backend := cfg.backend
        device := cfg.device
        fastKernels := cfg.fastKernels
        fastGpuMatmulPrecision := cfg.fastGpuMatmulPrecision }
    seed := cfg.seed }

end Trainer

/-- Public trainer type carrying a model, task, runtime configuration, and seed. -/
abbrev Trainer (σ τ : Shape) :=
  Trainer.Handle σ τ

end TorchLean
