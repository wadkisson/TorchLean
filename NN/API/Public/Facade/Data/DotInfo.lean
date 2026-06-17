/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.NN

/-!
# TorchLean Public Dot-Notation Model Info

Dot-notation model summaries for built models and seeded model builders.
-/

@[expose] public section

namespace Runtime.Autograd.TorchLean.NN.Seq

/-- Dot-notation model info for built TorchLean sequential models. -/
abbrev info {σ τ : TorchLean.Shape} (model : Seq σ τ) : String :=
  TorchLean.nn.info model

end Runtime.Autograd.TorchLean.NN.Seq

namespace NN.API.rand.SeedM

/-!
Seeded builders are inspected with seed `0`. The seed only chooses initial parameter values; layer
names, shapes, and parameter counts come from the model structure.
-/

/-- Dot-notation model info for public seeded model builders. -/
abbrev info {σ τ : TorchLean.Shape}
    (model : SeedM (Runtime.Autograd.TorchLean.NN.Seq σ τ)) : String :=
  TorchLean.nn.info (NN.API.nn.run 0 model)

end NN.API.rand.SeedM
