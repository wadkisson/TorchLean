/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.List.Basic
public import NN.Proofs.RuntimeApprox.NF.Ops

/-!
# NF Proof Utilities

Small proof utilities shared across NF backend approximation modules.

We keep these helpers in a dedicated file so we don’t re-prove the same list-fold facts in every
large operator proof (Conv2D, linalg, etc.).
-/

@[expose] public section


namespace Proofs
namespace RuntimeApprox

namespace NFBackend

open Spec
open Tensor
open NN.MLTheory.Robustness.Spec

open TorchLean.Floats

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]
variable {rnd : ℝ → ℤ} [NeuralValidRndToNearest rnd]

local notation "R" => TorchLean.Floats.NF β fexp rnd

/-!
## List folds

`foldl_congr` is the workhorse for rewriting a `foldl` step function when the new step is
pointwise equal to the old one.
-/

lemma foldl_congr {α β : Type} (l : List β) (f g : α → β → α) (init : α)
    (h : ∀ a b, f a b = g a b) :
    l.foldl f init = l.foldl g init := by
  induction l generalizing init with
  | nil => rfl
  | cons b tl ih =>
      simp [List.foldl, h, ih]

/--
`foldl` over `flatMap` is the same as the corresponding nested `foldl`.

Convolution proofs use this to align flat index enumerations with nested channel/spatial loops, but
the statement is list-generic and belongs with the other NF fold utilities.
-/
lemma foldl_flatMap {α β γ : Type} (l : List α) (g : α → List β) (f : γ → β → γ) (init : γ) :
    (l.flatMap g).foldl f init = l.foldl (fun acc a => (g a).foldl f acc) init := by
  induction l generalizing init with
  | nil =>
      simp
  | cons a tl ih =>
      simp [List.flatMap_cons, List.foldl_append, ih]

end NFBackend

end RuntimeApprox
end Proofs
