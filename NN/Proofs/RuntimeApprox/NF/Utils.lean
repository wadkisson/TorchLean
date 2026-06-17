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

/-!
## Generic facts about `approxT`

These lemmas are backend-independent but heavily used in NF backend developments.
-/

omit [NeuralValidExp fexp] [NeuralValidRndToNearest rnd] in
lemma approxT_eps_nonneg {s : Shape} {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ}
    (hx : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps) :
    0 ≤ eps := by
  have h0 :
      0 ≤ tensorDistance (α := SpecScalar) linfNorm xS
        (tensorToSpec (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR) := by
    -- `tensor_distance` is `linf_norm` of a difference, hence nonnegative.
    simpa [NN.MLTheory.Robustness.Spec.tensorDistance, linfNorm] using
      (linf_norm_nonneg
        (t :=
          NN.MLTheory.Robustness.Spec.tensorDistance.tensor_sub (α := SpecScalar) (s := s) xS
            (tensorToSpec (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)))
  exact le_trans h0 hx

omit [NeuralValidExp fexp] [NeuralValidRndToNearest rnd] in
lemma approxT_dim_of_forall {n : Nat} {s : Shape}
    {xS : SpecTensor (.dim n s)} {xR : Tensor R (.dim n s)} {eps : ℝ}
    (hε : 0 ≤ eps)
    (h : ∀ i : Fin n,
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (match xS with | .dim f => f i) (match xR with | .dim f => f i) eps) :
    approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps := by
  classical
  cases xS with
  | dim xSf =>
    cases xR with
    | dim xRf =>
      have hf :
          ∀ i ∈ List.finRange n,
            tensorDistance (α := SpecScalar) linfNorm (xSf i)
                (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                  (xRf i))
              ≤ eps := by
        intro i _hi
        have hfi := h i
        simpa [approxT, approxWith] using hfi
      have hfold :=
        List.foldl_max_le_of_le (List.finRange n)
          (fun i =>
            tensorDistance (α := SpecScalar) linfNorm (xSf i)
              (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) (xRf
                i)))
          (acc := (0 : ℝ)) (eps := eps) hε hf
      simp [approxT, approxWith, tensorDistance,
        linfNorm, RuntimeApprox.linfNorm, tensorToSpec, Spec.mapTensor]
      change
        List.foldl
          (fun a i =>
            max a
              (tensorLinfNorm
                ((xSf i).subSpec
                  (mapTensor (toSpec (β := β) (fexp := fexp) (rnd := rnd)) (xRf i)))))
          0 (List.finRange n) ≤ eps
      simpa [tensorDistance, linfNorm, RuntimeApprox.linfNorm, tensorToSpec, MathFunctions.abs,
        Spec.mapTensor] using hfold

end NFBackend

end RuntimeApprox
end Proofs
