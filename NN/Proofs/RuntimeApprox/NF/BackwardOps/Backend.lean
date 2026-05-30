/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.BackwardOps.Sparse

/-!
# NF Backward Approximation Backend

Core approximation lemmas for the rounded NF backend: constants, sparse context writes, and the
context-addition bound used when reverse-mode contributions have to be accumulated.
-/

@[expose] public section

namespace Proofs
namespace RuntimeApprox

open Spec
open Tensor
open NN.MLTheory.Robustness.Spec

noncomputable section

namespace NFBackend

open TorchLean.Floats
open Proofs.RuntimeRoundingApprox

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]
variable {rnd : ℝ → ℤ} [NeuralValidRndToNearest rnd]

local notation "R" => TorchLean.Floats.NF β fexp rnd

-- `toSpec` is already defined in `NN.Proofs.RuntimeApprox.NF.Ops`.

private lemma toSpec_one_bound :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (1 : R) - (1 : ℝ)) ≤
      neuralUlp β fexp (1 : ℝ) TrainingPhase.forward / 2 := by
  -- `1 : R` is `NF.ofReal 1`, so this is the standard single-step rounding error bound.
  simpa [NFBackend.toSpec, TorchLean.Floats.NF.toReal, TorchLean.Floats.NF.ofReal,
    TorchLean.Floats.NF.roundR,
    Proofs.RuntimeRoundingApprox.roundR] using
      (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) (1 : ℝ))

omit [NeuralValidExp fexp] [NeuralValidRndToNearest rnd] in
lemma approxT_fill_const {cS : ℝ} {cR : R} {eps : ℝ} (h : abs (toSpec (β := β) (fexp := fexp) (rnd
  := rnd) cR - cS) ≤ eps) :
    ∀ {s : Shape}, approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Spec.fill cS s) (Spec.fill cR s) eps := by
  intro s
  induction s with
  | scalar =>
      simpa [Spec.fill] using (approxT_scalar_iff (α := R)
        (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) (x := cS) (xR := cR) (eps := eps)
          |>.2 h)
  | dim n s ih =>
      cases n with
      | zero =>
          -- vacuous: `Fin 0` is empty, and the `foldl max` is `0`.
          have heps : 0 ≤ eps := le_trans (abs_nonneg _) h
          simpa [Spec.fill, approxT, approxWith, tensorToSpec, linfNorm,
            RuntimeApprox.linfNorm,
            tensorDistance, NN.MLTheory.Robustness.Spec.tensorDistance.tensor_sub,
            tensorLinfNorm, Spec.mapTensor] using heps
      | succ n =>
          -- Each component satisfies the IH; take the `foldl max` upper bound.
          have heps : 0 ≤ eps := le_trans (abs_nonneg _) h
          -- Unfold `approxT` for the outer `.dim`.
          -- Reduce to a `foldl max` bound over component distances.
          have hcomp :
              ∀ i : Fin (Nat.succ n),
                tensorDistance (α := SpecScalar) linfNorm
                    (Spec.fill cS s)
                    (tensorToSpec (α := R)
                      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) (Spec.fill cR s))
                  ≤ eps := by
            intro i
            -- This is exactly the IH at the inner shape (independent of `i`).
            simpa [approxT, approxWith] using ih
          have hfold :=
            List.foldl_max_le_of_le (List.finRange (Nat.succ n))
              (fun i =>
                tensorDistance (α := SpecScalar) linfNorm
                    (Spec.fill cS s)
                    (tensorToSpec (α := R)
                      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) (Spec.fill cR s)))
              (acc := (0 : ℝ)) (eps := eps) heps (by
                intro i hi
                simpa using hcomp i)
          -- Finish by rewriting back to the dim tensor form.
          simpa [Spec.fill, approxT, approxWith, tensorToSpec, linfNorm,
            RuntimeApprox.linfNorm,
            tensorDistance, NN.MLTheory.Robustness.Spec.tensorDistance.tensor_sub,
            tensorLinfNorm, Spec.mapTensor] using hfold

lemma approxT_fill_one :
    ∀ {s : Shape}, approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Spec.fill (1 : ℝ) s) (Spec.fill (1 : R) s) (neuralUlp β fexp (1 : ℝ) TrainingPhase.forward /
        2) := by
  intro s
  exact
    approxT_fill_const (β := β) (fexp := fexp) (rnd := rnd)
      (cS := (1 : ℝ)) (cR := (1 : R))
      (eps := neuralUlp β fexp (1 : ℝ) TrainingPhase.forward / 2)
      (toSpec_one_bound (β := β) (fexp := fexp) (rnd := rnd)) (s := s)

lemma approxT_fill_zero :
    ∀ {s : Shape}, approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Spec.fill (0 : ℝ) s) (Spec.fill (0 : R) s) 0 := by
  intro s
  refine approxT_fill_const (β := β) (fexp := fexp) (rnd := rnd) (cS := (0 : ℝ)) (cR := (0 : R))
    (eps := 0) ?_ (s := s)
  simp

lemma idx_shape_eq_of_i_eq {Γ : List Shape} {s₁ s₂ : Shape} (a : Idx Γ s₁) (b : Idx Γ s₂)
    (h : a.i = b.i) : s₁ = s₂ := by
  have : Γ.get a.i = Γ.get b.i := by simp [h]
  calc
    s₁ = Γ.get a.i := by simpa using a.h.symm
    _ = Γ.get b.i := this
    _ = s₂ := by simpa using b.h

/--
Cast a tensor across a shape equality induced by equal `Idx` positions.

Given `a : Idx Γ s₁`, `b : Idx Γ s₂`, and `h : a.i = b.i`, this produces a function
`Tensor α s₂ → Tensor α s₁` that casts along the implied equality `s₁ = s₂`.
-/
def tensorCastOfIdxEq {α : Type} {Γ : List Shape} {s₁ s₂ : Shape} (a : Idx Γ s₁) (b : Idx Γ s₂)
    (h : a.i = b.i) : Tensor α s₂ → Tensor α s₁ :=
  Spec.tensorCast (α := α) (s := s₂) (t := s₁) (idx_shape_eq_of_i_eq (Γ := Γ) (a := a) (b := b)
    h).symm

omit [NeuralValidExp fexp] [NeuralValidRndToNearest rnd] in
lemma approxT_tensor_cast {s t : Shape} (h : s = t)
    {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ}
    (hx : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps) :
    approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Spec.tensorCast (α := SpecScalar) (s := s) (t := t) h xS)
      (Spec.tensorCast (α := R) (s := s) (t := t) h xR)
      eps := by
  cases h
  simpa [Spec.tensorCast] using hx

lemma approxCtx_zeros {Γ : List Shape} :
    approxCtx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (TList.zeros (α := SpecScalar) (ss := Γ))
      (TList.zeros (α := R) (ss := Γ))
      (EList.zeros (ss := Γ)) := by
  induction Γ with
  | nil =>
      simp [TList.zeros, EList.zeros, approxCtx]
  | cons s Γ ih =>
      refine And.intro ?_ ih
      simpa [TList.zeros, EList.zeros] using (approxT_fill_zero (β := β) (fexp := fexp) (rnd := rnd)
        (s := s))

lemma approxCtx_setIdx {Γ : List Shape} {s : Shape} (idx : Idx Γ s)
    {tS : SpecTensor s} {tR : Tensor R s} {eps : ℝ}
    (ht : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) tS tR eps) :
    approxCtx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (TList.setIdx (α := SpecScalar) (Γ := Γ) (s := s) idx tS)
      (TList.setIdx (α := R) (Γ := Γ) (s := s) idx tR)
      (EList.setIdx (Γ := Γ) (s := s) idx eps) := by
  classical
  cases idx with
  | mk i hshape =>
      induction Γ with
      | nil =>
          cases i with
          | mk val isLt =>
              exact False.elim ((Nat.not_lt_zero val) isLt)
      | cons s0 Γ ih =>
          cases i with
          | mk iVal hiVal =>
              cases iVal with
              | zero =>
                  -- head is the distinguished index
                  cases hshape
                  refine And.intro ?_ ?_
                  · simpa [TList.setIdx, EList.setIdx] using ht
                  · simpa [TList.setIdx, EList.setIdx] using
                      (approxCtx_zeros (β := β) (fexp := fexp) (rnd := rnd) (Γ := Γ))
              | succ j =>
                  have hshape' : Γ.get ⟨j, Nat.lt_of_succ_lt_succ hiVal⟩ = s := by
                    simpa using hshape
                  have iht := ih (i := ⟨j, Nat.lt_of_succ_lt_succ hiVal⟩) (hshape := hshape')
                  refine And.intro ?_ ?_
                  · simpa [TList.setIdx, EList.setIdx] using
                      (approxT_fill_zero (β := β) (fexp := fexp) (rnd := rnd) (s := s0))
                  · simpa [TList.setIdx, EList.setIdx] using iht

lemma approxCtx_set2Idx_ne {Γ : List Shape} {s₁ s₂ : Shape} (a : Idx Γ s₁) (b : Idx Γ s₂)
    {t₁S : SpecTensor s₁} {t₁R : Tensor R s₁} {eps₁ : ℝ}
    {t₂S : SpecTensor s₂} {t₂R : Tensor R s₂} {eps₂ : ℝ}
    (h₁ : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) t₁S t₁R eps₁)
    (h₂ : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) t₂S t₂R eps₂)
    (hne : a.i ≠ b.i) :
    approxCtx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (TList.set2Idx (α := SpecScalar) (Γ := Γ) (s₁ := s₁) (s₂ := s₂) a t₁S b t₂S)
      (TList.set2Idx (α := R) (Γ := Γ) (s₁ := s₁) (s₂ := s₂) a t₁R b t₂R)
      (EList.set2Idx (Γ := Γ) (s₁ := s₁) (s₂ := s₂) a eps₁ b eps₂ 0) := by
  classical
  induction Γ with
  | nil =>
      cases a with
      | mk i _ =>
          cases i with
          | mk val isLt =>
              exact False.elim ((Nat.not_lt_zero val) isLt)
  | cons s0 Γ ih =>
      cases a with
      | mk ia haShape =>
          cases b with
          | mk ib hbShape =>
              cases ia with
              | mk iaVal iaLt =>
                  cases ib with
                  | mk ibVal ibLt =>
                      cases iaVal with
                      | zero =>
                          cases ibVal with
                          | zero =>
                              exact False.elim (hne (by rfl))
                          | succ j =>
                              cases haShape
                              refine And.intro ?_ ?_
                              · simpa [TList.set2Idx, EList.set2Idx] using h₁
                              ·
                                let bTail : Idx Γ s₂ :=
                                  ⟨⟨j, Nat.lt_of_succ_lt_succ ibLt⟩, by simpa using hbShape⟩
                                have := approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
                                  (Γ := Γ) (s := s₂) bTail (tS := t₂S) (tR := t₂R) (eps := eps₂) h₂
                                simpa [TList.set2Idx, EList.set2Idx, bTail] using this
                      | succ i =>
                          cases ibVal with
                          | zero =>
                              cases hbShape
                              refine And.intro ?_ ?_
                              · simpa [TList.set2Idx, EList.set2Idx] using h₂
                              ·
                                let aTail : Idx Γ s₁ :=
                                  ⟨⟨i, Nat.lt_of_succ_lt_succ iaLt⟩, by simpa using haShape⟩
                                have := approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
                                  (Γ := Γ) (s := s₁) aTail (tS := t₁S) (tR := t₁R) (eps := eps₁) h₁
                                simpa [TList.set2Idx, EList.set2Idx, aTail] using this
                          | succ j =>
                              let aTail : Idx Γ s₁ :=
                                ⟨⟨i, Nat.lt_of_succ_lt_succ iaLt⟩, by simpa using haShape⟩
                              let bTail : Idx Γ s₂ :=
                                ⟨⟨j, Nat.lt_of_succ_lt_succ ibLt⟩, by simpa using hbShape⟩
                              have hneTail : aTail.i ≠ bTail.i := by
                                intro hij
                                apply hne
                                apply Fin.ext
                                have : i = j := by
                                  simpa [aTail, bTail] using congrArg Fin.val hij
                                simp [this]
                              refine And.intro ?_ ?_
                              · simpa [TList.set2Idx, EList.set2Idx] using
                                  (approxT_fill_zero (β := β) (fexp := fexp) (rnd := rnd) (s := s0))
                              ·
                                have := ih (a := aTail) (b := bTail) hneTail
                                simpa [TList.set2Idx, EList.set2Idx, aTail, bTail] using this

lemma approxCtx_set3Idx_ne {Γ : List Shape} {s₁ s₂ s₃ : Shape} (a : Idx Γ s₁) (b : Idx Γ s₂) (c :
  Idx Γ s₃)
    {t₁S : SpecTensor s₁} {t₁R : Tensor R s₁} {eps₁ : ℝ}
    {t₂S : SpecTensor s₂} {t₂R : Tensor R s₂} {eps₂ : ℝ}
    {t₃S : SpecTensor s₃} {t₃R : Tensor R s₃} {eps₃ : ℝ}
    (h₁ : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) t₁S t₁R eps₁)
    (h₂ : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) t₂S t₂R eps₂)
    (h₃ : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) t₃S t₃R eps₃)
    (hab : a.i ≠ b.i) (hac : a.i ≠ c.i) (hbc : b.i ≠ c.i) :
    approxCtx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (TList.set3IdxNe (α := SpecScalar) (Γ := Γ) (s₁ := s₁) (s₂ := s₂) (s₃ := s₃) a t₁S b t₂S c
        t₃S hab hac hbc)
      (TList.set3IdxNe (α := R) (Γ := Γ) (s₁ := s₁) (s₂ := s₂) (s₃ := s₃) a t₁R b t₂R c t₃R hab hac
        hbc)
      (EList.set3IdxNe (Γ := Γ) (s₁ := s₁) (s₂ := s₂) (s₃ := s₃) a eps₁ b eps₂ c eps₃ hab hac hbc)
        := by
  classical
  induction Γ with
  | nil =>
      cases a with
      | mk i _ =>
          cases i with
          | mk val isLt =>
              exact False.elim ((Nat.not_lt_zero val) isLt)
  | cons s0 Γ ih =>
      cases a with
      | mk ia haShape =>
          cases b with
          | mk ib hbShape =>
              cases c with
              | mk ic hcShape =>
                  cases ia with
                  | mk iaVal iaLt =>
                      cases ib with
                      | mk ibVal ibLt =>
                          cases ic with
                          | mk icVal icLt =>
                              cases iaVal with
                              | zero =>
                                  cases ibVal with
                                  | zero =>
                                      exact False.elim (hab rfl)
                                  | succ j =>
                                      cases icVal with
                                      | zero =>
                                          exact False.elim (hac rfl)
                                      | succ k =>
                                          cases haShape
                                          refine And.intro ?_ ?_
                                          · simpa [TList.set3IdxNe, EList.set3IdxNe] using h₁
                                          ·
                                            let bTail : Idx Γ s₂ := ⟨⟨j, Nat.lt_of_succ_lt_succ
                                              ibLt⟩, by simpa using hbShape⟩
                                            let cTail : Idx Γ s₃ := ⟨⟨k, Nat.lt_of_succ_lt_succ
                                              icLt⟩, by simpa using hcShape⟩
                                            have hbcTail : bTail.i ≠ cTail.i := by
                                              intro hij
                                              apply hbc
                                              apply Fin.ext
                                              have : j = k := by
                                                simpa [bTail, cTail] using congrArg Fin.val hij
                                              simp [this]
                                            have :=
                                              approxCtx_set2Idx_ne (β := β) (fexp := fexp) (rnd :=
                                                rnd)
                                                (Γ := Γ) (s₁ := s₂) (s₂ := s₃) bTail cTail
                                                (t₁S := t₂S) (t₁R := t₂R) (eps₁ := eps₂)
                                                (t₂S := t₃S) (t₂R := t₃R) (eps₂ := eps₃)
                                                h₂ h₃ hbcTail
                                            simpa [TList.set3IdxNe, EList.set3IdxNe, bTail, cTail]
                                              using this
                              | succ i =>
                                  cases ibVal with
                                  | zero =>
                                      cases icVal with
                                      | zero =>
                                          exact False.elim (hbc rfl)
                                      | succ k =>
                                          cases hbShape
                                          refine And.intro ?_ ?_
                                          · simpa [TList.set3IdxNe, EList.set3IdxNe] using h₂
                                          ·
                                            let aTail : Idx Γ s₁ := ⟨⟨i, Nat.lt_of_succ_lt_succ
                                              iaLt⟩, by simpa using haShape⟩
                                            let cTail : Idx Γ s₃ := ⟨⟨k, Nat.lt_of_succ_lt_succ
                                              icLt⟩, by simpa using hcShape⟩
                                            have hacTail : aTail.i ≠ cTail.i := by
                                              intro hij
                                              apply hac
                                              apply Fin.ext
                                              have : i = k := by
                                                simpa [aTail, cTail] using congrArg Fin.val hij
                                              simp [this]
                                            have :=
                                              approxCtx_set2Idx_ne (β := β) (fexp := fexp) (rnd :=
                                                rnd)
                                                (Γ := Γ) (s₁ := s₁) (s₂ := s₃) aTail cTail
                                                (t₁S := t₁S) (t₁R := t₁R) (eps₁ := eps₁)
                                                (t₂S := t₃S) (t₂R := t₃R) (eps₂ := eps₃)
                                                h₁ h₃ hacTail
                                            simpa [TList.set3IdxNe, EList.set3IdxNe, aTail, cTail]
                                              using this
                                  | succ j =>
                                      cases icVal with
                                      | zero =>
                                          cases hcShape
                                          refine And.intro ?_ ?_
                                          · simpa [TList.set3IdxNe, EList.set3IdxNe] using h₃
                                          ·
                                            let aTail : Idx Γ s₁ := ⟨⟨i, Nat.lt_of_succ_lt_succ
                                              iaLt⟩, by simpa using haShape⟩
                                            let bTail : Idx Γ s₂ := ⟨⟨j, Nat.lt_of_succ_lt_succ
                                              ibLt⟩, by simpa using hbShape⟩
                                            have habTail : aTail.i ≠ bTail.i := by
                                              intro hij
                                              apply hab
                                              apply Fin.ext
                                              have : i = j := by
                                                simpa [aTail, bTail] using congrArg Fin.val hij
                                              simp [this]
                                            have :=
                                              approxCtx_set2Idx_ne (β := β) (fexp := fexp) (rnd :=
                                                rnd)
                                                (Γ := Γ) (s₁ := s₁) (s₂ := s₂) aTail bTail
                                                (t₁S := t₁S) (t₁R := t₁R) (eps₁ := eps₁)
                                                (t₂S := t₂S) (t₂R := t₂R) (eps₂ := eps₂)
                                                h₁ h₂ habTail
                                            simpa [TList.set3IdxNe, EList.set3IdxNe, aTail, bTail]
                                              using this
                                      | succ k =>
                                          let aTail : Idx Γ s₁ := ⟨⟨i, Nat.lt_of_succ_lt_succ iaLt⟩,
                                            by simpa using haShape⟩
                                          let bTail : Idx Γ s₂ := ⟨⟨j, Nat.lt_of_succ_lt_succ ibLt⟩,
                                            by simpa using hbShape⟩
                                          let cTail : Idx Γ s₃ := ⟨⟨k, Nat.lt_of_succ_lt_succ icLt⟩,
                                            by simpa using hcShape⟩
                                          have habTail : aTail.i ≠ bTail.i := by
                                            intro hij
                                            apply hab
                                            apply Fin.ext
                                            have : i = j := by
                                              simpa [aTail, bTail] using congrArg Fin.val hij
                                            simp [this]
                                          have hacTail : aTail.i ≠ cTail.i := by
                                            intro hij
                                            apply hac
                                            apply Fin.ext
                                            have : i = k := by
                                              simpa [aTail, cTail] using congrArg Fin.val hij
                                            simp [this]
                                          have hbcTail : bTail.i ≠ cTail.i := by
                                            intro hij
                                            apply hbc
                                            apply Fin.ext
                                            have : j = k := by
                                              simpa [bTail, cTail] using congrArg Fin.val hij
                                            simp [this]
                                          have iht := ih (a := aTail) (b := bTail) (c := cTail)
                                            habTail hacTail hbcTail
                                          refine And.intro ?_ ?_
                                          · simpa [TList.set3IdxNe, EList.set3IdxNe] using
                                              (approxT_fill_zero (β := β) (fexp := fexp) (rnd :=
                                                rnd) (s := s0))
                                          · simpa [TList.set3IdxNe, EList.set3IdxNe, aTail, bTail,
                                            cTail] using iht

-- ---------------------------------------------------------------------------
-- Context-wise addition bound (used by global backprop accumulation)
-- ---------------------------------------------------------------------------

/--
Context-wise addition bound (NF runtime vs spec).

This produces an `EList` of `linf_norm` bounds for adding two contexts elementwise, and is used when
reverse-mode accumulation must combine contributions from multiple consumers.
-/
def ctxAddBound : {Δ : List Shape} → EList Δ → EList Δ → TList R Δ → TList R Δ → EList Δ
  | [], .nil, .nil, .nil, .nil => .nil
  | _ :: ss, .cons ex exs, .cons ey eys, .cons x xs, .cons y ys =>
      .cons (linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := _) ex ey x y))
        (ctxAddBound (Δ := ss) exs eys xs ys)

/--
Soundness of context-wise addition under `approxCtx`.

If `xS ~ xR ± epsx` and `yS ~ yR ± epsy`, then `(xS + yS) ~ (xR + yR)` with error bounded by
`ctxAddBound epsx epsy xR yR`.
-/
theorem approxCtx_add {Δ : List Shape} :
    ∀ (xS yS : TList SpecScalar Δ) (xR yR : TList R Δ) (epsx epsy : EList Δ),
      approxCtx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR epsx →
      approxCtx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) yS yR epsy →
        approxCtx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (TList.add (α := SpecScalar) xS yS)
          (TList.add (α := R) xR yR)
          (ctxAddBound (β := β) (fexp := fexp) (rnd := rnd) epsx epsy xR yR) := by
  intro xS yS xR yR epsx epsy hx hy
  induction Δ with
  | nil =>
      cases xS
      cases yS
      cases xR
      cases yR
      cases epsx
      cases epsy
      simp [TList.add, ctxAddBound, approxCtx]
  | cons s ss ih =>
      cases xS with
      | cons xSh xSt =>
          cases yS with
          | cons ySh ySt =>
              cases xR with
              | cons xRh xRt =>
                  cases yR with
                  | cons yRh yRt =>
                      cases epsx with
                      | cons ex exs =>
                          cases epsy with
                          | cons ey eys =>
                              refine And.intro ?_ ?_
                              · -- head uses `approxT_add_spec`
                                have hx0 : approxT (α := R) (toSpec := toSpec (β := β) (fexp :=
                                  fexp) (rnd := rnd)) xSh xRh ex :=
                                  hx.1
                                have hy0 : approxT (α := R) (toSpec := toSpec (β := β) (fexp :=
                                  fexp) (rnd := rnd)) ySh yRh ey :=
                                  hy.1
                                simpa [TList.add, ctxAddBound] using
                                  (approxT_add_spec (β := β) (fexp := fexp) (rnd := rnd)
                                    (s := s) (xS := xSh) (yS := ySh) (xR := xRh) (yR := yRh)
                                    (epsx := ex) (epsy := ey) hx0 hy0)
                              · -- tail by IH
                                have hxT : approxCtx (α := R) (toSpec := toSpec (β := β) (fexp :=
                                  fexp) (rnd := rnd)) xSt xRt exs :=
                                  hx.2
                                have hyT : approxCtx (α := R) (toSpec := toSpec (β := β) (fexp :=
                                  fexp) (rnd := rnd)) ySt yRt eys :=
                                  hy.2
                                simpa [TList.add, ctxAddBound] using
                                  ih (xS := xSt) (yS := ySt) (xR := xRt) (yR := yRt) (epsx := exs)
                                    (epsy := eys) hxT hyT

end NFBackend

end

end RuntimeApprox
end Proofs
