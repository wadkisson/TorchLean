/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Bridge.FP32
public import NN.Floats.IEEEExec.Rules.SpecialRules

/-!
# Total FP32 Bridge: Finite and Special Values

“Total” bridge theorems combining:

- `IEEE32Exec`'s proved NaN/Inf propagation rules, and
- the `FP32`-on-`ℝ` refinement theorems for the finite/no-overflow branch (`Bridge/FP32.lean`).

The key end-user view is `toReal?`:
- `toReal? x = none` for NaN/Inf,
- `toReal? x = some r` for finite values, with `r : ℝ`.

In most of TorchLean, the finite path is treated as real arithmetic + float32 rounding while
special-value behavior is kept explicit. This file packages that split in one place.

The per-op lemmas are phrased in the style:

`toReal? (op …) = if isFinite (op …) then some (fp32Round …) else none`.

That makes the trust boundary readable at the call site: the `if` is exactly where NaN/Inf (or
overflow-to-Inf) can occur.

Background references (for float32 rounding/special values):
- IEEE 754-2019: https://doi.org/10.1109/IEEESTD.2019.8766229
- Goldberg (1991): https://doi.org/10.1145/103162.103163
- Flocq (Boldo–Melquiond, 2011): https://doi.org/10.1109/ARITH.2011.40
-/

@[expose] public section


namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

noncomputable section

/-! ## Basic facts: `isFinite` ↔ `toDyadic?`/`toReal?` -/

/-- `toDyadic? x = none` implies `x` is not finite. -/
theorem isFinite_eq_false_of_toDyadic?_eq_none (x : IEEE32Exec) (hx : toDyadic? x = none) :
    isFinite x = false := by
  unfold toDyadic? at hx
  cases hcond : (isNaN x || isInf x) with
  | false =>
      -- In the non-special branch, `toDyadic?` always returns `some …`.
      cases hE : (expField x == 0) <;> cases hF : (fracField x == 0) <;>
        simp [hcond, hE, hF] at hx
  | true =>
      cases hnan : isNaN x with
      | true =>
          have hexp : expField x = expAllOnes := expField_eq_expAllOnes_of_isNaN (x := x) hnan
          exact isFinite_eq_false_of_expField_eq_expAllOnes (x := x) hexp
      | false =>
          -- `isNaN x = false` and `isNaN x || isInf x = true` implies `isInf x = true`.
          cases hinf : isInf x with
          | true =>
              have hexp : expField x = expAllOnes := expField_eq_expAllOnes_of_isInf (x := x) hinf
              exact isFinite_eq_false_of_expField_eq_expAllOnes (x := x) hexp
          | false =>
              -- Contradiction: the disjunction cannot be `true`.
              have hcondFalse : (isNaN x || isInf x) = false := by simp [hnan, hinf]
              have hcontra : False := by
                have : (false : Bool) = true := by
                  simp [hcondFalse]  at hcond
                cases this
              exact hcontra.elim

/-- If `x` is not finite, then `toDyadic? x = none`. -/
theorem toDyadic?_eq_none_of_isFinite_eq_false (x : IEEE32Exec) (hx : isFinite x = false) :
    toDyadic? x = none := by
  -- By cases on the fraction field: expField=all-ones gives either Inf or NaN.
  unfold toDyadic?
  unfold isFinite at hx
  have hx' : (expField x != expAllOnes) = false := by simpa using hx
  have hexp : expField x = expAllOnes := by
    by_contra hne
    have htrue : (expField x != expAllOnes) = true := (bne_iff_ne).2 hne
    have : False := by
      simp [htrue] at hx'
    exact this.elim
  have hexpB : (expField x == expAllOnes) = true := (beq_iff_eq).2 hexp
  by_cases hfrac : fracField x = 0
  · have hfracB : (fracField x == 0) = true := (beq_iff_eq).2 hfrac
    have hinf : isInf x = true := by simp [isInf, hexpB, hfracB]
    simp [hinf]
  · have hfracNeB : (fracField x != 0) = true := (bne_iff_ne).2 hfrac
    have hnan : isNaN x = true := by simp [isNaN, hexpB, hfracNeB]
    simp [hnan]

/-- `toReal?` returns `none` on non-finite values. -/
theorem toReal?_eq_none_of_isFinite_eq_false (x : IEEE32Exec) (hx : isFinite x = false) :
    toReal? x = none := by
  have hdy : toDyadic? x = none := toDyadic?_eq_none_of_isFinite_eq_false (x := x) hx
  simp [toReal?, hdy]

/-- On finite values, `toReal? x` is just `some (toReal x)`. -/
theorem toReal?_eq_some_toReal_of_isFinite_eq_true (x : IEEE32Exec) (hx : isFinite x = true) :
    toReal? x = some (toReal x) := by
  -- `toDyadic? x` cannot be `none`, otherwise `isFinite x = false`.
  cases hdy : toDyadic? x with
  | none =>
      have hxFalse : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdy
      have : False := by
        simp [hx]  at hxFalse
      exact this.elim
  | some d =>
      simp [toReal?, toReal, hdy]

/-! ## Helpers: NaN/Inf/zero interactions -/

/-- NaNs are not infinities. -/
theorem isInf_eq_false_of_isNaN (x : IEEE32Exec) (hx : isNaN x = true) :
    isInf x = false := by
  -- `isNaN` means `fracField x != 0`, hence `fracField x == 0` is false.
  have hnan : (expField x == expAllOnes && fracField x != 0) = true := by
    simpa [isNaN] using hx
  have hfracNe : (fracField x != 0) = true := by
    have : (expField x == expAllOnes) = true ∧ (fracField x != 0) = true := by
      simpa [Bool.and_eq_true] using hnan
    exact this.2
  have hne : fracField x ≠ 0 := (bne_iff_ne).1 hfracNe
  have hfracEqFalse : (fracField x == 0) = false := (beq_eq_false_iff_ne).2 hne
  simp [isInf, hfracEqFalse]

/-- Infinities are not zeros. -/
theorem isZero_eq_false_of_isInf (x : IEEE32Exec) (hx : isInf x = true) :
    isZero x = false := by
  have hinf : (expField x == expAllOnes && fracField x == 0) = true := by
    simpa [isInf] using hx
  have hexp : expField x = expAllOnes := by
    have : (expField x == expAllOnes) = true ∧ (fracField x == 0) = true := by
      simpa [Bool.and_eq_true] using hinf
    exact (beq_iff_eq).1 this.1
  have hexpNe0 : expField x ≠ 0 := by
    intro h0
    have : expAllOnes = 0 := by simpa [hexp] using h0
    exact (by decide : (expAllOnes : UInt32) ≠ 0) this
  have hexp0 : (expField x == 0) = false := (beq_eq_false_iff_ne).2 hexpNe0
  simp [isZero, hexp0]

/-- If dyadic decoding fails and the value is not NaN, then it must be infinite. -/
theorem isInf_eq_true_of_toDyadic?_eq_none_of_isNaN_eq_false (x : IEEE32Exec)
    (hx : toDyadic? x = none) (hxNaN : isNaN x = false) :
    isInf x = true := by
  -- `toDyadic? x = none` means we took the `isNaN x || isInf x` branch.
  unfold toDyadic? at hx
  cases hcond : (isNaN x || isInf x) with
  | true =>
      -- With `isNaN x = false`, the disjunction being true forces `isInf x = true`.
      cases hinf : isInf x with
      | true => rfl
      | false =>
          have : (isNaN x || isInf x) = false := by simp [hxNaN, hinf]
          have : False := by
            simp [this]  at hcond
          exact this.elim
  | false =>
      -- Contradiction: in the non-special branch `toDyadic?` always returns `some …`.
      cases hE : (expField x == 0) <;> cases hF : (fracField x == 0) <;>
        simp [hcond, hE, hF] at hx

end

end IEEE32Exec

end TorchLean.Floats.IEEE754
