/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Bridge.FP32Total.Effective

/-!
# Total FP32 Bridge: Minimum and Maximum

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

/-! ## Total `minimum` and `maximum` (including infinities) -/

/--
Total characterization of `toReal? (minimum x y)` via `toReal? x` and `toReal? y`.

This lemma covers the cases where one side is `+∞` (which acts as a neutral element for `min`) and
the cases where `toReal?` is `none` because of NaN.
-/
theorem toReal?_minimum_eq_match_total (x y : IEEE32Exec) :
    toReal? (minimum x y) =
      match toReal? x, toReal? y with
      | some rx, some ry => some (min rx ry)
      | some rx, none => if isInf y && (!signBit y) then some rx else none
      | none, some ry => if isInf x && (!signBit x) then some ry else none
      | none, none => none := by
  classical
  cases hx : toDyadic? x with
  | some dx =>
      cases hy : toDyadic? y with
      | some dy =>
          -- Both finite.
          have hxFin : isFinite x = true := by
            cases hfx : isFinite x with
            | true => rfl
            | false =>
                have hnone : toDyadic? x = none := toDyadic?_eq_none_of_isFinite_eq_false (x := x)
                  hfx
                have : (none : Option Dyadic) = some dx := by
                  simp [hnone]  at hx
                cases this
          have hyFin : isFinite y = true := by
            cases hfy : isFinite y with
            | true => rfl
            | false =>
                have hnone : toDyadic? y = none := toDyadic?_eq_none_of_isFinite_eq_false (x := y)
                  hfy
                have : (none : Option Dyadic) = some dy := by
                  simp [hnone]  at hy
                cases this
          have hmin : toReal? (minimum x y) = some (min (toReal x) (toReal y)) :=
            toReal?_minimum_eq_min_of_isFinite (x := x) (y := y) hxFin hyFin
          simpa [IEEE32Exec.toReal?, IEEE32Exec.toReal_eq, hx, hy] using hmin
      | none =>
          -- `x` finite, `y` special.
          have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hx)
          have hxInf : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hx)
          cases hyNaN : isNaN y with
          | true =>
              -- NaN propagates.
              have hxS : isSNaN x = false := by simp [isSNaN, hxNaN]
              have hchoose : chooseNaN2 x y = some (quietNaN y) := by
                simp [chooseNaN2, hxS, hxNaN, hyNaN]
              have hminEq : minimum x y = quietNaN y := by simp [minimum, hchoose]
              have hfinFalse : isFinite (quietNaN y) = false :=
                isFinite_eq_false_of_chooseNaN2_some (x := x) (y := y) (nan := quietNaN y) hchoose
              have hnone : toReal? (quietNaN y) = none :=
                toReal?_eq_none_of_isFinite_eq_false (x := quietNaN y) hfinFalse
              have hyInfFalse : isInf y = false := isInf_eq_false_of_isNaN (x := y) hyNaN
              simpa [hminEq, hnone, IEEE32Exec.toReal?, hx, hy, hyInfFalse, hyNaN]
          | false =>
              -- Inf case.
              have hyInf : isInf y = true :=
                isInf_eq_true_of_toDyadic?_eq_none_of_isNaN_eq_false (x := y) hy hyNaN
              have hchoose : chooseNaN2 x y = none :=
                chooseNaN2_none_of_not_isNaN (x := x) (y := y) hxNaN hyNaN
              cases hsy : signBit y with
                | true =>
                    have hcmp : compare x y = some .gt := by
                      simp [compare, hxNaN, hyNaN, hxInf, hyInf, hsy]
                    have hminEq : minimum x y = y := by simp [minimum, hchoose, hcmp]
                    simp [hminEq, IEEE32Exec.toReal?, hx, hyInf]
                | false =>
                    have hcmp : compare x y = some .lt := by
                      simp [compare, hxNaN, hyNaN, hxInf, hyInf, hsy]
                    have hminEq : minimum x y = x := by simp [minimum, hchoose, hcmp]
                    simp [hminEq, IEEE32Exec.toReal?, hx, hyInf]
  | none =>
      cases hy : toDyadic? y with
      | some dy =>
          -- `x` special, `y` finite.
          have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hy)
          have hyInf : isInf y = false := isInf_eq_false_of_toDyadic?_some (hx := hy)
          cases hxNaN : isNaN x with
          | true =>
              -- NaN propagates.
              have hyS : isSNaN y = false := by simp [isSNaN, hyNaN]
              -- `chooseNaN2 x y` always selects a NaN when `x` is NaN.
              cases hchoose : chooseNaN2 x y with
              | some nan =>
                  have hminEq : minimum x y = nan := minimum_eq_of_chooseNaN2_some (x := x) (y := y)
                    (nan := nan) hchoose
                  have hfinFalse : isFinite nan = false :=
                    isFinite_eq_false_of_chooseNaN2_some (x := x) (y := y) (nan := nan) hchoose
                  have hnone : toReal? nan = none :=
                    toReal?_eq_none_of_isFinite_eq_false (x := nan) hfinFalse
                  have hxInfFalse : isInf x = false := isInf_eq_false_of_isNaN (x := x) hxNaN
                  simpa [hminEq, hnone, IEEE32Exec.toReal?, hx, hy, hxInfFalse, hxNaN]
              | none =>
                  -- Impossible: `x` is NaN, so `chooseNaN2` cannot be `none`.
                  have : False := by
                    cases hxS : isSNaN x <;>
                      simp [chooseNaN2, hxS, hxNaN, hyS] at hchoose
                  exact this.elim
          | false =>
              -- Inf case.
              have hxInf : isInf x = true :=
                isInf_eq_true_of_toDyadic?_eq_none_of_isNaN_eq_false (x := x) hx hxNaN
              have hchoose : chooseNaN2 x y = none :=
                chooseNaN2_none_of_not_isNaN (x := x) (y := y) hxNaN hyNaN
              cases hsx : signBit x with
                | true =>
                    -- `x = -Inf`, so `minimum x y = x`.
                    have hcmp : compare x y = some .lt := by
                      simp [compare, hxNaN, hyNaN, hxInf, hyInf, hsx]
                    have hminEq : minimum x y = x := by simp [minimum, hchoose, hcmp]
                    simp [hminEq, IEEE32Exec.toReal?, hy, hxInf]
                | false =>
                    -- `x = +Inf`, so `minimum x y = y`.
                    have hcmp : compare x y = some .gt := by
                      simp [compare, hxNaN, hyNaN, hxInf, hyInf, hsx]
                    have hminEq : minimum x y = y := by simp [minimum, hchoose, hcmp]
                    simp [hminEq, IEEE32Exec.toReal?, hy, hxInf]
      | none =>
          -- Both special.
          cases hxNaN : isNaN x with
          | true =>
              -- NaN propagates.
              cases hchoose : chooseNaN2 x y with
              | some nan =>
                  have hminEq : minimum x y = nan := minimum_eq_of_chooseNaN2_some (x := x) (y := y)
                    (nan := nan) hchoose
                  have hfinFalse : isFinite nan = false :=
                    isFinite_eq_false_of_chooseNaN2_some (x := x) (y := y) (nan := nan) hchoose
                  have hnone : toReal? nan = none :=
                    toReal?_eq_none_of_isFinite_eq_false (x := nan) hfinFalse
                  simpa [hminEq, hnone, IEEE32Exec.toReal?, hx, hy]
              | none =>
                  -- Impossible: `x` is NaN.
                  have : False := by
                    cases hxS : isSNaN x <;> cases hyS : isSNaN y <;>
                      simp [chooseNaN2, hxS, hyS, hxNaN] at hchoose
                  exact this.elim
          | false =>
              cases hyNaN : isNaN y with
              | true =>
                  -- NaN propagates.
                  cases hchoose : chooseNaN2 x y with
                  | some nan =>
                      have hminEq : minimum x y = nan := minimum_eq_of_chooseNaN2_some (x := x) (y
                        := y) (nan := nan) hchoose
                      have hfinFalse : isFinite nan = false :=
                        isFinite_eq_false_of_chooseNaN2_some (x := x) (y := y) (nan := nan) hchoose
                      have hnone : toReal? nan = none :=
                        toReal?_eq_none_of_isFinite_eq_false (x := nan) hfinFalse
                      simpa [hminEq, hnone, IEEE32Exec.toReal?, hx, hy]
                  | none =>
                      -- Impossible: `y` is NaN.
                      have : False := by
                        cases hxS : isSNaN x <;> cases hyS : isSNaN y <;>
                          simp [chooseNaN2, hxS, hyS, hyNaN, hxNaN] at hchoose
                      exact this.elim
              | false =>
                  -- Both are Infs; `minimum` returns an Inf.
                  have hxInf : isInf x = true :=
                    isInf_eq_true_of_toDyadic?_eq_none_of_isNaN_eq_false (x := x) hx hxNaN
                  have hyInf : isInf y = true :=
                    isInf_eq_true_of_toDyadic?_eq_none_of_isNaN_eq_false (x := y) hy hyNaN
                  have hchoose : chooseNaN2 x y = none :=
                    chooseNaN2_none_of_not_isNaN (x := x) (y := y) hxNaN hyNaN
                  have hx0 : isZero x = false := isZero_eq_false_of_isInf (x := x) hxInf
                  have hy0 : isZero y = false := isZero_eq_false_of_isInf (x := y) hyInf
                  cases hs : (signBit x == signBit y) with
                  | true =>
                      have hcmp : compare x y = some .eq := by
                        simp [compare, hxNaN, hyNaN, hxInf, hyInf, hs]
                      have hminEq : minimum x y = x := by
                        simp [minimum, hchoose, hcmp, hx0, hy0]
                      simp [hminEq, IEEE32Exec.toReal?, hx, hy]
                  | false =>
                      cases hsx : signBit x with
                      | true =>
                          -- Then `signBit y = false` (since `==` is false).
                          cases hsy : signBit y with
                          | true =>
                              have : (signBit x == signBit y) = true := by simp [hsx, hsy]
                              have : False := by
                                simp [this]  at hs
                              exact this.elim
                          | false =>
                              have hcmp : compare x y = some .lt := by
                                simp [compare, hxNaN, hyNaN, hxInf, hyInf, hsx, hsy]
                              have hminEq : minimum x y = x := by simp [minimum, hchoose, hcmp]
                              simp [hminEq, IEEE32Exec.toReal?, hx, hy]
                      | false =>
                          -- Then `signBit y = true` (since `==` is false).
                          cases hsy : signBit y with
                          | false =>
                              have : (signBit x == signBit y) = true := by simp [hsx, hsy]
                              have : False := by
                                simp [this]  at hs
                              exact this.elim
                          | true =>
                              have hcmp : compare x y = some .gt := by
                                simp [compare, hxNaN, hyNaN, hxInf, hyInf, hsx, hsy]
                              have hminEq : minimum x y = y := by simp [minimum, hchoose, hcmp]
                              simp [hminEq, IEEE32Exec.toReal?, hx, hy]

/--
Total characterization of `toReal? (maximum x y)` via `toReal? x` and `toReal? y`.

This lemma covers the cases where one side is `-∞` (which acts as a neutral element for `max`) and
the cases where `toReal?` is `none` because of NaN.
-/
theorem toReal?_maximum_eq_match_total (x y : IEEE32Exec) :
    toReal? (maximum x y) =
      match toReal? x, toReal? y with
      | some rx, some ry => some (max rx ry)
      | some rx, none => if isInf y && (signBit y) then some rx else none
      | none, some ry => if isInf x && (signBit x) then some ry else none
      | none, none => none := by
  classical
  cases hx : toDyadic? x with
  | some dx =>
      cases hy : toDyadic? y with
      | some dy =>
          -- Both finite.
          have hxFin : isFinite x = true := by
            cases hfx : isFinite x with
            | true => rfl
            | false =>
                have hnone : toDyadic? x = none := toDyadic?_eq_none_of_isFinite_eq_false (x := x)
                  hfx
                have : (none : Option Dyadic) = some dx := by
                  simp [hnone]  at hx
                cases this
          have hyFin : isFinite y = true := by
            cases hfy : isFinite y with
            | true => rfl
            | false =>
                have hnone : toDyadic? y = none := toDyadic?_eq_none_of_isFinite_eq_false (x := y)
                  hfy
                have : (none : Option Dyadic) = some dy := by
                  simp [hnone]  at hy
                cases this
          have hmax : toReal? (maximum x y) = some (max (toReal x) (toReal y)) :=
            toReal?_maximum_eq_max_of_isFinite (x := x) (y := y) hxFin hyFin
          simpa [IEEE32Exec.toReal?, IEEE32Exec.toReal_eq, hx, hy] using hmax
      | none =>
          -- `x` finite, `y` special.
          have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hx)
          have hxInf : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hx)
          cases hyNaN : isNaN y with
          | true =>
              -- NaN propagates.
              have hxS : isSNaN x = false := by simp [isSNaN, hxNaN]
              -- `chooseNaN2` selects a NaN; result is non-finite.
              cases hchoose : chooseNaN2 x y with
              | some nan =>
                  have hmaxEq : maximum x y = nan := maximum_eq_of_chooseNaN2_some (x := x) (y := y)
                    (nan := nan) hchoose
                  have hfinFalse : isFinite nan = false :=
                    isFinite_eq_false_of_chooseNaN2_some (x := x) (y := y) (nan := nan) hchoose
                  have hnone : toReal? nan = none :=
                    toReal?_eq_none_of_isFinite_eq_false (x := nan) hfinFalse
                  have hyInfFalse : isInf y = false := isInf_eq_false_of_isNaN (x := y) hyNaN
                  rw [hmaxEq, hnone]
                  simp [IEEE32Exec.toReal?, hx, hy, hyInfFalse]
              | none =>
                  have : False := by
                    cases hyS : isSNaN y <;> simp [chooseNaN2, hxS, hxNaN, hyNaN, hyS] at hchoose
                  exact this.elim
          | false =>
              -- Inf case.
              have hyInf : isInf y = true :=
                isInf_eq_true_of_toDyadic?_eq_none_of_isNaN_eq_false (x := y) hy hyNaN
              have hchoose : chooseNaN2 x y = none :=
                chooseNaN2_none_of_not_isNaN (x := x) (y := y) hxNaN hyNaN
              cases hsy : signBit y with
                | true =>
                    -- `y = -Inf`, so `maximum x y = x`.
                    have hcmp : compare x y = some .gt := by
                      simp [compare, hxNaN, hyNaN, hxInf, hyInf, hsy]
                    have hmaxEq : maximum x y = x := by simp [maximum, hchoose, hcmp]
                    simp [hmaxEq, IEEE32Exec.toReal?, hx, hyInf]
                | false =>
                    -- `y = +Inf`, so `maximum x y = y`.
                    have hcmp : compare x y = some .lt := by
                      simp [compare, hxNaN, hyNaN, hxInf, hyInf, hsy]
                    have hmaxEq : maximum x y = y := by simp [maximum, hchoose, hcmp]
                    simp [hmaxEq, IEEE32Exec.toReal?, hx, hyInf]
  | none =>
      cases hy : toDyadic? y with
      | some dy =>
          -- `x` special, `y` finite.
          have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hy)
          have hyInf : isInf y = false := isInf_eq_false_of_toDyadic?_some (hx := hy)
          cases hxNaN : isNaN x with
          | true =>
              -- NaN propagates.
              have hyS : isSNaN y = false := by simp [isSNaN, hyNaN]
              cases hchoose : chooseNaN2 x y with
              | some nan =>
                  have hmaxEq : maximum x y = nan := maximum_eq_of_chooseNaN2_some (x := x) (y := y)
                    (nan := nan) hchoose
                  have hfinFalse : isFinite nan = false :=
                    isFinite_eq_false_of_chooseNaN2_some (x := x) (y := y) (nan := nan) hchoose
                  have hnone : toReal? nan = none :=
                    toReal?_eq_none_of_isFinite_eq_false (x := nan) hfinFalse
                  have hxInfFalse : isInf x = false := isInf_eq_false_of_isNaN (x := x) hxNaN
                  rw [hmaxEq, hnone]
                  simp [IEEE32Exec.toReal?, hx, hy, hxInfFalse]
              | none =>
                  have : False := by
                    cases hxS : isSNaN x <;> simp [chooseNaN2, hxS, hxNaN, hyS] at hchoose
                  exact this.elim
          | false =>
              -- Inf case.
              have hxInf : isInf x = true :=
                isInf_eq_true_of_toDyadic?_eq_none_of_isNaN_eq_false (x := x) hx hxNaN
              have hchoose : chooseNaN2 x y = none :=
                chooseNaN2_none_of_not_isNaN (x := x) (y := y) hxNaN hyNaN
              cases hsx : signBit x with
                | true =>
                    -- `x = -Inf`, so `maximum x y = y`.
                    have hcmp : compare x y = some .lt := by
                      simp [compare, hxNaN, hyNaN, hxInf, hyInf, hsx]
                    have hmaxEq : maximum x y = y := by simp [maximum, hchoose, hcmp]
                    simp [hmaxEq, IEEE32Exec.toReal?, hy, hxInf]
                | false =>
                    -- `x = +Inf`, so `maximum x y = x`.
                    have hcmp : compare x y = some .gt := by
                      simp [compare, hxNaN, hyNaN, hxInf, hyInf, hsx]
                    have hmaxEq : maximum x y = x := by simp [maximum, hchoose, hcmp]
                    simp [hmaxEq, IEEE32Exec.toReal?, hy, hxInf]
      | none =>
          -- Both special.
          cases hxNaN : isNaN x with
          | true =>
              cases hchoose : chooseNaN2 x y with
              | some nan =>
                  have hmaxEq : maximum x y = nan := maximum_eq_of_chooseNaN2_some (x := x) (y := y)
                    (nan := nan) hchoose
                  have hfinFalse : isFinite nan = false :=
                    isFinite_eq_false_of_chooseNaN2_some (x := x) (y := y) (nan := nan) hchoose
                  have hnone : toReal? nan = none :=
                    toReal?_eq_none_of_isFinite_eq_false (x := nan) hfinFalse
                  simpa [hmaxEq, hnone, IEEE32Exec.toReal?, hx, hy]
              | none =>
                  have : False := by
                    cases hxS : isSNaN x <;> cases hyS : isSNaN y <;>
                      simp [chooseNaN2, hxS, hyS, hxNaN] at hchoose
                  exact this.elim
          | false =>
              cases hyNaN : isNaN y with
              | true =>
                  cases hchoose : chooseNaN2 x y with
                  | some nan =>
                      have hmaxEq : maximum x y = nan := maximum_eq_of_chooseNaN2_some (x := x) (y
                        := y) (nan := nan) hchoose
                      have hfinFalse : isFinite nan = false :=
                        isFinite_eq_false_of_chooseNaN2_some (x := x) (y := y) (nan := nan) hchoose
                      have hnone : toReal? nan = none :=
                        toReal?_eq_none_of_isFinite_eq_false (x := nan) hfinFalse
                      simpa [hmaxEq, hnone, IEEE32Exec.toReal?, hx, hy]
                  | none =>
                      have : False := by
                        cases hxS : isSNaN x <;> cases hyS : isSNaN y <;>
                          simp [chooseNaN2, hxS, hyS, hyNaN, hxNaN] at hchoose
                      exact this.elim
              | false =>
                  -- Both Infs.
                  have hxInf : isInf x = true :=
                    isInf_eq_true_of_toDyadic?_eq_none_of_isNaN_eq_false (x := x) hx hxNaN
                  have hyInf : isInf y = true :=
                    isInf_eq_true_of_toDyadic?_eq_none_of_isNaN_eq_false (x := y) hy hyNaN
                  have hchoose : chooseNaN2 x y = none :=
                    chooseNaN2_none_of_not_isNaN (x := x) (y := y) hxNaN hyNaN
                  have hx0 : isZero x = false := isZero_eq_false_of_isInf (x := x) hxInf
                  have hy0 : isZero y = false := isZero_eq_false_of_isInf (x := y) hyInf
                  cases hs : (signBit x == signBit y) with
                  | true =>
                      have hcmp : compare x y = some .eq := by
                        simp [compare, hxNaN, hyNaN, hxInf, hyInf, hs]
                      have hmaxEq : maximum x y = x := by
                        simp [maximum, hchoose, hcmp, hx0, hy0]
                      simp [hmaxEq, IEEE32Exec.toReal?, hx, hy]
                  | false =>
                      cases hsx : signBit x with
                      | true =>
                          cases hsy : signBit y with
                          | true =>
                              have : (signBit x == signBit y) = true := by simp [hsx, hsy]
                              have : False := by
                                simp [this]  at hs
                              exact this.elim
                          | false =>
                              have hcmp : compare x y = some .lt := by
                                simp [compare, hxNaN, hyNaN, hxInf, hyInf, hsx, hsy]
                              have hmaxEq : maximum x y = y := by simp [maximum, hchoose, hcmp]
                              simp [hmaxEq, IEEE32Exec.toReal?, hx, hy]
                      | false =>
                          cases hsy : signBit y with
                          | false =>
                              have : (signBit x == signBit y) = true := by simp [hsx, hsy]
                              have : False := by
                                simp [this]  at hs
                              exact this.elim
                          | true =>
                              have hcmp : compare x y = some .gt := by
                                simp [compare, hxNaN, hyNaN, hxInf, hyInf, hsx, hsy]
                              have hmaxEq : maximum x y = x := by simp [maximum, hchoose, hcmp]
                              simp [hmaxEq, IEEE32Exec.toReal?, hx, hy]

end

end IEEE32Exec

end TorchLean.Floats.IEEE754
