/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Bridge.FP32Total.Arithmetic
public import NN.Floats.IEEEExec.Bridge.FP32Total.Order

/-!
# Total FP32 Bridge: Effective Finite Results

The arithmetic bridge states refinement under a finiteness hypothesis. This module packages the
same facts as executable implications: when `toReal?` returns `some r`, the theorem identifies `r`
with the corresponding `FP32` rounded-real operation. It also derives useful finite-result facts
for operations whose IEEE rules rule out exceptional results.

See `FP32Total.Core` for the total bridge convention and bibliography.
-/

@[expose] public section


namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

noncomputable section

/-! ## Effective finite results -/

/-- `fp32Round` agrees with the canonical output of the effective nearest-even calculation. -/
theorem fp32Round_eq_computed (z : ℝ) :
    fp32Round z =
      neuralToReal (β := binaryRadix) {
        mantissa := neuralNearestEvenMantissa
          (neuralScaledMantissa binaryRadix fexp32 z)
        exponent := neuralCexp binaryRadix fexp32 z } := by
  change neuralRound (β := binaryRadix) (fexp := fexp32) rnd32 z = _
  exact FP32.round_eq_computed z

/-- Effective representation of finite executable subtraction. -/
theorem toReal_sub_eq_computed_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true)
    (hfin : isFinite (sub x y) = true) :
    toReal (sub x y) =
      neuralToReal (β := binaryRadix) {
        mantissa := neuralNearestEvenMantissa
          (neuralScaledMantissa binaryRadix fexp32 (toReal x - toReal y))
        exponent := neuralCexp binaryRadix fexp32 (toReal x - toReal y) } := by
  rw [toReal_sub_eq_fp32Round_of_isFinite x y hx hy hfin]
  exact fp32Round_eq_computed (toReal x - toReal y)

/-- Effective representation of finite executable multiplication. -/
theorem toReal_mul_eq_computed_of_isFinite (x y : IEEE32Exec)
    (hfin : isFinite (mul x y) = true) :
    toReal (mul x y) =
      neuralToReal (β := binaryRadix) {
        mantissa := neuralNearestEvenMantissa
          (neuralScaledMantissa binaryRadix fexp32 (toReal x * toReal y))
        exponent := neuralCexp binaryRadix fexp32 (toReal x * toReal y) } := by
  rw [toReal_mul_eq_fp32Round_of_isFinite x y hfin]
  exact fp32Round_eq_computed (toReal x * toReal y)

/-- Effective representation of finite executable division. -/
theorem toReal_div_eq_computed_of_isFinite (x y : IEEE32Exec)
    (hfin : isFinite (div x y) = true) :
    toReal (div x y) =
      neuralToReal (β := binaryRadix) {
        mantissa := neuralNearestEvenMantissa
          (neuralScaledMantissa binaryRadix fexp32 (toReal x / toReal y))
        exponent := neuralCexp binaryRadix fexp32 (toReal x / toReal y) } := by
  rw [toReal_div_eq_fp32Round_of_isFinite x y hfin]
  exact fp32Round_eq_computed (toReal x / toReal y)

/-! ## “Both” view: `toReal?` semantics as an `ite` -/

/-- `toReal? (add x y)` as an `ite` over finiteness. -/
theorem toReal?_add_eq_ite (x y : IEEE32Exec) :
    toReal? (add x y) =
      if isFinite (add x y) then some (fp32Round (toReal x + toReal y)) else none := by
  cases hfin : isFinite (add x y) with
  | true =>
      have hto : toReal (add x y) = fp32Round (toReal x + toReal y) :=
        toReal_add_eq_fp32Round_of_isFinite (x := x) (y := y) (by simpa using hfin)
      have hSome : toReal? (add x y) = some (toReal (add x y)) :=
        toReal?_eq_some_toReal_of_isFinite_eq_true (x := add x y) (by simpa using hfin)
      rw [hSome]
      rw [hto]
      simp
  | false =>
      have hNone : toReal? (add x y) = none :=
        toReal?_eq_none_of_isFinite_eq_false (x := add x y) hfin
      rw [hNone]
      simp

/-- `toReal? (mul x y)` as an `ite` over finiteness. -/
theorem toReal?_mul_eq_ite (x y : IEEE32Exec) :
    toReal? (mul x y) =
      if isFinite (mul x y) then some (fp32Round (toReal x * toReal y)) else none := by
  cases hfin : isFinite (mul x y) with
  | true =>
      have hto : toReal (mul x y) = fp32Round (toReal x * toReal y) :=
        toReal_mul_eq_fp32Round_of_isFinite (x := x) (y := y) (by simpa using hfin)
      have hSome : toReal? (mul x y) = some (toReal (mul x y)) :=
        toReal?_eq_some_toReal_of_isFinite_eq_true (x := mul x y) (by simpa using hfin)
      rw [hSome]
      rw [hto]
      simp
  | false =>
      have hNone : toReal? (mul x y) = none :=
        toReal?_eq_none_of_isFinite_eq_false (x := mul x y) hfin
      rw [hNone]
      simp

/-- `toReal? (fma x y z)` as an `ite` over finiteness. -/
theorem toReal?_fma_eq_ite (x y z : IEEE32Exec) :
    toReal? (fma x y z) =
      if isFinite (fma x y z) then some (fp32Round (toReal x * toReal y + toReal z)) else none := by
  cases hfin : isFinite (fma x y z) with
  | true =>
      have hto : toReal (fma x y z) = fp32Round (toReal x * toReal y + toReal z) :=
        toReal_fma_eq_fp32Round_of_isFinite (x := x) (y := y) (z := z) (by simpa using hfin)
      have hSome : toReal? (fma x y z) = some (toReal (fma x y z)) :=
        toReal?_eq_some_toReal_of_isFinite_eq_true (x := fma x y z) (by simpa using hfin)
      rw [hSome]
      rw [hto]
      simp
  | false =>
      have hNone : toReal? (fma x y z) = none :=
        toReal?_eq_none_of_isFinite_eq_false (x := fma x y z) hfin
      rw [hNone]
      simp

/-- `toReal? (sqrt x)` as an `ite` over finiteness. -/
theorem toReal?_sqrt_eq_ite (x : IEEE32Exec) :
    toReal? (sqrt x) =
      if isFinite (sqrt x) then some (fp32Round (Real.sqrt (toReal x))) else none := by
  cases hfin : isFinite (sqrt x) with
  | true =>
      have hto : toReal (sqrt x) = fp32Round (Real.sqrt (toReal x)) :=
        toReal_sqrt_eq_fp32Round_of_isFinite (x := x) (by simpa using hfin)
      have hSome : toReal? (sqrt x) = some (toReal (sqrt x)) :=
        toReal?_eq_some_toReal_of_isFinite_eq_true (x := sqrt x) (by simpa using hfin)
      rw [hSome]
      rw [hto]
      simp
  | false =>
      have hNone : toReal? (sqrt x) = none :=
        toReal?_eq_none_of_isFinite_eq_false (x := sqrt x) hfin
      rw [hNone]
      simp

/-- `toReal? (div x y)` as an `ite` over finiteness. -/
theorem toReal?_div_eq_ite (x y : IEEE32Exec) :
    toReal? (div x y) =
      if isFinite (div x y) then some (fp32Round (toReal x / toReal y)) else none := by
  cases hfin : isFinite (div x y) with
  | true =>
      have hto : toReal (div x y) = fp32Round (toReal x / toReal y) :=
        toReal_div_eq_fp32Round_of_isFinite (x := x) (y := y) (by simpa using hfin)
      have hSome : toReal? (div x y) = some (toReal (div x y)) :=
        toReal?_eq_some_toReal_of_isFinite_eq_true (x := div x y) (by simpa using hfin)
      rw [hSome]
      rw [hto]
      simp
  | false =>
      have hNone : toReal? (div x y) = none :=
        toReal?_eq_none_of_isFinite_eq_false (x := div x y) hfin
      rw [hNone]
      simp

/--
`minimum` of two finite values is finite.
-/
theorem isFinite_minimum_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    isFinite (minimum x y) = true := by
  classical
  cases hdx : toDyadic? x with
  | none =>
      have : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      have : False := by
        simp [hx]  at this
      exact this.elim
  | some dx =>
      cases hdy : toDyadic? y with
      | none =>
          have : isFinite y = false := isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
          have : False := by
            simp [hy]  at this
          exact this.elim
      | some dy =>
          have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hdx)
          have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hdy)
          have hchoose : chooseNaN2 x y = none := by
            simp [chooseNaN2, isSNaN, hxNaN, hyNaN]
          have hcmp : compare x y = some (cmpDyadic dx dy) :=
            compare_eq_some_cmpDyadic_of_toDyadic? (x := x) (y := y) (hx := hdx) (hy := hdy)
          cases hord : cmpDyadic dx dy with
          | lt =>
              have hcmp' : compare x y = some .lt := by simpa [hord] using hcmp
              have hmin : minimum x y = x := by simp [minimum, hchoose, hcmp']
              simpa [hmin] using hx
          | gt =>
              have hcmp' : compare x y = some .gt := by simpa [hord] using hcmp
              have hmin : minimum x y = y := by simp [minimum, hchoose, hcmp']
              simpa [hmin] using hy
          | eq =>
              have hcmp' : compare x y = some .eq := by simpa [hord] using hcmp
              cases hzeros : (isZero x && isZero y) with
              | true =>
                  have hmin : minimum x y = (if signBit x || signBit y then negZero else posZero) :=
                    by
                    simp [minimum, hchoose, hcmp', hzeros]
                  cases hs : (signBit x || signBit y) <;> simp [hmin, hs] <;> decide
              | false =>
                  have hmin : minimum x y = x := by simp [minimum, hchoose, hcmp', hzeros]
                  simpa [hmin] using hx

/--
On finite inputs, `toReal? (minimum x y)` returns `some (min (toReal x) (toReal y))`.
-/
theorem toReal?_minimum_eq_min_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    toReal? (minimum x y) = some (min (toReal x) (toReal y)) := by
  have hfin : isFinite (minimum x y) = true := isFinite_minimum_of_isFinite (x := x) (y := y) hx hy
  have hSome : toReal? (minimum x y) = some (toReal (minimum x y)) :=
    toReal?_eq_some_toReal_of_isFinite_eq_true (x := minimum x y) hfin
  have hto : toReal (minimum x y) = min (toReal x) (toReal y) :=
    toReal_minimum_eq_min_of_isFinite (x := x) (y := y) hx hy
  rw [hSome, hto]

/--
`maximum` of two finite values is finite.
-/
theorem isFinite_maximum_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    isFinite (maximum x y) = true := by
  classical
  cases hdx : toDyadic? x with
  | none =>
      have : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      have : False := by
        simp [hx]  at this
      exact this.elim
  | some dx =>
      cases hdy : toDyadic? y with
      | none =>
          have : isFinite y = false := isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
          have : False := by
            simp [hy]  at this
          exact this.elim
      | some dy =>
          have hxNaN : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hdx)
          have hyNaN : isNaN y = false := isNaN_eq_false_of_toDyadic?_some (hx := hdy)
          have hchoose : chooseNaN2 x y = none := by
            simp [chooseNaN2, isSNaN, hxNaN, hyNaN]
          have hcmp : compare x y = some (cmpDyadic dx dy) :=
            compare_eq_some_cmpDyadic_of_toDyadic? (x := x) (y := y) (hx := hdx) (hy := hdy)
          cases hord : cmpDyadic dx dy with
          | lt =>
              have hcmp' : compare x y = some .lt := by simpa [hord] using hcmp
              have hmax : maximum x y = y := by simp [maximum, hchoose, hcmp']
              simpa [hmax] using hy
          | gt =>
              have hcmp' : compare x y = some .gt := by simpa [hord] using hcmp
              have hmax : maximum x y = x := by simp [maximum, hchoose, hcmp']
              simpa [hmax] using hx
          | eq =>
              have hcmp' : compare x y = some .eq := by simpa [hord] using hcmp
              cases hzeros : (isZero x && isZero y) with
              | true =>
                  have hmax : maximum x y =
                      (if (!signBit x) || (!signBit y) then posZero else negZero) := by
                    simp [maximum, hchoose, hcmp', hzeros]
                  cases hs : ((!signBit x) || (!signBit y)) <;> simp [hmax, hs] <;> decide
              | false =>
                  have hmax : maximum x y = x := by simp [maximum, hchoose, hcmp', hzeros]
                  simpa [hmax] using hx

/--
On finite inputs, `toReal? (maximum x y)` returns `some (max (toReal x) (toReal y))`.
-/
theorem toReal?_maximum_eq_max_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    toReal? (maximum x y) = some (max (toReal x) (toReal y)) := by
  have hfin : isFinite (maximum x y) = true := isFinite_maximum_of_isFinite (x := x) (y := y) hx hy
  have hSome : toReal? (maximum x y) = some (toReal (maximum x y)) :=
    toReal?_eq_some_toReal_of_isFinite_eq_true (x := maximum x y) hfin
  have hto : toReal (maximum x y) = max (toReal x) (toReal y) :=
    toReal_maximum_eq_max_of_isFinite (x := x) (y := y) hx hy
  rw [hSome, hto]

end

end IEEE32Exec

end TorchLean.Floats.IEEE754
