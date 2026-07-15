/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Bridge.FP32Total.Core

/-!
# Total FP32 Bridge: Ordering

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

/-- On finite values, IEEE-754 `minimum` agrees with real `min`. -/
theorem toReal_minimum_eq_min_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    toReal (minimum x y) = min (toReal x) (toReal y) := by
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
          exact toReal_minimum_eq_min (x := x) (y := y) (dx := dx) (dy := dy) hdx hdy

/-- On finite values, IEEE-754 `maximum` agrees with real `max`. -/
theorem toReal_maximum_eq_max_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    toReal (maximum x y) = max (toReal x) (toReal y) := by
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
          exact toReal_maximum_eq_max (x := x) (y := y) (dx := dx) (dy := dy) hdx hdy

/-- On finite values, `compare x y = some .lt` iff `toReal x < toReal y`. -/
theorem compare_eq_some_lt_iff_toReal_lt_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    compare x y = some .lt ↔ toReal x < toReal y := by
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
          exact compare_eq_some_lt_iff_toReal_lt (x := x) (y := y) (dx := dx) (dy := dy) hdx hdy

/-- On finite values, `compare x y = some .eq` iff `toReal x = toReal y`. -/
theorem compare_eq_some_eq_iff_toReal_eq_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    compare x y = some .eq ↔ toReal x = toReal y := by
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
          exact compare_eq_some_eq_iff_toReal_eq (x := x) (y := y) (dx := dx) (dy := dy) hdx hdy

/-- On finite values, `compare x y = some .gt` iff `toReal y < toReal x`. -/
theorem compare_eq_some_gt_iff_toReal_gt_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    compare x y = some .gt ↔ toReal y < toReal x := by
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
          exact compare_eq_some_gt_iff_toReal_gt (x := x) (y := y) (dx := dx) (dy := dy) hdx hdy

end

end IEEE32Exec

end TorchLean.Floats.IEEE754
