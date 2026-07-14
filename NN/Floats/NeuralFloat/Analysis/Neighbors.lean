/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Analysis.Ulp

/-!
# Successor and Predecessor

The neighboring values follow Flocq's `Core/Ulp.v`.  At a positive radix boundary, the spacing
below the value can differ from the spacing above it, so `neuralPredPos` uses the preceding
magnitude's exponent in that case.
-/

@[expose] public section

namespace TorchLean.Floats

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]

/-- Previous-value formula for a nonnegative input. -/
noncomputable def neuralPredPos (x : ℝ) : ℝ :=
  if x = neuralBpow β (neuralMagnitude β x - 1) then
    x - neuralBpow β (fexp (neuralMagnitude β x - 1))
  else
    x - neuralUlp β fexp x

/-- Successor in the generic format, defined by positive spacing and sign symmetry. -/
noncomputable def neuralSucc (x : ℝ) : ℝ :=
  if 0 ≤ x then x + neuralUlp β fexp x else -neuralPredPos (β := β) (fexp := fexp) (-x)

/-- Predecessor, defined as the negated successor of the negated input. -/
noncomputable def neuralPred (x : ℝ) : ℝ :=
  -neuralSucc (β := β) (fexp := fexp) (-x)

/-- On nonnegative inputs, successor adds one ULP. -/
theorem neuralSucc_eq_of_nonneg {x : ℝ} (hx : 0 ≤ x) :
    neuralSucc (β := β) (fexp := fexp) x = x + neuralUlp β fexp x := by
  simp [neuralSucc, hx]

/-- Successor and predecessor are exchanged by negation. -/
@[simp] theorem neuralSucc_neg (x : ℝ) :
    neuralSucc (β := β) (fexp := fexp) (-x) = -neuralPred (β := β) (fexp := fexp) x := by
  simp [neuralPred]

/-- Predecessor and successor are exchanged by negation. -/
@[simp] theorem neuralPred_neg (x : ℝ) :
    neuralPred (β := β) (fexp := fexp) (-x) = -neuralSucc (β := β) (fexp := fexp) x := by
  simp [neuralPred]

/-- The successor of zero is the format's zero ULP. -/
theorem neuralSucc_zero :
    neuralSucc (β := β) (fexp := fexp) 0 = neuralUlp β fexp 0 := by
  simp [neuralSucc]

/-- The predecessor of zero is the negated zero ULP. -/
theorem neuralPred_zero :
    neuralPred (β := β) (fexp := fexp) 0 = -neuralUlp β fexp 0 := by
  rw [neuralPred, neg_zero, neuralSucc_zero]

/-- On nonnegative inputs, the symmetric predecessor agrees with `neuralPredPos`. -/
theorem neuralPred_eq_pos {x : ℝ} (hx : 0 ≤ x) :
    neuralPred (β := β) (fexp := fexp) x = neuralPredPos (β := β) (fexp := fexp) x := by
  by_cases hx0 : x = 0
  · subst x
    rw [neuralPred_zero]
    have hb : (0 : ℝ) ≠ neuralBpow β (-1) := (neuralBpow.ne_zero β (-1)).symm
    simp [neuralPredPos, neuralMagnitude, hb]
  · have hxpos : 0 < x := lt_of_le_of_ne hx (Ne.symm hx0)
    have hnx : ¬0 ≤ -x := not_le.mpr (neg_neg_of_pos hxpos)
    simp [neuralPred, neuralSucc, hnx]

/-- The predecessor of a radix power uses the spacing from the bin immediately below it. -/
theorem neuralPred_bpow (e : ℤ) :
    neuralPred (β := β) (fexp := fexp) (neuralBpow β e) =
      neuralBpow β e - neuralBpow β (fexp e) := by
  rw [neuralPred_eq_pos (neuralBpow.nonneg β e)]
  simp [neuralPredPos]

/-- The positive predecessor formula never exceeds its input. -/
theorem neuralPredPos_le (x : ℝ) : neuralPredPos (β := β) (fexp := fexp) x ≤ x := by
  unfold neuralPredPos
  split
  · exact sub_le_self x (neuralBpow.nonneg β _)
  · exact sub_le_self x (neuralUlp.nonneg β fexp x)

/-- Away from zero, the positive predecessor formula is strictly smaller than its input. -/
theorem neuralPredPos_lt {x : ℝ} (hx : x ≠ 0) :
    neuralPredPos (β := β) (fexp := fexp) x < x := by
  unfold neuralPredPos
  split
  · exact sub_lt_self x (neuralBpow.pos β _)
  · exact sub_lt_self x (neuralUlp.pos_of_ne_zero β fexp x hx)

/-- Successor never falls below its input. -/
theorem le_neuralSucc (x : ℝ) : x ≤ neuralSucc (β := β) (fexp := fexp) x := by
  by_cases hx : 0 ≤ x
  · rw [neuralSucc_eq_of_nonneg hx]
    exact le_add_of_nonneg_right (neuralUlp.nonneg β fexp x)
  · simp only [neuralSucc, hx, if_false]
    simpa only [neg_neg] using
      neg_le_neg (neuralPredPos_le (β := β) (fexp := fexp) (-x))

/-- Predecessor never exceeds its input. -/
theorem neuralPred_le (x : ℝ) : neuralPred (β := β) (fexp := fexp) x ≤ x := by
  rw [← neg_le_neg_iff]
  simpa using le_neuralSucc (β := β) (fexp := fexp) (-x)

/-- Successor is strictly larger away from zero. -/
theorem lt_neuralSucc {x : ℝ} (hx : x ≠ 0) :
    x < neuralSucc (β := β) (fexp := fexp) x := by
  by_cases hnonneg : 0 ≤ x
  · rw [neuralSucc_eq_of_nonneg hnonneg]
    exact lt_add_of_pos_right x (neuralUlp.pos_of_ne_zero β fexp x hx)
  · have hnegne : -x ≠ 0 := neg_ne_zero.mpr hx
    simp only [neuralSucc, hnonneg, if_false]
    simpa only [neg_neg] using
      neg_lt_neg (neuralPredPos_lt (β := β) (fexp := fexp) hnegne)

/-- Predecessor is strictly smaller away from zero. -/
theorem neuralPred_lt {x : ℝ} (hx : x ≠ 0) :
    neuralPred (β := β) (fexp := fexp) x < x := by
  rw [← neg_lt_neg_iff]
  simpa using lt_neuralSucc (β := β) (fexp := fexp) (neg_ne_zero.mpr hx)

/-- The successor of a positive representable value is representable. -/
theorem neural_generic_format_succ_of_pos {x : ℝ} (hx : 0 < x)
    (hfmt : neuralGenericFormat β fexp x) :
    neuralGenericFormat β fexp (neuralSucc (β := β) (fexp := fexp) x) := by
  let ex := neuralMagnitude β x
  let e := fexp ex
  let s := neuralScaledMantissa β fexp x
  obtain ⟨n, hn⟩ := neural_scaled_mantissa_int_of_generic
    (β := β) (fexp := fexp) x hfmt
  have hx0 : x ≠ 0 := hx.ne'
  have hbpos : 0 < neuralBpow β e := neuralBpow.pos β e
  have hcexp : neuralCexp β fexp x = e := rfl
  have hsdiv : s = x / neuralBpow β e := by
    simpa [s, hcexp] using neuralScaledMantissa_eq_div (β := β) (fexp := fexp) x
  have hlarge : e < ex := by
    simpa [e, ex, neuralCexp] using
      neuralCexp_lt_magnitude_of_pos_generic (β := β) (fexp := fexp) hx hfmt
  have hulp : neuralUlp β fexp x = neuralBpow β e := by
    rw [neuralUlp.of_ne_zero β fexp x hx0]
    rfl
  have hd : 0 ≤ ex - e := by linarith
  obtain ⟨N, hN⟩ := neuralBpow_eq_natCast_of_nonneg β (ex - e) hd
  have hsUpper : s < neuralBpow β (ex - e) := by
    rw [hsdiv]
    calc
      x / neuralBpow β e < neuralBpow β ex / neuralBpow β e :=
        (div_lt_div_iff_of_pos_right hbpos).2
          (by simpa [ex, abs_of_pos hx] using (neuralMagnitude_spec β x hx0).2)
      _ = neuralBpow β (ex - e) := (neuralBpow.sub_exp β ex e).symm
  have hnN : n < (N : ℤ) := by
    exact_mod_cast (show (n : ℝ) < N by simpa [s, hn, hN] using hsUpper)
  have hnSucc : n + 1 ≤ (N : ℤ) := Int.add_one_le_iff.mpr hnN
  let y := neuralSucc (β := β) (fexp := fexp) x
  change neuralGenericFormat β fexp y
  have hy : y = ((n + 1 : ℤ) : ℝ) * neuralBpow β e := by
    dsimp [y]
    rw [neuralSucc_eq_of_nonneg hx.le, hulp]
    have hrepr := neural_scaled_mantissa_mul_bpow (β := β) (fexp := fexp) x
    rw [hn, hcexp] at hrepr
    rw [← hrepr]
    push_cast
    ring
  have hyUpper : y ≤ neuralBpow β ex := by
    have hnSuccR : ((n + 1 : ℤ) : ℝ) ≤ N := by exact_mod_cast hnSucc
    calc
      y = ((n + 1 : ℤ) : ℝ) * neuralBpow β e := hy
      _ ≤ (N : ℝ) * neuralBpow β e :=
        mul_le_mul_of_nonneg_right hnSuccR hbpos.le
      _ = neuralBpow β (ex - e) * neuralBpow β e := by rw [hN]
      _ = neuralBpow β ex := by
        rw [← neuralBpow.add_exp]
        congr 1
        linarith
  rcases hyUpper.eq_or_lt with heq | hlt
  · rw [heq]
    exact neural_generic_format_bpow ex
      (((NeuralValidExp.flocq_valid (fexp := fexp) ex).1 hlarge))
  · have hxy : x ≤ y := by
      simpa [y] using le_neuralSucc (β := β) (fexp := fexp) x
    have hypos : 0 < y := hx.trans_le hxy
    have hyLower : neuralBpow β (ex - 1) ≤ y := by
      have hxLower : neuralBpow β (ex - 1) ≤ x := by
        simpa [ex, abs_of_pos hx] using (neuralMagnitude_spec β x hx0).1
      exact hxLower.trans hxy
    have hmagY : neuralMagnitude β y = ex :=
      neuralMagnitude_eq_of_bpow_bounds β y ex hypos.ne'
        (by simpa [abs_of_pos hypos] using hyLower)
        (by simpa [abs_of_pos hypos] using hlt)
    apply neural_generic_format_of_toReal_of_cexp_le
      ({ mantissa := n + 1, exponent := e } : NeuralFloat β) y
    · simpa [neuralToReal] using hy
    · simp [neuralCexp, hmagY, e]

/--
Subtracting one ULP from a positive representable value that is not a radix boundary remains
representable in the same magnitude bin.
-/
theorem neural_generic_format_sub_ulp_of_pos {x : ℝ} (hx : 0 < x)
    (hfmt : neuralGenericFormat β fexp x)
    (hboundary : x ≠ neuralBpow β (neuralMagnitude β x - 1)) :
    neuralGenericFormat β fexp (x - neuralUlp β fexp x) := by
  let ex := neuralMagnitude β x
  let e := fexp ex
  let s := neuralScaledMantissa β fexp x
  obtain ⟨n, hn⟩ := neural_scaled_mantissa_int_of_generic
    (β := β) (fexp := fexp) x hfmt
  have hx0 : x ≠ 0 := hx.ne'
  have hbpos : 0 < neuralBpow β e := neuralBpow.pos β e
  have hcexp : neuralCexp β fexp x = e := rfl
  have hsdiv : s = x / neuralBpow β e := by
    simpa [s, hcexp] using neuralScaledMantissa_eq_div (β := β) (fexp := fexp) x
  have hlarge : e < ex := by
    simpa [e, ex, neuralCexp] using
      neuralCexp_lt_magnitude_of_pos_generic (β := β) (fexp := fexp) hx hfmt
  have hulp : neuralUlp β fexp x = neuralBpow β e := by
    rw [neuralUlp.of_ne_zero β fexp x hx0]
    rfl
  have hd : 0 ≤ ex - 1 - e := by linarith
  obtain ⟨N, hN⟩ := neuralBpow_eq_natCast_of_nonneg β (ex - 1 - e) hd
  have hxLower : neuralBpow β (ex - 1) < x := by
    have hle : neuralBpow β (ex - 1) ≤ x := by
      simpa [ex, abs_of_pos hx] using (neuralMagnitude_spec β x hx0).1
    exact lt_of_le_of_ne hle (Ne.symm (by simpa [ex] using hboundary))
  have hsLower : neuralBpow β (ex - 1 - e) < s := by
    rw [hsdiv]
    calc
      neuralBpow β (ex - 1 - e) =
          neuralBpow β (ex - 1) / neuralBpow β e := neuralBpow.sub_exp β _ _
      _ < x / neuralBpow β e := (div_lt_div_iff_of_pos_right hbpos).2 hxLower
  have hNn : (N : ℤ) < n := by
    exact_mod_cast (show (N : ℝ) < n by simpa [s, hn, hN] using hsLower)
  have hNpred : (N : ℤ) ≤ n - 1 := by linarith
  let y := x - neuralUlp β fexp x
  change neuralGenericFormat β fexp y
  have hy : y = ((n - 1 : ℤ) : ℝ) * neuralBpow β e := by
    dsimp [y]
    rw [hulp]
    have hrepr := neural_scaled_mantissa_mul_bpow (β := β) (fexp := fexp) x
    rw [hn, hcexp] at hrepr
    rw [← hrepr]
    push_cast
    ring
  have hyLower : neuralBpow β (ex - 1) ≤ y := by
    have hNpredR : (N : ℝ) ≤ (n - 1 : ℤ) := by exact_mod_cast hNpred
    calc
      neuralBpow β (ex - 1) =
          neuralBpow β (ex - 1 - e) * neuralBpow β e := by
        rw [← neuralBpow.add_exp]
        congr 1
        linarith
      _ = (N : ℝ) * neuralBpow β e := by rw [hN]
      _ ≤ ((n - 1 : ℤ) : ℝ) * neuralBpow β e :=
        mul_le_mul_of_nonneg_right hNpredR hbpos.le
      _ = y := hy.symm
  have hyUpper : y < neuralBpow β ex := by
    have hyltx : y < x := by
      dsimp [y]
      exact sub_lt_self x (neuralUlp.pos_of_ne_zero β fexp x hx0)
    exact hyltx.trans (by simpa [ex, abs_of_pos hx] using (neuralMagnitude_spec β x hx0).2)
  have hypos : 0 < y := (neuralBpow.pos β (ex - 1)).trans_le hyLower
  have hmagY : neuralMagnitude β y = ex :=
    neuralMagnitude_eq_of_bpow_bounds β y ex hypos.ne'
      (by simpa [abs_of_pos hypos] using hyLower)
      (by simpa [abs_of_pos hypos] using hyUpper)
  apply neural_generic_format_of_toReal_of_cexp_le
    ({ mantissa := n - 1, exponent := e } : NeuralFloat β) y
  · simpa [neuralToReal] using hy
  · simp [neuralCexp, hmagY, e]

/-- Subtracting the preceding-bin spacing from a representable radix power is representable. -/
theorem neural_generic_format_bpow_sub_prev_spacing (k : ℤ)
    (hfmt : neuralGenericFormat β fexp (neuralBpow β k)) :
    neuralGenericFormat β fexp (neuralBpow β k - neuralBpow β (fexp k)) := by
  let ef := fexp k
  have hef : ef ≤ k := by
    simpa [ef] using neural_fexp_le_of_generic_bpow (β := β) (fexp := fexp) k hfmt
  rcases hef.eq_or_lt with heq | hlt
  · have hy0 : neuralBpow β k - neuralBpow β ef = 0 := by rw [heq]; ring
    rw [hy0]
    exact neural_generic_format_zero
  · let d := k - ef
    have hd1 : 1 ≤ d := by simp [d]; linarith
    have hd0 : 0 ≤ d := le_trans (by norm_num) hd1
    have hdsub : 0 ≤ d - 1 := by linarith
    have hpowOne : (1 : ℝ) ≤ neuralBpow β (d - 1) := by
      change (1 : ℝ) ≤ β.toReal ^ (d - 1)
      exact one_le_zpow₀ (le_of_lt (NeuralRadix.gt_one β)) hdsub
    have hbaseTwo : (2 : ℝ) ≤ β.toReal := by
      change (2 : ℝ) ≤ (β.base : ℝ)
      exact_mod_cast β.base_valid
    have hstep : neuralBpow β d = β.toReal * neuralBpow β (d - 1) := by
      calc
        neuralBpow β d = neuralBpow β (1 + (d - 1)) := by congr 1; linarith
        _ = neuralBpow β 1 * neuralBpow β (d - 1) := neuralBpow.add_exp β _ _
        _ = β.toReal * neuralBpow β (d - 1) := by simp [neuralBpow]
    have hdouble : 2 * neuralBpow β (d - 1) ≤
        β.toReal * neuralBpow β (d - 1) :=
      mul_le_mul_of_nonneg_right hbaseTwo (neuralBpow.nonneg β _)
    have hgap : neuralBpow β (d - 1) ≤ neuralBpow β d - 1 := by
      rw [hstep]
      linarith
    let y := neuralBpow β k - neuralBpow β ef
    change neuralGenericFormat β fexp y
    have hypos : 0 < y := by
      dsimp [y]
      exact sub_pos.mpr ((neuralBpow_lt_neuralBpow_iff β ef k).2 hlt)
    have hyUpper : y < neuralBpow β k := by
      dsimp [y]
      exact sub_lt_self _ (neuralBpow.pos β ef)
    have hyLower : neuralBpow β (k - 1) ≤ y := by
      have hb := neuralBpow.nonneg β ef
      calc
        neuralBpow β (k - 1) =
            neuralBpow β (d - 1) * neuralBpow β ef := by
          rw [← neuralBpow.add_exp]
          congr 1
          dsimp [d]
          ring
        _ ≤ (neuralBpow β d - 1) * neuralBpow β ef :=
          mul_le_mul_of_nonneg_right hgap hb
        _ = y := by
          dsimp [y]
          rw [sub_mul, one_mul, ← neuralBpow.add_exp]
          congr 1
          simp [d]
    have hmagY : neuralMagnitude β y = k :=
      neuralMagnitude_eq_of_bpow_bounds β y k hypos.ne'
        (by simpa [abs_of_pos hypos] using hyLower)
        (by simpa [abs_of_pos hypos] using hyUpper)
    obtain ⟨N, hN⟩ := neuralBpow_eq_natCast_of_nonneg β d hd0
    apply neural_generic_format_of_toReal_of_cexp_le
      ({ mantissa := Int.ofNat N - 1, exponent := ef } : NeuralFloat β) y
    · dsimp [y]
      simp only [neuralToReal]
      calc
        neuralBpow β k - neuralBpow β ef =
            neuralBpow β d * neuralBpow β ef - 1 * neuralBpow β ef := by
          rw [← neuralBpow.add_exp]
          congr 2
          · dsimp [d]
            ring
          · ring
        _ = (neuralBpow β d - 1) * neuralBpow β ef := by ring
        _ = ((N : ℝ) - 1) * neuralBpow β ef := by rw [hN]
        _ = ((Int.ofNat N - 1 : ℤ) : ℝ) * neuralBpow β ef := by norm_num
    · simp [neuralCexp, hmagY, ef]

/-- The positive predecessor of a positive representable value is representable. -/
theorem neural_generic_format_predPos_of_pos {x : ℝ} (hx : 0 < x)
    (hfmt : neuralGenericFormat β fexp x) :
    neuralGenericFormat β fexp (neuralPredPos (β := β) (fexp := fexp) x) := by
  unfold neuralPredPos
  split_ifs with hboundary
  · let k := neuralMagnitude β x - 1
    have hpowFmt : neuralGenericFormat β fexp (neuralBpow β k) := by
      rw [← hboundary]
      exact hfmt
    rw [hboundary]
    simpa [k] using neural_generic_format_bpow_sub_prev_spacing
      (β := β) (fexp := fexp) k hpowFmt
  · exact neural_generic_format_sub_ulp_of_pos hx hfmt hboundary

/-- The successor of every representable value is representable. -/
theorem neural_generic_format_succ {x : ℝ} (hfmt : neuralGenericFormat β fexp x) :
    neuralGenericFormat β fexp (neuralSucc (β := β) (fexp := fexp) x) := by
  rcases lt_trichotomy x 0 with hx | hx | hx
  · have hnegpos : 0 < -x := neg_pos.mpr hx
    have hnegfmt : neuralGenericFormat β fexp (-x) := neural_generic_format_neg x hfmt
    have hpred := neural_generic_format_predPos_of_pos
      (β := β) (fexp := fexp) hnegpos hnegfmt
    have hsucc : neuralSucc (β := β) (fexp := fexp) x =
        -neuralPredPos (β := β) (fexp := fexp) (-x) := by
      simp [neuralSucc, not_le.mpr hx]
    rw [hsucc]
    exact neural_generic_format_neg _ hpred
  · subst x
    rw [neuralSucc_zero]
    exact neural_generic_format_ulp_zero
  · exact neural_generic_format_succ_of_pos hx hfmt

/-- The predecessor of every representable value is representable. -/
theorem neural_generic_format_pred {x : ℝ} (hfmt : neuralGenericFormat β fexp x) :
    neuralGenericFormat β fexp (neuralPred (β := β) (fexp := fexp) x) := by
  rw [neuralPred]
  apply neural_generic_format_neg
  exact neural_generic_format_succ (neural_generic_format_neg x hfmt)

/-- A positive representable value's successor does not exceed its magnitude boundary. -/
theorem neuralSucc_le_magnitude_bpow_of_pos {x : ℝ} (hx : 0 < x)
    (hfmt : neuralGenericFormat β fexp x) :
    neuralSucc (β := β) (fexp := fexp) x ≤ neuralBpow β (neuralMagnitude β x) := by
  let ex := neuralMagnitude β x
  let e := fexp ex
  let s := neuralScaledMantissa β fexp x
  obtain ⟨n, hn⟩ := neural_scaled_mantissa_int_of_generic
    (β := β) (fexp := fexp) x hfmt
  have hx0 : x ≠ 0 := hx.ne'
  have hbpos : 0 < neuralBpow β e := neuralBpow.pos β e
  have hcexp : neuralCexp β fexp x = e := rfl
  have hsdiv : s = x / neuralBpow β e := by
    simpa [s, hcexp] using neuralScaledMantissa_eq_div (β := β) (fexp := fexp) x
  have hlarge : e < ex := by
    simpa [e, ex, neuralCexp] using
      neuralCexp_lt_magnitude_of_pos_generic (β := β) (fexp := fexp) hx hfmt
  have hulp : neuralUlp β fexp x = neuralBpow β e := by
    rw [neuralUlp.of_ne_zero β fexp x hx0]
    rfl
  have hd : 0 ≤ ex - e := by linarith
  obtain ⟨N, hN⟩ := neuralBpow_eq_natCast_of_nonneg β (ex - e) hd
  have hsUpper : s < neuralBpow β (ex - e) := by
    rw [hsdiv]
    calc
      x / neuralBpow β e < neuralBpow β ex / neuralBpow β e :=
        (div_lt_div_iff_of_pos_right hbpos).2
          (by simpa [ex, abs_of_pos hx] using (neuralMagnitude_spec β x hx0).2)
      _ = neuralBpow β (ex - e) := (neuralBpow.sub_exp β ex e).symm
  have hnN : n < (N : ℤ) := by
    exact_mod_cast (show (n : ℝ) < N by simpa [s, hn, hN] using hsUpper)
  have hnSuccR : ((n + 1 : ℤ) : ℝ) ≤ N := by
    exact_mod_cast (Int.add_one_le_iff.mpr hnN)
  rw [neuralSucc_eq_of_nonneg hx.le, hulp]
  have hrepr := neural_scaled_mantissa_mul_bpow (β := β) (fexp := fexp) x
  rw [hn, hcexp] at hrepr
  calc
    x + neuralBpow β e = (n : ℝ) * neuralBpow β e + neuralBpow β e := by
      rw [hrepr]
    _ =
        ((n + 1 : ℤ) : ℝ) * neuralBpow β e := by push_cast; ring
    _ ≤ (N : ℝ) * neuralBpow β e :=
      mul_le_mul_of_nonneg_right hnSuccR hbpos.le
    _ = neuralBpow β (ex - e) * neuralBpow β e := by rw [hN]
    _ = neuralBpow β ex := by
      rw [← neuralBpow.add_exp]
      congr 1
      linarith
    _ = neuralBpow β (neuralMagnitude β x) := rfl

/-- For positive representable values, no representable value lies strictly before the successor. -/
theorem neuralSucc_le_of_lt_pos {x y : ℝ} (hx : 0 < x)
    (hxfmt : neuralGenericFormat β fexp x) (hyfmt : neuralGenericFormat β fexp y)
    (hxy : x < y) : neuralSucc (β := β) (fexp := fexp) x ≤ y := by
  have hy : 0 < y := hx.trans hxy
  let ex := neuralMagnitude β x
  let ey := neuralMagnitude β y
  have hmag : ex ≤ ey := by
    have hxLower := (neuralMagnitude_spec β x hx.ne').1
    have hyUpper := (neuralMagnitude_spec β y hy.ne').2
    have hpowers : neuralBpow β (ex - 1) < neuralBpow β ey := by
      exact hxLower.trans (by simpa [abs_of_pos hx, abs_of_pos hy] using hxy.le) |>.trans_lt hyUpper
    have : ex - 1 < ey := (neuralBpow_lt_neuralBpow_iff β _ _).mp hpowers
    linarith
  rcases hmag.eq_or_lt with heq | hlt
  · let e := fexp ex
    obtain ⟨nx, hnx⟩ := neural_scaled_mantissa_int_of_generic
      (β := β) (fexp := fexp) x hxfmt
    obtain ⟨ny, hny⟩ := neural_scaled_mantissa_int_of_generic
      (β := β) (fexp := fexp) y hyfmt
    have hcexpX : neuralCexp β fexp x = e := by simp [neuralCexp, ex, e]
    have hcexpY : neuralCexp β fexp y = e := by simp [neuralCexp, ey, ex, heq, e]
    have hmant : nx < ny := by
      have hbpos := neuralBpow.pos β e
      have hsxy : neuralScaledMantissa β fexp x < neuralScaledMantissa β fexp y := by
        rw [neuralScaledMantissa_eq_div, neuralScaledMantissa_eq_div, hcexpX, hcexpY]
        exact (div_lt_div_iff_of_pos_right hbpos).2 hxy
      exact_mod_cast (show (nx : ℝ) < ny by simpa [hnx, hny] using hsxy)
    have hsuccMant : nx + 1 ≤ ny := Int.add_one_le_iff.mpr hmant
    rw [neuralSucc_eq_of_nonneg hx.le,
      neuralUlp.of_ne_zero β fexp x hx.ne', hcexpX]
    have hxrepr := neural_scaled_mantissa_mul_bpow (β := β) (fexp := fexp) x
    have hyrepr := neural_scaled_mantissa_mul_bpow (β := β) (fexp := fexp) y
    rw [hnx, hcexpX] at hxrepr
    rw [hny, hcexpY] at hyrepr
    rw [← hxrepr, ← hyrepr]
    have hsuccMantR : ((nx + 1 : ℤ) : ℝ) ≤ ny := by exact_mod_cast hsuccMant
    calc
      (nx : ℝ) * neuralBpow β e + neuralBpow β e =
          ((nx + 1 : ℤ) : ℝ) * neuralBpow β e := by push_cast; ring
      _ ≤ (ny : ℝ) * neuralBpow β e :=
        mul_le_mul_of_nonneg_right hsuccMantR (neuralBpow.nonneg β e)
  · have hsuccBound := neuralSucc_le_magnitude_bpow_of_pos
      (β := β) (fexp := fexp) hx hxfmt
    have hboundary : neuralBpow β ex ≤ neuralBpow β (ey - 1) :=
      (neuralBpow_le_neuralBpow_iff β _ _).2 (by linarith)
    have hyLower : neuralBpow β (ey - 1) ≤ y := by
      simpa [ey, abs_of_pos hy] using (neuralMagnitude_spec β y hy.ne').1
    exact hsuccBound.trans (hboundary.trans hyLower)

end TorchLean.Floats
