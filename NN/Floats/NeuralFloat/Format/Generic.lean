/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Format.Magnitude
public import NN.Floats.NeuralFloat.Rounding.Core

/-!
# Generic Format Properties

Elementary closure properties of the Flocq-style generic format. These lemmas are independent of
the standard FIX, FLX, and FLT families.
-/

@[expose] public section

namespace TorchLean.Floats

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]

/-- Magnitude is invariant under negation. -/
@[simp] theorem neuralMagnitude_neg (x : ℝ) : neuralMagnitude β (-x) = neuralMagnitude β x := by
  simp [neuralMagnitude]

/-- The canonical exponent is invariant under negation. -/
@[simp] theorem neuralCexp_neg (x : ℝ) : neuralCexp β fexp (-x) = neuralCexp β fexp x := by
  simp [neuralCexp]

/-- Negation negates the canonical scaled mantissa. -/
@[simp] theorem neuralScaledMantissa_neg (x : ℝ) :
    neuralScaledMantissa β fexp (-x) = -neuralScaledMantissa β fexp x := by
  simp [neuralScaledMantissa]

/-- Zero belongs to every valid generic format. -/
@[simp] theorem neural_generic_format_zero : neuralGenericFormat β fexp 0 := by
  apply neural_generic_format_of_scaled_mantissa_int (n := 0)
  simp [neuralScaledMantissa]

/-- Generic formats are closed under negation. -/
theorem neural_generic_format_neg (x : ℝ) (hx : neuralGenericFormat β fexp x) :
    neuralGenericFormat β fexp (-x) := by
  obtain ⟨n, hn⟩ :=
    (neural_generic_format_iff_scaled_mantissa_int (β := β) (fexp := fexp) x).mp hx
  apply neural_generic_format_of_scaled_mantissa_int (n := -n)
  simp [hn]

/-- A value is representable exactly when its negation is representable. -/
@[simp] theorem neural_generic_format_neg_iff (x : ℝ) :
    neuralGenericFormat β fexp (-x) ↔ neuralGenericFormat β fexp x := by
  constructor
  · intro hx
    simpa using neural_generic_format_neg (β := β) (fexp := fexp) (-x) hx
  · exact neural_generic_format_neg (β := β) (fexp := fexp) x

/-- A value is representable exactly when its absolute value is representable. -/
theorem neural_generic_format_abs_iff (x : ℝ) :
    neuralGenericFormat β fexp (abs x) ↔ neuralGenericFormat β fexp x := by
  rcases le_total 0 x with hx | hx
  · simp [abs_of_nonneg hx]
  · rw [abs_of_nonpos hx, neural_generic_format_neg_iff]

/-- A radix power is representable whenever its canonical exponent is no larger than its exponent. -/
theorem neural_generic_format_bpow (e : ℤ) (h : fexp (e + 1) ≤ e) :
    neuralGenericFormat β fexp (neuralBpow β e) := by
  apply neural_generic_format_of_scaled_mantissa_int
    (n := Int.ofNat (β.base ^ (e - fexp (e + 1)).toNat))
  have hd : 0 ≤ e - fexp (e + 1) := sub_nonneg.mpr h
  calc
    neuralScaledMantissa β fexp (neuralBpow β e) =
        neuralBpow β e * neuralBpow β (-fexp (e + 1)) := by
      simp [neuralScaledMantissa, neuralCexp]
    _ = neuralBpow β (e - fexp (e + 1)) := by
      rw [← neuralBpow.add_exp]
      congr 1
    _ = Int.ofNat (β.base ^ (e - fexp (e + 1)).toNat) := by
      obtain ⟨k, hk⟩ := Int.eq_ofNat_of_zero_le hd
      rw [hk]
      simp [neuralBpow, NeuralRadix.toReal]

/--
A mantissa/exponent representation is generic when its stored exponent is at least the canonical
exponent selected for its value.
-/
theorem neural_generic_format_of_toReal_of_cexp_le (f : NeuralFloat β) (x : ℝ)
    (hxf : x = neuralToReal f) (he : neuralCexp β fexp x ≤ f.exponent) :
    neuralGenericFormat β fexp x := by
  obtain ⟨n, hn⟩ := neuralBpow_eq_natCast_of_nonneg β
    (f.exponent - neuralCexp β fexp x) (sub_nonneg.mpr he)
  apply neural_generic_format_of_scaled_mantissa_int (n := f.mantissa * Int.ofNat n)
  rw [neuralScaledMantissa_eq_div]
  have hxfdiv : x / neuralBpow β (neuralCexp β fexp x) =
      neuralToReal f / neuralBpow β (neuralCexp β fexp x) :=
    congrArg (fun y : ℝ => y / neuralBpow β (neuralCexp β fexp x)) hxf
  rw [hxfdiv, neuralToReal]
  calc
    (f.mantissa : ℝ) * neuralBpow β f.exponent /
        neuralBpow β (neuralCexp β fexp x) =
        (f.mantissa : ℝ) * neuralBpow β (f.exponent - neuralCexp β fexp x) := by
      rw [neuralBpow.sub_exp]
      field_simp
    _ = (f.mantissa : ℝ) * n := by rw [hn]
    _ = (f.mantissa * Int.ofNat n : ℤ) := by norm_num

/-- A positive representable value uses a canonical exponent strictly below its magnitude. -/
theorem neuralCexp_lt_magnitude_of_pos_generic {x : ℝ} (hx : 0 < x)
    (hfmt : neuralGenericFormat β fexp x) :
    neuralCexp β fexp x < neuralMagnitude β x := by
  obtain ⟨n, hn⟩ := neural_scaled_mantissa_int_of_generic
    (β := β) (fexp := fexp) x hfmt
  have hbpos := neuralBpow.pos β (neuralCexp β fexp x)
  have hspos : 0 < neuralScaledMantissa β fexp x := by
    rw [neuralScaledMantissa_eq_div]
    positivity
  have hnpos : 0 < n := by
    exact_mod_cast (show (0 : ℝ) < n by simpa [hn] using hspos)
  by_contra hnot
  have hmagle : neuralMagnitude β x ≤ neuralCexp β fexp x := le_of_not_gt hnot
  have hxUpper : x < neuralBpow β (neuralMagnitude β x) := by
    simpa [abs_of_pos hx] using (neuralMagnitude_spec β x hx.ne').2
  have hslt : neuralScaledMantissa β fexp x < 1 := by
    rw [neuralScaledMantissa_eq_div, div_lt_one hbpos]
    exact hxUpper.trans_le ((neuralBpow_le_neuralBpow_iff β _ _).2 hmagle)
  have hnlt : n < 1 := by exact_mod_cast (show (n : ℝ) < 1 by simpa [hn] using hslt)
  exact (not_lt_of_ge (Int.add_one_le_iff.mpr hnpos)) hnlt

/-- Representability of a radix power forces its canonical exponent below that power. -/
theorem neural_fexp_succ_le_of_generic_bpow (e : ℤ)
    (hfmt : neuralGenericFormat β fexp (neuralBpow β e)) : fexp (e + 1) ≤ e := by
  have h := neuralCexp_lt_magnitude_of_pos_generic
    (β := β) (fexp := fexp) (neuralBpow.pos β e) hfmt
  simp [neuralCexp] at h
  linarith

/-- If `β^e` is representable, the exponent selected for the bin below it is at most `e`. -/
theorem neural_fexp_le_of_generic_bpow (e : ℤ)
    (hfmt : neuralGenericFormat β fexp (neuralBpow β e)) : fexp e ≤ e := by
  have hnext := neural_fexp_succ_le_of_generic_bpow (β := β) (fexp := fexp) e hfmt
  by_contra hnot
  have heLt : e < fexp e := lt_of_not_ge hnot
  have heSucc : e + 1 ≤ fexp e := Int.add_one_le_iff.mpr heLt
  have hconst := ((NeuralValidExp.flocq_valid (fexp := fexp) e).2 heLt.le).2 (e + 1) heSucc
  linarith

/--
A value representable with `fexp₁` remains representable with `fexp₂` when the second canonical
exponent is no larger at that value's magnitude.  The condition is local because representability
of `x` only depends on the exponent selected at `magnitude x`.
-/
theorem neural_generic_inclusion_mag {fexp₁ fexp₂ : ℤ → ℤ}
    [NeuralValidExp fexp₁] [NeuralValidExp fexp₂] {x : ℝ}
    (hexp : x ≠ 0 → fexp₂ (neuralMagnitude β x) ≤ fexp₁ (neuralMagnitude β x))
    (hx : neuralGenericFormat β fexp₁ x) : neuralGenericFormat β fexp₂ x := by
  by_cases hx0 : x = 0
  · subst x
    exact neural_generic_format_zero
  obtain ⟨n, hn⟩ := neural_scaled_mantissa_int_of_generic
    (β := β) (fexp := fexp₁) x hx
  let f : NeuralFloat β :=
    { mantissa := n, exponent := neuralCexp β fexp₁ x }
  have hxf : x = neuralToReal f := by
    have hrepr := neural_scaled_mantissa_mul_bpow (β := β) (fexp := fexp₁) x
    rw [hn] at hrepr
    exact hrepr.symm
  apply neural_generic_format_of_toReal_of_cexp_le f x hxf
  change fexp₂ (neuralMagnitude β x) ≤ fexp₁ (neuralMagnitude β x)
  exact hexp hx0

/-- Pointwise-smaller exponent selection defines a containing generic format. -/
theorem neural_generic_inclusion {fexp₁ fexp₂ : ℤ → ℤ}
    [NeuralValidExp fexp₁] [NeuralValidExp fexp₂]
    (hexp : ∀ e, fexp₂ e ≤ fexp₁ e) {x : ℝ}
    (hx : neuralGenericFormat β fexp₁ x) : neuralGenericFormat β fexp₂ x :=
  neural_generic_inclusion_mag (fun _ => hexp _) hx

/-- No generic-format value lies strictly between consecutive points on its canonical grid. -/
theorem neural_generic_format_discrete (x : ℝ) (m : ℤ)
    (hlower : (m : ℝ) * neuralBpow β (neuralCexp β fexp x) < x)
    (hupper : x < ((m + 1 : ℤ) : ℝ) * neuralBpow β (neuralCexp β fexp x)) :
    ¬neuralGenericFormat β fexp x := by
  intro hx
  obtain ⟨n, hn⟩ := neural_scaled_mantissa_int_of_generic
    (β := β) (fexp := fexp) x hx
  have hb : 0 < neuralBpow β (neuralCexp β fexp x) := neuralBpow.pos β _
  have hrepr := neural_scaled_mantissa_mul_bpow (β := β) (fexp := fexp) x
  rw [hn] at hrepr
  have hlower' : (m : ℝ) * neuralBpow β (neuralCexp β fexp x) <
      (n : ℝ) * neuralBpow β (neuralCexp β fexp x) := hlower.trans_eq hrepr.symm
  have hupper' : (n : ℝ) * neuralBpow β (neuralCexp β fexp x) <
      ((m + 1 : ℤ) : ℝ) * neuralBpow β (neuralCexp β fexp x) := hrepr.trans_lt hupper
  have hmnR : (m : ℝ) < (n : ℝ) := by
    nlinarith [hlower']
  have hnmR : (n : ℝ) < ((m + 1 : ℤ) : ℝ) := by
    nlinarith [hupper']
  have hmn : m < n := by exact_mod_cast hmnR
  have hnm : n < m + 1 := by exact_mod_cast hnmR
  linarith

/-- Every generic-format value has a canonical mantissa/exponent representation. -/
theorem neural_canonical_exists_of_generic {x : ℝ} (hx : neuralGenericFormat β fexp x) :
    ∃ f : NeuralFloat β, x = neuralToReal f ∧ NeuralCanonical β fexp f := by
  let f : NeuralFloat β :=
    { mantissa := ⌊neuralScaledMantissa β fexp x⌋
      exponent := neuralCexp β fexp x }
  refine ⟨f, hx, ?_⟩
  unfold NeuralCanonical
  change neuralCexp β fexp x = neuralCexp β fexp (neuralToReal f)
  rw [← hx]

/-- The real value of a canonical representation belongs to its generic format. -/
theorem neural_generic_format_of_canonical (f : NeuralFloat β)
    (hf : NeuralCanonical β fexp f) :
    neuralGenericFormat β fexp (neuralToReal f) := by
  apply neural_generic_format_of_toReal_of_cexp_le f (neuralToReal f) rfl
  exact hf.ge

/-- Canonical representations of the same real value are equal. -/
theorem neural_canonical_unique {f g : NeuralFloat β}
    (hf : NeuralCanonical β fexp f) (hg : NeuralCanonical β fexp g)
    (hval : neuralToReal f = neuralToReal g) : f = g := by
  have hexp : f.exponent = g.exponent := by
    rw [hf, hg, hval]
  have hmant : f.mantissa = g.mantissa := by
    have hb : 0 < neuralBpow β f.exponent := neuralBpow.pos β _
    have hreal : (f.mantissa : ℝ) * neuralBpow β f.exponent =
        (g.mantissa : ℝ) * neuralBpow β f.exponent := by
      simpa [neuralToReal, hexp] using hval
    have hcast : (f.mantissa : ℝ) = (g.mantissa : ℝ) := by nlinarith
    exact_mod_cast hcast
  cases f
  cases g
  simp_all

end TorchLean.Floats
