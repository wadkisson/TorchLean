/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Rounding.Core

/-!
# Generic Rounding Properties

Order properties that follow directly from scaling by a positive radix power. These results do not
assume a particular standard format.
-/

@[expose] public section

namespace TorchLean.Floats

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]

/-- Rounding toward negative infinity never exceeds the exact value. -/
theorem neural_round_floor_le (x : ℝ) :
    neuralRound (β := β) (fexp := fexp) neuralFloorRound x ≤ x := by
  calc
    neuralRound (β := β) (fexp := fexp) neuralFloorRound x =
        (⌊neuralScaledMantissa β fexp x⌋ : ℝ) *
          neuralBpow β (neuralCexp β fexp x) := rfl
    _ ≤ neuralScaledMantissa β fexp x * neuralBpow β (neuralCexp β fexp x) :=
      mul_le_mul_of_nonneg_right (Int.floor_le _) (neuralBpow.nonneg β _)
    _ = x := neural_scaled_mantissa_mul_bpow (β := β) (fexp := fexp) x

/-- Rounding toward positive infinity never falls below the exact value. -/
theorem le_neural_round_ceil (x : ℝ) :
    x ≤ neuralRound (β := β) (fexp := fexp) neuralCeilRound x := by
  calc
    x = neuralScaledMantissa β fexp x * neuralBpow β (neuralCexp β fexp x) :=
      (neural_scaled_mantissa_mul_bpow (β := β) (fexp := fexp) x).symm
    _ ≤ (⌈neuralScaledMantissa β fexp x⌉ : ℝ) *
          neuralBpow β (neuralCexp β fexp x) :=
      mul_le_mul_of_nonneg_right (Int.le_ceil _) (neuralBpow.nonneg β _)
    _ = neuralRound (β := β) (fexp := fexp) neuralCeilRound x := rfl

/-- Every valid integer rounding rule lies between floor and ceiling. -/
theorem neural_floor_le_valid_round (rnd : ℝ → ℤ) [NeuralValidRnd rnd] (x : ℝ) :
    neuralFloorRound x ≤ rnd x := by
  unfold neuralFloorRound
  rw [← NeuralValidRnd.id (rnd := rnd) ⌊x⌋]
  exact NeuralValidRnd.monotone (rnd := rnd) (⌊x⌋ : ℝ) x (Int.floor_le x)

/-- Every valid integer rounding rule lies between floor and ceiling. -/
theorem neural_valid_round_le_ceil (rnd : ℝ → ℤ) [NeuralValidRnd rnd] (x : ℝ) :
    rnd x ≤ neuralCeilRound x := by
  unfold neuralCeilRound
  rw [← NeuralValidRnd.id (rnd := rnd) ⌈x⌉]
  exact NeuralValidRnd.monotone (rnd := rnd) x (⌈x⌉ : ℝ) (Int.le_ceil x)

/-- Every valid integer rounding chooses either floor or ceiling. -/
theorem neural_valid_round_eq_floor_or_ceil (rnd : ℝ → ℤ) [NeuralValidRnd rnd] (x : ℝ) :
    rnd x = neuralFloorRound x ∨ rnd x = neuralCeilRound x := by
  have hfloor := neural_floor_le_valid_round (rnd := rnd) x
  have hceil := neural_valid_round_le_ceil (rnd := rnd) x
  by_cases hle : rnd x ≤ neuralFloorRound x
  · exact Or.inl (le_antisymm hle hfloor)
  · right
    have hsucc : neuralFloorRound x + 1 ≤ rnd x :=
      Int.add_one_le_iff.mpr (lt_of_not_ge hle)
    have hceilSucc : neuralCeilRound x ≤ neuralFloorRound x + 1 := by
      exact Int.ceil_le_floor_add_one x
    exact le_antisymm hceil (hceilSucc.trans hsucc)

/-- Every valid format rounding lies above directed-down rounding. -/
theorem neural_round_floor_le_round (rnd : ℝ → ℤ) [NeuralValidRnd rnd] (x : ℝ) :
    neuralRound (β := β) (fexp := fexp) neuralFloorRound x ≤
      neuralRound (β := β) (fexp := fexp) rnd x := by
  have hi := neural_floor_le_valid_round (rnd := rnd) (neuralScaledMantissa β fexp x)
  have hiR : (neuralFloorRound (neuralScaledMantissa β fexp x) : ℝ) ≤
      (rnd (neuralScaledMantissa β fexp x) : ℝ) := by exact_mod_cast hi
  unfold neuralRound neuralToReal
  exact mul_le_mul_of_nonneg_right
    hiR (neuralBpow.nonneg β _)

/-- Every valid format rounding lies below directed-up rounding. -/
theorem neural_round_le_ceil (rnd : ℝ → ℤ) [NeuralValidRnd rnd] (x : ℝ) :
    neuralRound (β := β) (fexp := fexp) rnd x ≤
      neuralRound (β := β) (fexp := fexp) neuralCeilRound x := by
  have hi := neural_valid_round_le_ceil (rnd := rnd) (neuralScaledMantissa β fexp x)
  have hiR : (rnd (neuralScaledMantissa β fexp x) : ℝ) ≤
      (neuralCeilRound (neuralScaledMantissa β fexp x) : ℝ) := by exact_mod_cast hi
  unfold neuralRound neuralToReal
  exact mul_le_mul_of_nonneg_right
    hiR (neuralBpow.nonneg β _)

/-- Every valid generic rounding chooses either directed-down or directed-up rounding. -/
theorem neuralRound_eq_floor_or_ceil (rnd : ℝ → ℤ) [NeuralValidRnd rnd] (x : ℝ) :
    neuralRound (β := β) (fexp := fexp) rnd x =
        neuralRound (β := β) (fexp := fexp) neuralFloorRound x ∨
      neuralRound (β := β) (fexp := fexp) rnd x =
        neuralRound (β := β) (fexp := fexp) neuralCeilRound x := by
  rcases neural_valid_round_eq_floor_or_ceil rnd (neuralScaledMantissa β fexp x) with h | h
  · left
    unfold neuralRound
    rw [h]
  · right
    unfold neuralRound
    rw [h]

end TorchLean.Floats
