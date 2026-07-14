/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Format.Formats
public import NN.Floats.NeuralFloat.Analysis.Ulp

/-!
# ULP Conditions for Standard Formats

FIX, positive-precision FLX, and positive-precision FLT do not flush the selected ULP to zero.
These witnesses make the generic ULP representability theorem available for each standard family.
-/

@[expose] public section

namespace TorchLean.Floats

/-- Fixed-point exponent selection preserves ULP representability. -/
abbrev fixNotFlushToZero (emin : ℤ) : NeuralExpNotFlushToZero (FIXExp emin) where
  ulpExponent := by simp [FIXExp]

/-- Positive-precision unbounded floats preserve ULP representability. -/
abbrev flxNotFlushToZero (prec : ℤ) (hprec : 0 < prec) :
    NeuralExpNotFlushToZero (FLXExp prec) where
  ulpExponent := by
    intro e
    simp [FLXExp]
    linarith

/-- Positive-precision lower-bounded floats preserve ULP representability. -/
abbrev fltNotFlushToZero (emin prec : ℤ) (hprec : 0 < prec) :
    NeuralExpNotFlushToZero (FLTExp emin prec) where
  ulpExponent := by
    intro e
    simp only [FLTExp]
    apply max_le
    · have hprecOne : 1 ≤ prec := by linarith
      linarith
    · exact le_max_right _ _

end TorchLean.Floats
