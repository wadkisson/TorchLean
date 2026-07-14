/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.Calc.Operations
public import NN.Floats.Calc.Round

/-!
# Rounded Addition of Mantissa/Exponent Values

Addition first aligns and adds the two exact representations, then rounds the exact sum into the
selected generic format.  The result remains a `NeuralFloat`, so effective arithmetic can continue
without returning to an unstructured real value.
-/

@[expose] public section

namespace TorchLean.Floats
namespace NeuralFloat

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]

/-- Exact addition followed by rounding into the selected format. -/
noncomputable def addRounded (rnd : ℝ → ℤ) (f g : NeuralFloat β) : NeuralFloat β :=
  neuralRoundedFloat (β := β) (fexp := fexp) rnd (neuralToReal (addExact f g))

/-- Rounded representation addition agrees with rounded real addition. -/
@[simp] theorem toReal_addRounded (rnd : ℝ → ℤ) (f g : NeuralFloat β) :
    neuralToReal (addRounded (fexp := fexp) rnd f g) =
      neuralRound (β := β) (fexp := fexp) rnd (neuralToReal f + neuralToReal g) := by
  unfold addRounded neuralRoundedFloat
  rw [toReal_addExact]
  rfl

end NeuralFloat
end TorchLean.Floats
