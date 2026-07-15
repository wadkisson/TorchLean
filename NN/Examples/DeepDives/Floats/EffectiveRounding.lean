/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Bridge.FP32Total
public import NN.Floats.IEEEExec.DirectedRoundingSoundness
public import NN.Floats.IEEEExec.Reductions
public import NN.Floats.IEEEExec.Rules.SpecialRules
public import NN.Floats.NeuralFloat.Rounding
public import NN.Floats.NeuralFloat.Scalar.Conversion
public import NN.Spec.Core.Tensor
public import NN.Spec.Core.TensorOps

/-!
# Effective Rounding in Tensor Semantics

This example uses the same shape-indexed tensor operation twice.  The `FP32` tensor gives the
proof-oriented rounded-real semantics.  The `IEEE32Exec` tensor executes binary32 arithmetic from
bits.  On a finite result, the IEEE bridge and the effective rounding calculation identify the
same canonical mantissa and exponent.
-/

@[expose] public section

namespace NN.Examples.DeepDives.Floats.EffectiveRounding

open Spec
open TorchLean.Floats
open TorchLean.Floats.IEEE754
open TorchLean.Floats.IEEE754.IEEE32Exec

def width : Nat := 4
abbrev vectorShape : Spec.Shape := .dim width .scalar

/-- Executable binary32 tensors used by the example. -/
def runtimeOnes : Spec.Tensor IEEE32Exec vectorShape :=
  Spec.fill posOne vectorShape

def runtimeTwos : Spec.Tensor IEEE32Exec vectorShape :=
  Spec.fill (ofBits 0x40000000) vectorShape

/-- This addition executes pointwise in the bit-level IEEE32 model. -/
def runtimeSum : Spec.Tensor IEEE32Exec vectorShape :=
  Spec.Tensor.addSpec runtimeOnes runtimeTwos

/-- Proof-oriented tensors use the same tensor API with rounded-real FP32 scalars. -/
noncomputable def specOne : FP32 :=
  NF.ofReal (β := binaryRadix) (fexp := fexp32) (rnd := rnd32) 1

noncomputable def specTwo : FP32 :=
  NF.ofReal (β := binaryRadix) (fexp := fexp32) (rnd := rnd32) 2

noncomputable def specOnes : Spec.Tensor FP32 vectorShape :=
  Spec.fill specOne vectorShape

noncomputable def specTwos : Spec.Tensor FP32 vectorShape :=
  Spec.fill specTwo vectorShape

noncomputable def specSum : Spec.Tensor FP32 vectorShape :=
  Spec.Tensor.addSpec specOnes specTwos

/-- Every proof-oriented tensor entry exposes the effective nearest-even representation. -/
theorem specSum_entry_computed (i : Fin width) :
    FP32.toReal (Spec.Tensor.vecGet specSum i) =
      neuralToReal (β := binaryRadix) {
        mantissa := neuralNearestEvenMantissa
          (neuralScaledMantissa binaryRadix fexp32 (specOne.val + specTwo.val))
        exponent := neuralCexp binaryRadix fexp32 (specOne.val + specTwo.val) } := by
  change FP32.toReal (specOne + specTwo) = _
  exact FP32.add_toReal_eq_computed specOne specTwo

/--
Every executable tensor entry reaches the same effective representation once finiteness is checked.
The finiteness premise is discharged by computation for the concrete `1 + 2` example.
-/
theorem runtimeSum_entry_computed (i : Fin width) :
    toReal (Spec.Tensor.vecGet runtimeSum i) =
      neuralToReal (β := binaryRadix) {
        mantissa := neuralNearestEvenMantissa
          (neuralScaledMantissa binaryRadix fexp32
            (toReal posOne + toReal (ofBits 0x40000000)))
        exponent := neuralCexp binaryRadix fexp32
          (toReal posOne + toReal (ofBits 0x40000000)) } := by
  change toReal (add posOne (ofBits 0x40000000)) = _
  exact toReal_add_eq_computed_of_isFinite posOne (ofBits 0x40000000) (by decide)

/-! ## Named rounding modes and fused enclosures -/

/-- The public mode API avoids passing a raw integer-rounding function at each call site. -/
noncomputable def oneThirdDown : ℝ :=
  NeuralRoundingMode.towardNegative.round
    (β := binaryRadix) (fexp := fexp32) (1 / 3)

/-- Directed rounding gives a certified lower endpoint, not merely a differently named value. -/
theorem oneThirdDown_le : oneThirdDown ≤ 1 / 3 := by
  simpa [oneThirdDown, NeuralRoundingMode.round, NeuralRoundingMode.roundingFunction] using
    (neural_round_floor_le (β := binaryRadix) (fexp := fexp32) (1 / 3))

/-- A concrete fused multiply-add lower endpoint, computed directly from binary32 inputs. -/
def fusedLower : IEEE32Exec :=
  fmaDown posOne (ofBits 0x40000000) (ofBits 0x3e800000)

/-- The corresponding upper endpoint. -/
def fusedUpper : IEEE32Exec :=
  fmaUp posOne (ofBits 0x40000000) (ofBits 0x3e800000)

/-- The executable directed FMA endpoints enclose the exact single-rounding expression. -/
theorem fused_enclosure :
    toEReal fusedLower ≤
        ((toReal posOne * toReal (ofBits 0x40000000) + toReal (ofBits 0x3e800000) : ℝ) : EReal) ∧
      ((toReal posOne * toReal (ofBits 0x40000000) + toReal (ofBits 0x3e800000) : ℝ) : EReal) ≤
        toEReal fusedUpper := by
  constructor
  · exact toEReal_fmaDown_le _ _ _ (by decide) (by decide) (by decide)
  · exact toEReal_fmaUp_ge _ _ _ (by decide) (by decide) (by decide)

/-- Directed binary32 square-root endpoints for the exact input `2`. -/
def sqrtLower : IEEE32Exec :=
  sqrtDown (ofBits 0x40000000)

def sqrtUpper : IEEE32Exec :=
  sqrtUp (ofBits 0x40000000)

/-- The executable endpoints enclose the exact real value `sqrt 2`. -/
theorem sqrt_enclosure :
    toEReal sqrtLower ≤ (Real.sqrt (toReal (ofBits 0x40000000)) : EReal) ∧
      (Real.sqrt (toReal (ofBits 0x40000000)) : EReal) ≤ toEReal sqrtUpper := by
  constructor
  · exact toEReal_sqrtDown_le _ (by decide) (by decide)
  · exact toEReal_sqrtUp_ge _ (by decide) (by decide)

/-! ## Fixed grids, quantization, and double rounding -/

/-- A signed affine code set with quarter-unit spacing. The construction is not tied to a tensor
layout or storage width; those choices only determine the integer code bounds. -/
noncomputable def signedQuarterQuantizer : Conversion.AffineQuantizer where
  scale := 1 / 4
  zeroPoint := 0
  qmin := -128
  qmax := 127
  scale_pos := by norm_num
  codeRange := by norm_num

/-- Every in-range code is recovered exactly after dequantization and requantization. -/
theorem signedQuarterQuantizer_roundtrip {code : ℤ}
    (hlo : -128 ≤ code) (hhi : code ≤ 127) :
    signedQuarterQuantizer.quantize neuralNearestEven
        (signedQuarterQuantizer.dequantize code) = code := by
  exact signedQuarterQuantizer.quantize_dequantize neuralNearestEven hlo hhi

/-- When saturation is inactive, nearest-even reconstruction is within half a quantization step. -/
theorem signedQuarterQuantizer_error (x : ℝ)
    (hlo : signedQuarterQuantizer.qmin ≤
      signedQuarterQuantizer.rawCode neuralNearestEven x)
    (hhi : signedQuarterQuantizer.rawCode neuralNearestEven x ≤
      signedQuarterQuantizer.qmax) :
    abs (signedQuarterQuantizer.dequantize
      (signedQuarterQuantizer.quantize neuralNearestEven x) - x) ≤ 1 / 8 := by
  have h :=
    signedQuarterQuantizer.dequantize_quantize_error_le neuralNearestEven x hlo hhi
  convert h using 1
  all_goals norm_num [signedQuarterQuantizer]

/-- Round-to-odd on a sufficiently fine binary grid prevents double rounding on the quarter grid. -/
theorem quarterGrid_doubleRounding_safe (extra : ℕ) (x : ℝ) :
    neuralRoundAtScale neuralNearestEven (1 / 4)
        (neuralRoundAtScale neuralOddRound
          ((1 / 4) / (2 : ℝ) ^ (extra + 2)) x) =
      neuralRoundAtScale neuralNearestEven (1 / 4) x := by
  exact neuralRoundAtScale_nearestEven_after_odd_binary_extra extra (1 / 4) x (by norm_num)

/-! ## IEEE exception status -/

/-- The status-bearing API records division by zero separately from invalid operation. -/
theorem one_div_zero_status :
    (divWithStatus posOne posZero).status.divideByZero = true ∧
      (divWithStatus posOne posZero).status.invalid = false := by
  exact divWithStatus_divideByZero_of_finite_nonzero posOne posZero
    (by decide) (by decide) (by decide)

/-! ## A training-shaped reduction -/

/-- A fixed two-term dot-product tree. Its shape records the accumulation order. -/
def runtimeDotTree : SumTree (IEEE32Exec × IEEE32Exec) :=
  .node
    (.leaf (posOne, ofBits 0x40000000))
    (.leaf (ofBits 0x40400000, ofBits 0x40800000))

/-- Every intermediate product and addition in the example remains finite. -/
theorem runtimeDotTree_finite : FiniteEvalDot runtimeDotTree := by
  simp only [runtimeDotTree, FiniteEvalDot]
  decide

/--
The executable dot product decodes to the computed mantissa/exponent representation of its final
rounded accumulation. The two leaf multiplications are rounded before this final addition.
-/
theorem runtimeDotTree_computed :
    toReal (evalDotIEEE runtimeDotTree) =
      neuralToReal (β := binaryRadix) {
        mantissa := neuralNearestEvenMantissa
          (neuralScaledMantissa binaryRadix fexp32
            (evalRealDotIEEE (.leaf (posOne, ofBits 0x40000000)) +
              evalRealDotIEEE (.leaf (ofBits 0x40400000, ofBits 0x40800000))))
        exponent := neuralCexp binaryRadix fexp32
          (evalRealDotIEEE (.leaf (posOne, ofBits 0x40000000)) +
            evalRealDotIEEE (.leaf (ofBits 0x40400000, ofBits 0x40800000))) } := by
  rw [toReal_evalDotIEEE_eq_evalRealDotIEEE_of_FiniteEvalDot runtimeDotTree runtimeDotTree_finite]
  exact evalRealDotIEEE_node_eq_computed _ _

end NN.Examples.DeepDives.Floats.EffectiveRounding
