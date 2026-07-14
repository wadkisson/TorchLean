/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.BridgeFP32Total
public import NN.Floats.IEEEExec.Reductions
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
