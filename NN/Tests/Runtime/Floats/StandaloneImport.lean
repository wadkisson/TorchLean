/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats

/-!
# Standalone Floating-Point Import

This regression module deliberately imports only `NN.Floats`. It exercises the public numerical
surface without tensors, models, autograd, CUDA, certificate checkers, or external processes. The
repository linter separately enforces that the import closure cannot acquire those dependencies.
-/

@[expose] public section

namespace Tests.Floats.StandaloneImport

open TorchLean.Floats
open TorchLean.Floats.IEEE754
open TorchLean.Floats.IEEE754.IEEE32Exec
open TorchLean.Floats.Quantization

/-- Scalar affine quantization is available without TorchLean's tensor layer. -/
noncomputable def int8Quantizer : AffineQuantizer where
  scale := 1 / 10
  zeroPoint := 0
  qmin := -128
  qmax := 127
  scale_pos := by norm_num
  codeRange := by norm_num

/-- The standalone scalar quantizer retains its code-range theorem. -/
theorem int8Quantizer_codeRange (rnd : ℝ → ℤ) (x : ℝ) :
    int8Quantizer.qmin ≤ int8Quantizer.quantize rnd x ∧
      int8Quantizer.quantize rnd x ≤ int8Quantizer.qmax :=
  int8Quantizer.quantize_mem rnd x

/-- The standalone executable kernel retains exact bit-pattern round trips. -/
theorem executable_bits_roundTrip (bits : UInt32) :
    toBits (ofBits bits) = bits :=
  toBits_ofBits bits

/-- Execute one exact binary32 addition through the standalone import. -/
def run : IO Unit := do
  let one := ofBits 0x3F800000
  let two := one + one
  unless two.bits == 0x40000000 do
    throw <| IO.userError s!"standalone IEEE32 addition failed: bits={two.bits}"

end Tests.Floats.StandaloneImport
