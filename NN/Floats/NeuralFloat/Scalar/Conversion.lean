/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Core
public import NN.Floats.NeuralFloat.Metadata
public import NN.Floats.NeuralFloat.Rounding.Core

/-!
# NeuralFloat Conversion Helpers

This module contains small utilities for *moving values between*:

- real semantics (`ℝ`), and
- an annotated rounded value whose mathematical payload is `NeuralFloat β`.

Important scope note:

- The canonical rounding semantics is `neuralRound` (see `Rounding/Core.lean`), which
  models **one** “compute in `ℝ`, then round” step and comes with the key half-ULP theorem
  `neural_error_bound_ulp` (under round-to-nearest assumptions).
- The helpers in this file are conversion utilities and examples: packaging the rounded result into a
  `AnnotatedNeuralFloat` and attaching explicit error metadata.

If you are doing proofs about rounding error, prefer theorems in:

- `NN/Floats/NeuralFloat/Rounding/Core.lean` (core half-ULP bound), and
- `NN/Floats/NeuralFloat/Error/Bounds.lean` (small derived bounds).
-/

@[expose] public section


namespace TorchLean.Floats.Conversion

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]

/--
Conversion result with error tracking.
-/
structure NeuralConversionResult (α : Type) where
  /-- Converted value. -/
  value : α
  /--
  Absolute error bound for the conversion step.

  For conversions that only *repackage* an already-rounded value, this may be a metadata field
  rather than a proved bound.
  -/
  errorBound : ℝ
  /-- The attached absolute error bound is nonnegative. -/
  errorBound_nonneg : 0 ≤ errorBound
  /--
  Optional ULP-scale metadata at the input point.

  This is not itself an error bound. For round-to-nearest, a typical one-step error bound is
  `ulp(x)/2`, proved separately (see `neural_error_bound_ulp`).
  -/
  ulpScale : Option ℝ
  /-- Training phase, when all values in the result have one unambiguous provenance phase. -/
  phase : Option TrainingPhase

namespace NeuralConversionResult

/-- A conversion result is exact when its attached absolute error is zero. -/
def IsExact {α : Type} (result : NeuralConversionResult α) : Prop :=
  result.errorBound = 0

end NeuralConversionResult

/--
Convert an annotated float to its real semantics while retaining its conversion metadata.

In words: the returned `value` is exactly `neural_to_real f`.
The `errorBound` field is the metadata stored in the input record; it is not recomputed here.
-/
noncomputable def neuralFloatToReal (f : AnnotatedNeuralFloat β) : NeuralConversionResult ℝ :=
  { value := f.toReal,
    errorBound := f.metadata.errorBound,
    errorBound_nonneg := f.metadata.errorBound_nonneg,
    ulpScale := none,
    phase := some f.metadata.phase }

/--
Round a real number `x` onto the `(β,fexp,rnd)` grid and attach conversion metadata.

In words:
The returned payload is the same one used by `neural_round` (it uses
`mantissa := rnd (scaled_mantissa x)` and `exponent := cexp x`), and
`errorBound = |neural_to_real(value) - x|` is the actual absolute error for this rounding step.

If you additionally assume round-to-nearest (`NeuralValidRndToNearest`), the theorem
`neural_error_bound_ulp` bounds this error by `ulp(x)/2`.
-/
noncomputable def realToNeuralFloat (precision : NeuralPrecision) (rnd : ℝ → ℤ)
    [NeuralValidRnd rnd] (x : ℝ) (phase : TrainingPhase := TrainingPhase.forward) :
    NeuralConversionResult (AnnotatedNeuralFloat β) := by
  let m : ℤ := rnd (neuralScaledMantissa β fexp x)
  let e : ℤ := neuralCexp β fexp x
  let f : NeuralFloat β :=
    { mantissa := m
      exponent := e }
  let y : ℝ := neuralToReal (β := β) f
  let err : ℝ := abs (y - x)
  exact
    { value :=
        { value := f
          metadata :=
            { precision := precision
              errorBound := err
              errorBound_nonneg := abs_nonneg _
              phase := phase } }
      errorBound := err
      errorBound_nonneg := abs_nonneg _
      ulpScale := some (neuralUlp β fexp x)
      phase := some phase }

/-- Convert list of reals to neural floats -/
noncomputable def listToNeuralFloats (precision : NeuralPrecision) (rnd : ℝ → ℤ)
    [NeuralValidRnd rnd] (values : List ℝ) (phase : TrainingPhase := TrainingPhase.forward) :
    NeuralConversionResult
      (List (AnnotatedNeuralFloat β)) := by
  let conv := values.map (fun x =>
    realToNeuralFloat (β := β) (fexp := fexp) precision rnd x phase)
  let converted := conv.map (·.value)
  let maxError := (conv.map (·.errorBound)).foldl max 0
  let errorBound := max 0 maxError
  let maxUlp := (values.map (neuralUlp β fexp)).foldl max 0
  exact
    { value := converted
      errorBound := errorBound
      errorBound_nonneg := le_max_left 0 maxError
      ulpScale := some maxUlp
      phase := some phase }

/-- Convert neural floats back to reals -/
noncomputable def neuralFloatsToList (neural_floats : List (AnnotatedNeuralFloat β)) :
  NeuralConversionResult (List ℝ) := by
  let converted := neural_floats.map AnnotatedNeuralFloat.toReal
  let maxError := (neural_floats.map (·.metadata.errorBound)).foldl max 0
  let errorBound := max 0 maxError
  exact {
    value := converted,
    errorBound := errorBound,
    errorBound_nonneg := le_max_left 0 maxError,
    ulpScale := none,
    phase := none
  }

/-!
## Affine quantization

An affine quantizer represents a real value `x` by an integer code

`q = clamp(round(x / scale) + zeroPoint, qmin, qmax)`

and reconstructs `scale * (q - zeroPoint)`. The definition is independent of a storage width: an
8-bit, 4-bit, or custom code set is obtained by choosing the corresponding integer bounds. This is
the standard integer-arithmetic quantization map used by neural-network runtimes; see Jacob et al.,
"Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference",
CVPR 2018, doi:10.1109/CVPR.2018.00286.
-/

/-- Parameters of a bounded affine quantizer. -/
structure AffineQuantizer where
  /-- Distance between adjacent reconstructed real values. -/
  scale : ℝ
  /-- Integer code representing real zero when it lies in the code range. -/
  zeroPoint : ℤ
  /-- Smallest stored code. -/
  qmin : ℤ
  /-- Largest stored code. -/
  qmax : ℤ
  /-- A quantization scale is strictly positive. -/
  scale_pos : 0 < scale
  /-- The code interval is nonempty. -/
  codeRange : qmin ≤ qmax

namespace AffineQuantizer

/-- Saturate an integer to the quantizer's code interval. -/
def clampCode (q : AffineQuantizer) (code : ℤ) : ℤ :=
  max q.qmin (min q.qmax code)

/-- Integer code before saturation. -/
noncomputable def rawCode (q : AffineQuantizer) (rnd : ℝ → ℤ) (x : ℝ) : ℤ :=
  rnd (x / q.scale) + q.zeroPoint

/-- Quantize a real value using the supplied integer rounding rule and saturate it to the code set. -/
noncomputable def quantize (q : AffineQuantizer) (rnd : ℝ → ℤ) (x : ℝ) : ℤ :=
  q.clampCode (q.rawCode rnd x)

/-- Reconstruct a real value from an integer code. -/
noncomputable def dequantize (q : AffineQuantizer) (code : ℤ) : ℝ :=
  q.scale * ((code - q.zeroPoint : ℤ) : ℝ)

/-- Saturation always returns a valid code. -/
theorem clampCode_mem (q : AffineQuantizer) (code : ℤ) :
    q.qmin ≤ q.clampCode code ∧ q.clampCode code ≤ q.qmax := by
  constructor
  · exact le_max_left _ _
  · exact max_le q.codeRange (min_le_left _ _)

/-- Saturation fixes a code already inside the representable interval. -/
@[simp] theorem clampCode_eq_self (q : AffineQuantizer) {code : ℤ}
    (hlo : q.qmin ≤ code) (hhi : code ≤ q.qmax) :
    q.clampCode code = code := by
  simp [clampCode, hlo, hhi]

/-- Every quantized result lies in the declared code interval. -/
theorem quantize_mem (q : AffineQuantizer) (rnd : ℝ → ℤ) (x : ℝ) :
    q.qmin ≤ q.quantize rnd x ∧ q.quantize rnd x ≤ q.qmax :=
  q.clampCode_mem _

/-- Dequantizing the zero point gives real zero. -/
@[simp] theorem dequantize_zeroPoint (q : AffineQuantizer) :
    q.dequantize q.zeroPoint = 0 := by
  simp [dequantize]

/-- Quantization is monotone for every valid integer rounding rule. -/
theorem quantize_mono (q : AffineQuantizer) (rnd : ℝ → ℤ) [NeuralValidRnd rnd]
    {x y : ℝ} (hxy : x ≤ y) :
    q.quantize rnd x ≤ q.quantize rnd y := by
  have hscaled : x / q.scale ≤ y / q.scale :=
    div_le_div_of_nonneg_right hxy q.scale_pos.le
  have hround : rnd (x / q.scale) ≤ rnd (y / q.scale) :=
    NeuralValidRnd.monotone _ _ hscaled
  have hcode : rnd (x / q.scale) + q.zeroPoint ≤ rnd (y / q.scale) + q.zeroPoint :=
    by simpa [add_comm] using add_le_add_right hround q.zeroPoint
  have hmin :
      min q.qmax (rnd (x / q.scale) + q.zeroPoint) ≤
        min q.qmax (rnd (y / q.scale) + q.zeroPoint) :=
    min_le_min (le_refl q.qmax) hcode
  unfold quantize clampCode rawCode
  exact max_le_max (le_refl q.qmin) hmin

/--
Every in-range integer code survives a dequantize/quantize round trip.

This theorem uses only the two defining laws of a valid rounding rule: monotonicity and exactness on
integers. It does not rely on nearest rounding.
-/
@[simp] theorem quantize_dequantize (q : AffineQuantizer) (rnd : ℝ → ℤ) [NeuralValidRnd rnd]
    {code : ℤ} (hlo : q.qmin ≤ code) (hhi : code ≤ q.qmax) :
    q.quantize rnd (q.dequantize code) = code := by
  have hscale : q.scale ≠ 0 := ne_of_gt q.scale_pos
  have hscaled : q.dequantize code / q.scale = ((code - q.zeroPoint : ℤ) : ℝ) := by
    simp [dequantize, hscale]
  have hrnd : rnd (((code - q.zeroPoint : ℤ) : ℝ)) = code - q.zeroPoint :=
    NeuralValidRnd.id _
  rw [quantize, rawCode, hscaled]
  rw [hrnd]
  simp only [Int.sub_add_cancel]
  exact q.clampCode_eq_self hlo hhi

/--
Without saturation, nearest affine quantization has absolute reconstruction error at most half a
quantization step.
-/
theorem dequantize_rawCode_error_le (q : AffineQuantizer) (rnd : ℝ → ℤ)
    [NeuralValidRndToNearest rnd] (x : ℝ) :
    abs (q.dequantize (q.rawCode rnd x) - x) ≤ q.scale / 2 := by
  have hscale : q.scale ≠ 0 := ne_of_gt q.scale_pos
  have hround := NeuralValidRndToNearest.abs_sub_le_half (rnd := rnd) (x / q.scale)
  have hrewrite :
      q.dequantize (q.rawCode rnd x) - x =
        q.scale * ((rnd (x / q.scale) : ℝ) - x / q.scale) := by
    simp only [dequantize, rawCode, Int.cast_sub, Int.cast_add]
    field_simp
    ring
  rw [hrewrite, abs_mul, abs_of_pos q.scale_pos]
  have hhalf : (2⁻¹ : ℝ) = 1 / 2 := by norm_num
  rw [hhalf] at hround
  exact (mul_le_mul_of_nonneg_left hround q.scale_pos.le).trans_eq (by ring)

/-- The same half-step bound holds for saturated quantization whenever saturation is inactive. -/
theorem dequantize_quantize_error_le (q : AffineQuantizer) (rnd : ℝ → ℤ)
    [NeuralValidRndToNearest rnd] (x : ℝ)
    (hlo : q.qmin ≤ q.rawCode rnd x) (hhi : q.rawCode rnd x ≤ q.qmax) :
    abs (q.dequantize (q.quantize rnd x) - x) ≤ q.scale / 2 := by
  rw [quantize, q.clampCode_eq_self hlo hhi]
  exact q.dequantize_rawCode_error_le rnd x

end AffineQuantizer

end TorchLean.Floats.Conversion
