/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Rounding.Core
public import NN.Spec.Core.TensorOps

/-!
# Affine Quantization

This module gives one affine quantizer for scalars and shape-indexed tensors. For a positive scale
`s`, zero point `z`, and integer code interval `[qmin, qmax]`, quantization and reconstruction are

`Q(x) = clamp(round(x / s) + z, qmin, qmax)` and `D(k) = s (k - z)`.

The tensor operations are pointwise lifts of these equations; they do not build batch, channel, or
image conventions into the arithmetic. Theorems below cover code-range safety, monotonicity,
in-range code round trips, and the half-step reconstruction bound when saturation is inactive.

The equations follow the integer-arithmetic quantization scheme used by common neural-network
runtimes. See Jacob et al., "Quantization and Training of Neural Networks for Efficient
Integer-Arithmetic-Only Inference," CVPR 2018, doi:10.1109/CVPR.2018.00286.
-/

@[expose] public section

namespace TorchLean.Floats.Quantization

open Spec

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

/-- Apply the affine quantizer independently at every coordinate of an arbitrary-rank tensor. -/
noncomputable def quantizeTensor (q : AffineQuantizer) (rnd : ℝ → ℤ) {s : Shape}
    (x : Tensor ℝ s) : Tensor ℤ s :=
  mapTensor (q.quantize rnd) x

/-- Reconstruct every code in an arbitrary-rank tensor on the quantizer's real grid. -/
noncomputable def dequantizeTensor (q : AffineQuantizer) {s : Shape}
    (codes : Tensor ℤ s) : Tensor ℝ s :=
  mapTensor q.dequantize codes

/-- Pointwise condition saying that quantization does not clip any coordinate of `x`. -/
def SaturationInactive (q : AffineQuantizer) (rnd : ℝ → ℤ) {s : Shape}
    (x : Tensor ℝ s) : Prop :=
  Tensor.Forall (fun a => q.qmin ≤ q.rawCode rnd a ∧ q.rawCode rnd a ≤ q.qmax) x

/-- Pointwise condition saying that every stored code belongs to the quantizer's code set. -/
def CodesInRange (q : AffineQuantizer) {s : Shape} (codes : Tensor ℤ s) : Prop :=
  Tensor.Forall (fun code => q.qmin ≤ code ∧ code ≤ q.qmax) codes

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

/-- Every quantized scalar lies in the declared code interval. -/
theorem quantize_mem (q : AffineQuantizer) (rnd : ℝ → ℤ) (x : ℝ) :
    q.qmin ≤ q.quantize rnd x ∧ q.quantize rnd x ≤ q.qmax :=
  q.clampCode_mem _

/-- Every coordinate produced by tensor quantization lies in the declared code interval. -/
theorem quantizeTensor_inRange (q : AffineQuantizer) (rnd : ℝ → ℤ)
    {s : Shape} (x : Tensor ℝ s) :
    q.CodesInRange (q.quantizeTensor rnd x) := by
  apply Tensor.forall_mapTensor (Tensor.forall_true x)
  intro a _
  exact q.quantize_mem rnd a

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

/-- Pointwise order is preserved by tensor quantization. -/
theorem quantizeTensor_mono (q : AffineQuantizer) (rnd : ℝ → ℤ) [NeuralValidRnd rnd]
    {s : Shape} {x y : Tensor ℝ s}
    (hxy : Tensor.Forall (fun ab : ℝ × ℝ => ab.1 ≤ ab.2) (Spec.zip x y)) :
    Tensor.Forall (fun ab : ℤ × ℤ => ab.1 ≤ ab.2)
      (Spec.zip (q.quantizeTensor rnd x) (q.quantizeTensor rnd y)) := by
  induction s with
  | scalar =>
      cases x with
      | scalar a =>
          cases y with
          | scalar b => exact q.quantize_mono rnd hxy
  | dim n inner ih =>
      cases x with
      | dim xs =>
          cases y with
          | dim ys =>
              intro i
              exact ih (hxy i)

/-- Every in-range integer code survives a dequantize/quantize round trip. -/
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

/-- An in-range code tensor survives pointwise dequantization and requantization exactly. -/
theorem quantizeTensor_dequantizeTensor (q : AffineQuantizer) (rnd : ℝ → ℤ)
    [NeuralValidRnd rnd] {s : Shape} {codes : Tensor ℤ s} (hcodes : q.CodesInRange codes) :
    q.quantizeTensor rnd (q.dequantizeTensor codes) = codes := by
  induction s with
  | scalar =>
      cases codes with
      | scalar code =>
          simp only [quantizeTensor, dequantizeTensor, mapTensor]
          rw [q.quantize_dequantize rnd hcodes.1 hcodes.2]
  | dim n inner ih =>
      cases codes with
      | dim values =>
          simp only [quantizeTensor, dequantizeTensor, mapTensor]
          congr 1
          funext i
          exact ih (hcodes i)

/-- Without saturation, nearest affine quantization has error at most half a quantization step. -/
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

/-- The half-step bound survives saturation whenever clipping is inactive. -/
theorem dequantize_quantize_error_le (q : AffineQuantizer) (rnd : ℝ → ℤ)
    [NeuralValidRndToNearest rnd] (x : ℝ)
    (hlo : q.qmin ≤ q.rawCode rnd x) (hhi : q.rawCode rnd x ≤ q.qmax) :
    abs (q.dequantize (q.quantize rnd x) - x) ≤ q.scale / 2 := by
  rw [quantize, q.clampCode_eq_self hlo hhi]
  exact q.dequantize_rawCode_error_le rnd x

/-- If no coordinate clips, every tensor reconstruction error is at most half a step. -/
theorem dequantizeTensor_quantizeTensor_error_le (q : AffineQuantizer) (rnd : ℝ → ℤ)
    [NeuralValidRndToNearest rnd] {s : Shape} {x : Tensor ℝ s}
    (hinactive : q.SaturationInactive rnd x) :
    Tensor.Forall (fun e : ℝ => abs e ≤ q.scale / 2)
      ((q.dequantizeTensor (q.quantizeTensor rnd x)).subSpec x) := by
  induction s with
  | scalar =>
      cases x with
      | scalar a =>
          exact q.dequantize_quantize_error_le rnd a hinactive.1 hinactive.2
  | dim n inner ih =>
      cases x with
      | dim values =>
          intro i
          exact ih (hinactive i)

end AffineQuantizer

end TorchLean.Floats.Quantization
