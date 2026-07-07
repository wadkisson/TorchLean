/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Optimization.Muon

/-!
# Muon Certificate Examples

Small compiled examples showing how downstream proofs should consume the proof-oriented Muon
certificate API. Runtime code configures Muon through `TorchLean.optim.runtimeMuon`; these examples
intentionally use the `TorchLean.optim.muon` proof namespace rather than the internal `Optim.Muon`
theorem names.
-/

@[expose] public section

namespace TorchLean
namespace Examples
namespace MuonCertificates

open Spec

/--
Using the QR checked backend, a positive-pivot proof gives a certified step; from that step we can
recover both the exact Gram certificate for the direction and the parameter-update equation.
-/
theorem qr_update_step_direction_has_exact_gram {m n : Nat}
    (lr momentum : ℝ) (buf params grads : optim.muon.MatrixTensor ℝ m n)
    (hpivots :
      optim.muon.HasPositiveQRPivots
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := optim.muon.qrOrthogonalizer (m := m) (n := n) } :
            optim.muon.State ℝ (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    ∃ direction : optim.muon.MatrixTensor ℝ m n,
      optim.muon.HasExactColumnGram direction ∧
      (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := optim.muon.qrOrthogonalizer (m := m) (n := n) } :
            optim.muon.State ℝ (.dim m (.dim n .scalar)))
          params grads).2 =
        _root_.Spec.Tensor.subSpec params (_root_.Spec.Tensor.scaleSpec direction lr) := by
  obtain ⟨direction, hstep⟩ :=
    optim.muon.update_has_exact_certified_step_qr lr momentum buf params grads hpivots
  exact ⟨direction,
    optim.muon.exactCertifiedStep_direction_has_exact_column_gram hstep,
    optim.muon.exactCertifiedStep_params_eq hstep⟩

/--
Using the residual-checked Newton-Schulz backend, a residual proof gives a certified approximate
step; from that step we can recover the residual-bound certificate and the parameter equation.
-/
theorem newtonSchulz_update_step_direction_has_approx_gram
    {α : Type} [Context α] {m n : Nat} {eps : α}
    (coeffs : optim.muon.NewtonSchulzCoeffs α) (steps : Nat)
    (lr momentum : α) (buf params grads : optim.muon.MatrixTensor α m n)
    (hresidual :
      optim.muon.ResidualApproxSuccess eps
        (optim.muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              optim.muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            optim.muon.State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    ∃ direction : optim.muon.MatrixTensor α m n,
      optim.muon.HasApproxColumnGram eps direction ∧
      (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              optim.muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            optim.muon.State α (.dim m (.dim n .scalar)))
          params grads).2 =
        _root_.Spec.Tensor.subSpec params (_root_.Spec.Tensor.scaleSpec direction lr) := by
  obtain ⟨direction, hstep⟩ :=
    optim.muon.update_has_approx_certified_step_newtonSchulz_checked
      coeffs steps lr momentum buf params grads hresidual
  exact ⟨direction,
    optim.muon.approxCertifiedStep_direction_has_approx_column_gram hstep,
    optim.muon.approxCertifiedStep_params_eq hstep⟩

/--
If the fresh momentum buffer is already exact-column-orthogonal and fixed by one Newton-Schulz
step, the fixed-point checked backend upgrades Newton-Schulz from an approximate residual-checked
path to an exact certified step.
-/
theorem newtonSchulz_fixed_update_step_direction_has_exact_gram
    {α : Type} [Context α] {m n : Nat}
    (coeffs : optim.muon.NewtonSchulzCoeffs α) (steps : Nat)
    (lr momentum : α) (buf params grads : optim.muon.MatrixTensor α m n)
    (hsuccess :
      (optim.muon.newtonSchulzFixedPointCheckedExactOrthogonalizer
        (α := α) (m := m) (n := n) coeffs steps).Success
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              optim.muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            optim.muon.State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    ∃ direction : optim.muon.MatrixTensor α m n,
      optim.muon.HasExactColumnGram direction ∧
      (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              optim.muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            optim.muon.State α (.dim m (.dim n .scalar)))
          params grads).2 =
        _root_.Spec.Tensor.subSpec params (_root_.Spec.Tensor.scaleSpec direction lr) := by
  obtain ⟨direction, hstep⟩ :=
    optim.muon.update_has_exact_certified_step_newtonSchulz_fixed_checked
      coeffs steps lr momentum buf params grads hsuccess
  exact ⟨direction,
    optim.muon.exactCertifiedStep_direction_has_exact_column_gram hstep,
    optim.muon.exactCertifiedStep_params_eq hstep⟩

/--
At an exact-column-orthogonal matrix, one real Newton-Schulz step acts by the scalar
`a + b + c`. This is the algebra behind the fixed-point shortcut below.
-/
theorem newtonSchulz_step_scales_exact_gram_matrix {m n : Nat}
    (coeffs : optim.muon.NewtonSchulzCoeffs ℝ)
    (Q : optim.muon.MatrixTensor ℝ m n)
    (hgram : optim.muon.HasExactColumnGram Q) :
    _root_.Optim.Muon.newtonSchulzStep coeffs Q =
      _root_.Spec.Tensor.scaleSpec Q (coeffs.a + coeffs.b + coeffs.c) :=
  optim.muon.newtonSchulzStep_eq_scale_sum_of_exact_column_gram coeffs Q hgram

/--
The same algebra shows exact column Gram is preserved when the coefficient sum squares to one,
including the sign-flip case.
-/
theorem newtonSchulz_step_preserves_exact_gram_when_sum_squares_one {m n : Nat}
    (coeffs : optim.muon.NewtonSchulzCoeffs ℝ)
    (Q : optim.muon.MatrixTensor ℝ m n)
    (hgram : optim.muon.HasExactColumnGram Q)
    (hsquare : (coeffs.a + coeffs.b + coeffs.c) *
        (coeffs.a + coeffs.b + coeffs.c) = 1) :
    optim.muon.HasExactColumnGram (_root_.Optim.Muon.newtonSchulzStep coeffs Q) :=
  optim.muon.newtonSchulzStep_hasExactColumnGram_of_exact_column_gram_of_sum_square_one
    coeffs Q hgram hsquare

/--
If the coefficient sum is only approximately square-one, the same one-step algebra gives the
entrywise residual bound used by the approximate Muon certificate.
-/
theorem newtonSchulz_step_has_approx_gram_from_coeff_square_error {m n : Nat}
    (coeffs : optim.muon.NewtonSchulzCoeffs ℝ)
    (Q : optim.muon.MatrixTensor ℝ m n) (eps : ℝ)
    (hgram : optim.muon.HasExactColumnGram Q)
    (herr :
      MathFunctions.abs
        ((coeffs.a + coeffs.b + coeffs.c) * (coeffs.a + coeffs.b + coeffs.c) - 1) ≤ eps)
    (heps : 0 ≤ eps) :
    optim.muon.HasApproxColumnGram eps (_root_.Optim.Muon.newtonSchulzStep coeffs Q) :=
  optim.muon.newtonSchulzStep_hasApproxColumnGram_of_exact_column_gram_of_sum_square_error
    coeffs Q eps hgram herr heps

/--
Over `ℝ`, the common coefficient condition `a + b + c = 1` means an already exact-column-orthogonal
fresh buffer is automatically a fixed point, so the Newton-Schulz Muon update is exactly certified.
-/
theorem newtonSchulz_coeff_sum_update_step_direction_has_exact_gram {m n : Nat}
    (coeffs : optim.muon.NewtonSchulzCoeffs ℝ) (steps : Nat)
    (lr momentum : ℝ) (buf params grads : optim.muon.MatrixTensor ℝ m n)
    (hgram :
      optim.muon.HasExactColumnGram
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              optim.muon.newtonSchulzOrthogonalizer (α := ℝ) (m := m) (n := n) coeffs steps } :
            optim.muon.State ℝ (.dim m (.dim n .scalar)))
          params grads).1.buf)
    (hsum : coeffs.a + coeffs.b + coeffs.c = 1) :
    ∃ direction : optim.muon.MatrixTensor ℝ m n,
      optim.muon.HasExactColumnGram direction ∧
      (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              optim.muon.newtonSchulzOrthogonalizer (α := ℝ) (m := m) (n := n) coeffs steps } :
            optim.muon.State ℝ (.dim m (.dim n .scalar)))
          params grads).2 =
        _root_.Spec.Tensor.subSpec params (_root_.Spec.Tensor.scaleSpec direction lr) := by
  obtain ⟨direction, hstep⟩ :=
    optim.muon.update_has_exact_certified_step_newtonSchulz_exact_gram_checked
      coeffs steps lr momentum buf params grads hgram hsum
  exact ⟨direction,
    optim.muon.exactCertifiedStep_direction_has_exact_column_gram hstep,
    optim.muon.exactCertifiedStep_params_eq hstep⟩

/--
The initialized QR theorem has the same proof shape as the stateful update theorem, but starts from
the state produced by `optim.muon.init`.
-/
theorem qr_initialized_step_direction_has_exact_gram {m n : Nat}
    (lr momentum : ℝ) (params grads : optim.muon.MatrixTensor ℝ m n)
    (hpivots :
      optim.muon.HasPositiveQRPivots
        (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum
            (optim.muon.qrOrthogonalizer (m := m) (n := n)) params)
          params grads).1.buf) :
    ∃ direction : optim.muon.MatrixTensor ℝ m n,
      optim.muon.HasExactColumnGram direction ∧
      (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum
            (optim.muon.qrOrthogonalizer (m := m) (n := n)) params)
          params grads).2 =
        _root_.Spec.Tensor.subSpec params (_root_.Spec.Tensor.scaleSpec direction lr) := by
  obtain ⟨direction, hstep⟩ :=
    optim.muon.init_has_exact_certified_step_qr lr momentum params grads hpivots
  exact ⟨direction,
    optim.muon.exactCertifiedStep_direction_has_exact_column_gram hstep,
    optim.muon.exactCertifiedStep_params_eq hstep⟩

/--
The same QR certificate also exposes the whole next-state equation, not only the direction
certificate.
-/
theorem qr_initialized_step_state_eq {m n : Nat}
    (lr momentum : ℝ) (params grads : optim.muon.MatrixTensor ℝ m n)
    (hpivots :
      optim.muon.HasPositiveQRPivots
        (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum
            (optim.muon.qrOrthogonalizer (m := m) (n := n)) params)
          params grads).1.buf) :
    (_root_.Optim.Muon.update
        (_root_.Optim.Muon.init lr momentum
          (optim.muon.qrOrthogonalizer (m := m) (n := n)) params)
        params grads).1 =
      { _root_.Optim.Muon.init lr momentum
          (optim.muon.qrOrthogonalizer (m := m) (n := n)) params with
        buf :=
          _root_.Optim.OptimizerUtils.updateMomentumBuf
            (_root_.Optim.Muon.init lr momentum
              (optim.muon.qrOrthogonalizer (m := m) (n := n)) params).buf
            (_root_.Optim.Muon.init lr momentum
              (optim.muon.qrOrthogonalizer (m := m) (n := n)) params).momentum
            grads } := by
  obtain ⟨_direction, hstep⟩ :=
    optim.muon.init_has_exact_certified_step_qr lr momentum params grads hpivots
  exact optim.muon.exactCertifiedStep_state_eq hstep

/--
The initialized residual-checked Newton-Schulz theorem is the training-path version: initialize the
Muon state, run one update, then consume the certified step.
-/
theorem newtonSchulz_initialized_step_direction_has_approx_gram
    {α : Type} [Context α] {m n : Nat} {eps : α}
    (coeffs : optim.muon.NewtonSchulzCoeffs α) (steps : Nat)
    (lr momentum : α) (params grads : optim.muon.MatrixTensor α m n)
    (hresidual :
      optim.muon.ResidualApproxSuccess eps
        (optim.muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
        (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum
            (optim.muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
            params)
          params grads).1.buf) :
    ∃ direction : optim.muon.MatrixTensor α m n,
      optim.muon.HasApproxColumnGram eps direction ∧
      (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum
            (optim.muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
            params)
          params grads).2 =
        _root_.Spec.Tensor.subSpec params (_root_.Spec.Tensor.scaleSpec direction lr) := by
  obtain ⟨direction, hstep⟩ :=
    optim.muon.init_has_approx_certified_step_newtonSchulz_checked
      coeffs steps lr momentum params grads hresidual
  exact ⟨direction,
    optim.muon.approxCertifiedStep_direction_has_approx_column_gram hstep,
    optim.muon.approxCertifiedStep_params_eq hstep⟩

/--
The initialized residual-checked Newton-Schulz certificate also exposes the whole next-state
equation.
-/
theorem newtonSchulz_initialized_step_state_eq
    {α : Type} [Context α] {m n : Nat} {eps : α}
    (coeffs : optim.muon.NewtonSchulzCoeffs α) (steps : Nat)
    (lr momentum : α) (params grads : optim.muon.MatrixTensor α m n)
    (hresidual :
      optim.muon.ResidualApproxSuccess eps
        (optim.muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
        (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum
            (optim.muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
            params)
          params grads).1.buf) :
    (_root_.Optim.Muon.update
        (_root_.Optim.Muon.init lr momentum
          (optim.muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
          params)
        params grads).1 =
      { _root_.Optim.Muon.init lr momentum
          (optim.muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
          params with
        buf :=
          _root_.Optim.OptimizerUtils.updateMomentumBuf
            (_root_.Optim.Muon.init lr momentum
              (optim.muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
              params).buf
            (_root_.Optim.Muon.init lr momentum
              (optim.muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
              params).momentum
            grads } := by
  obtain ⟨_direction, hstep⟩ :=
    optim.muon.init_has_approx_certified_step_newtonSchulz_checked
      coeffs steps lr momentum params grads hresidual
  exact optim.muon.approxCertifiedStep_state_eq hstep

end MuonCertificates
end Examples
end TorchLean
