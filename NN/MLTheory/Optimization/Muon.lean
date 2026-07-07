/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Optimization.OptimizerLaws
public import NN.Proofs.Tensor.Basic.FactorizationsOrthonormal
public import NN.Proofs.Tensor.Basic.LinearAlgebra

/-!
# Muon Orthogonalization Contracts

Muon is more than momentum with a different name: after the momentum buffer is updated, the
parameter direction is supposed to be an orthogonalized matrix direction.  The runtime optimizer
therefore takes an explicit orthogonalizer backend, while this file records the proof contract that
such a backend should satisfy.

There are two useful levels:

* `ExactMatrixOrthogonalizer` says the backend returns a matrix `Q` with `QᵀQ = I`.
* `ApproxMatrixOrthogonalizer eps` says the Gram residual `QᵀQ - I` is entrywise bounded by `eps`.

The theorems below connect those contracts to the executable Muon step: if the backend satisfies one
of these contracts, the direction used in the parameter update is certified at the same level.
-/

@[expose] public section

namespace Optim

open Spec
open Tensor

namespace Muon

variable {α : Type} [Context α]

/-- A matrix-shaped TorchLean tensor. -/
abbrev MatrixTensor (α : Type) (m n : Nat) :=
  Tensor α (.dim m (.dim n .scalar))

/-- The column Gram matrix `QᵀQ`. -/
def columnGram {m n : Nat} (Q : MatrixTensor α m n) :
    MatrixTensor α n n :=
  matMulSpec (Spec.Tensor.matrixTransposeSpec Q) Q

/-- Exact column orthogonality for a matrix-shaped update direction. -/
def HasExactColumnGram {m n : Nat} (Q : MatrixTensor α m n) : Prop :=
  columnGram Q = identityTensorSpec n

/-- Exact column Gram means exactly `QᵀQ = I`. -/
theorem hasExactColumnGram_iff {m n : Nat} (Q : MatrixTensor α m n) :
    HasExactColumnGram Q ↔ columnGram Q = identityTensorSpec n :=
  Iff.rfl

/-- A backend exactly orthogonalizes one specified momentum buffer. -/
def ExactOrthogonalizesBuffer {m n : Nat}
    (orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar)))
    (buffer : MatrixTensor α m n) : Prop :=
  HasExactColumnGram (orthogonalizer.apply buffer)

/-- One-buffer exact orthogonalization is exact column Gram for the backend output. -/
theorem exactOrthogonalizesBuffer_iff {m n : Nat}
    (orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar)))
    (buffer : MatrixTensor α m n) :
    ExactOrthogonalizesBuffer orthogonalizer buffer ↔
      HasExactColumnGram (orthogonalizer.apply buffer) :=
  Iff.rfl

/-- Residual matrix `QᵀQ - I`, used for approximate orthogonalization certificates. -/
def columnGramResidual {m n : Nat} (Q : MatrixTensor α m n) :
    MatrixTensor α n n :=
  subSpec (columnGram Q) (identityTensorSpec n)

/-- Expanded form of the column-Gram residual `QᵀQ - I`. -/
theorem columnGramResidual_eq {m n : Nat} (Q : MatrixTensor α m n) :
    columnGramResidual Q = subSpec (columnGram Q) (identityTensorSpec n) := by
  rfl

/--
Entrywise approximate column orthogonality.

For an exact backend use `HasExactColumnGram`. For Newton-Schulz or CUDA implementations, this is
the certificate shape we want the backend to establish or export: every entry of `QᵀQ - I` is
bounded by `eps`.
-/
def HasApproxColumnGram {m n : Nat} (eps : α) (Q : MatrixTensor α m n) : Prop :=
  ∀ i : Fin n, ∀ j : Fin n,
    MathFunctions.abs (get2 (columnGramResidual Q) i j) ≤ eps

/-- Approximate column orthogonality is exactly the entrywise residual bound. -/
theorem hasApproxColumnGram_iff {m n : Nat} (eps : α) (Q : MatrixTensor α m n) :
    HasApproxColumnGram eps Q ↔
      ∀ i : Fin n, ∀ j : Fin n,
        MathFunctions.abs (get2 (columnGramResidual Q) i j) ≤ eps :=
  Iff.rfl

/-- A backend approximately orthogonalizes one specified momentum buffer. -/
def ApproxOrthogonalizesBuffer {m n : Nat} (eps : α)
    (orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar)))
    (buffer : MatrixTensor α m n) : Prop :=
  HasApproxColumnGram eps (orthogonalizer.apply buffer)

/-- One-buffer approximate orthogonalization is the residual bound for the backend output. -/
theorem approxOrthogonalizesBuffer_iff {m n : Nat} (eps : α)
    (orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar)))
    (buffer : MatrixTensor α m n) :
    ApproxOrthogonalizesBuffer eps orthogonalizer buffer ↔
      HasApproxColumnGram eps (orthogonalizer.apply buffer) :=
  Iff.rfl

/--
An exact matrix Muon orthogonalizer maps every momentum buffer to a direction whose columns have
Gram matrix `I`.
-/
def ExactMatrixOrthogonalizer {m n : Nat}
    (orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar))) : Prop :=
  ∀ buffer : MatrixTensor α m n, HasExactColumnGram (orthogonalizer.apply buffer)

/-- A matrix exact backend returns exact column-Gram directions for every buffer. -/
theorem exactMatrixOrthogonalizer_iff {m n : Nat}
    (orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar))) :
    ExactMatrixOrthogonalizer orthogonalizer ↔
      ∀ buffer : MatrixTensor α m n, HasExactColumnGram (orthogonalizer.apply buffer) :=
  Iff.rfl

/--
An approximate matrix Muon orthogonalizer maps every momentum buffer to a direction whose Gram
residual is entrywise bounded by `eps`.
-/
def ApproxMatrixOrthogonalizer {m n : Nat} (eps : α)
    (orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar))) : Prop :=
  ∀ buffer : MatrixTensor α m n, HasApproxColumnGram eps (orthogonalizer.apply buffer)

/-- A matrix approximate backend returns residual-bounded directions for every buffer. -/
theorem approxMatrixOrthogonalizer_iff {m n : Nat} (eps : α)
    (orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar))) :
    ApproxMatrixOrthogonalizer eps orthogonalizer ↔
      ∀ buffer : MatrixTensor α m n, HasApproxColumnGram eps (orthogonalizer.apply buffer) :=
  Iff.rfl

/-! ## Certified backend records -/

/--
Unconditionally certified exact Muon backend.

Use this when the orthogonalizer is known to return an exact `QᵀQ = I` direction for every buffer
of a fixed matrix shape.
-/
structure ExactCertifiedOrthogonalizer (α : Type) [Context α] (m n : Nat) where
  /-- Executable orthogonalizer used by Muon. -/
  orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar))
  /-- The backend returns an exactly orthogonalized direction for every buffer. -/
  certified : ExactMatrixOrthogonalizer orthogonalizer

/--
Unconditionally certified approximate Muon backend.

Use this when the orthogonalizer is known to return a direction whose Gram residual is bounded by
`eps` for every buffer of a fixed matrix shape.
-/
structure ApproxCertifiedOrthogonalizer (α : Type) [Context α] (m n : Nat) (eps : α) where
  /-- Executable orthogonalizer used by Muon. -/
  orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar))
  /-- The backend returns an approximately orthogonalized direction for every buffer. -/
  certified : ApproxMatrixOrthogonalizer eps orthogonalizer

/--
Checked exact Muon backend with a per-buffer success predicate.

This is the practical interface for algorithms whose correctness has preconditions on the matrix
being orthogonalized. The QR backend below is the first instance: its success predicate is positive
executable `R` pivots.
-/
structure CheckedExactOrthogonalizer (α : Type) [Context α] (m n : Nat) where
  /-- Executable orthogonalizer used by Muon. -/
  orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar))
  /-- Backend-specific success/certification condition for an input buffer. -/
  Success : MatrixTensor α m n → Prop
  /-- Whenever `Success buffer` holds, the backend exactly orthogonalizes that buffer. -/
  certified : ∀ buffer : MatrixTensor α m n,
    Success buffer → ExactOrthogonalizesBuffer orthogonalizer buffer

/--
Checked approximate Muon backend with a per-buffer success predicate.

This is the intended proof shape for Newton-Schulz/CUDA-style backends: the kernel may be fast and
approximate, but the exported proof or checker must establish `Success buffer`, which then gives the
entrywise `QᵀQ - I` bound.
-/
structure CheckedApproxOrthogonalizer (α : Type) [Context α] (m n : Nat) (eps : α) where
  /-- Executable orthogonalizer used by Muon. -/
  orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar))
  /-- Backend-specific success/certification condition for an input buffer. -/
  Success : MatrixTensor α m n → Prop
  /-- Whenever `Success buffer` holds, the backend approximately orthogonalizes that buffer. -/
  certified : ∀ buffer : MatrixTensor α m n,
    Success buffer → ApproxOrthogonalizesBuffer eps orthogonalizer buffer

/-! ## Newton-Schulz-shaped approximate backends -/

/--
Coefficients for the odd Newton-Schulz polynomial used by Muon-style orthogonalization.

The column-oriented shape is `X ↦ aX + bX(XᵀX) + cX(XᵀX)^2`, matching the `QᵀQ`
certificate used below. TorchLean keeps the coefficients explicit so experiments and
backend-specific proofs can choose the polynomial they actually use.
-/
structure NewtonSchulzCoeffs (α : Type) where
  /-- Linear coefficient. -/
  a : α
  /-- Cubic coefficient. -/
  b : α
  /-- Quintic coefficient. -/
  c : α

/-- Left Gram matrix `XXᵀ`, useful for row-oriented rectangular Newton-Schulz updates. -/
def leftGram {m n : Nat} (X : MatrixTensor α m n) : MatrixTensor α m m :=
  matMulSpec X (Spec.Tensor.matrixTransposeSpec X)

/-- Right/column Gram matrix `XᵀX`, matching TorchLean's column-orthogonality certificate. -/
def rightGram {m n : Nat} (X : MatrixTensor α m n) : MatrixTensor α n n :=
  matMulSpec (Spec.Tensor.matrixTransposeSpec X) X

/-- One row-oriented Newton-Schulz polynomial step using `XXᵀ`. -/
def newtonSchulzLeftStep {m n : Nat} (coeffs : NewtonSchulzCoeffs α)
    (X : MatrixTensor α m n) : MatrixTensor α m n :=
  let G := leftGram X
  let GX := matMulSpec G X
  let G2X := matMulSpec G GX
  addSpec (addSpec (scaleSpec X coeffs.a) (scaleSpec GX coeffs.b)) (scaleSpec G2X coeffs.c)

/-- One column-oriented Newton-Schulz polynomial step using `XᵀX`. -/
def newtonSchulzStep {m n : Nat} (coeffs : NewtonSchulzCoeffs α)
    (X : MatrixTensor α m n) : MatrixTensor α m n :=
  let G := rightGram X
  let XG := matMulSpec X G
  let XG2 := matMulSpec XG G
  addSpec (addSpec (scaleSpec X coeffs.a) (scaleSpec XG coeffs.b)) (scaleSpec XG2 coeffs.c)

/-- Iterate the row-oriented Newton-Schulz polynomial step a fixed number of times. -/
def newtonSchulzLeftIter {m n : Nat} (coeffs : NewtonSchulzCoeffs α) :
    Nat → MatrixTensor α m n → MatrixTensor α m n
  | 0, X => X
  | steps + 1, X => newtonSchulzLeftIter coeffs steps (newtonSchulzLeftStep coeffs X)

/-- Iterate the column-oriented Newton-Schulz polynomial step a fixed number of times. -/
def newtonSchulzIter {m n : Nat} (coeffs : NewtonSchulzCoeffs α) :
    Nat → MatrixTensor α m n → MatrixTensor α m n
  | 0, X => X
  | steps + 1, X => newtonSchulzIter coeffs steps (newtonSchulzStep coeffs X)

/-- Row-oriented Newton-Schulz-shaped orthogonalizer backend. -/
def newtonSchulzLeftOrthogonalizer {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) :
    Orthogonalizer α (.dim m (.dim n .scalar)) :=
  { apply := fun buffer => newtonSchulzLeftIter coeffs steps buffer }

/-- Column-oriented Newton-Schulz-shaped orthogonalizer backend. -/
def newtonSchulzOrthogonalizer {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) :
    Orthogonalizer α (.dim m (.dim n .scalar)) :=
  { apply := fun buffer => newtonSchulzIter coeffs steps buffer }

/--
Residual-check success predicate for approximate Muon backends.

This is the lightest sound checker boundary: after a backend returns a direction, prove or check that
the direction's Gram residual is bounded by `eps`.
-/
def ResidualApproxSuccess {m n : Nat} (eps : α)
    (orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar)))
    (buffer : MatrixTensor α m n) : Prop :=
  HasApproxColumnGram eps (orthogonalizer.apply buffer)

/-- The residual-check success predicate is exactly the approximate Gram certificate. -/
theorem residualApproxSuccess_iff {m n : Nat} (eps : α)
    (orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar)))
    (buffer : MatrixTensor α m n) :
    ResidualApproxSuccess eps orthogonalizer buffer ↔
      HasApproxColumnGram eps (orthogonalizer.apply buffer) :=
  Iff.rfl

/--
Turn any orthogonalizer into a checked approximate backend by using the Gram-residual bound itself
as the success predicate.
-/
def residualCheckedApproxOrthogonalizer {m n : Nat} (eps : α)
    (orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar))) :
    CheckedApproxOrthogonalizer α m n eps :=
  { orthogonalizer := orthogonalizer
    Success := ResidualApproxSuccess eps orthogonalizer
    certified := fun _ hresidual => hresidual }

/-- The success predicate of a residual-checked backend is the approximate Gram certificate. -/
theorem residualCheckedApproxOrthogonalizer_success_iff {m n : Nat} (eps : α)
    (orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar)))
    (buffer : MatrixTensor α m n) :
    (residualCheckedApproxOrthogonalizer eps orthogonalizer).Success buffer ↔
      HasApproxColumnGram eps (orthogonalizer.apply buffer) :=
  Iff.rfl

/--
Newton-Schulz packaged as a checked approximate backend.

The backend is the explicit Newton-Schulz tensor program, and the success predicate is the
post-check that its returned direction satisfies the requested Gram-residual bound.
-/
def newtonSchulzResidualCheckedOrthogonalizer {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) (eps : α) :
    CheckedApproxOrthogonalizer α m n eps :=
  residualCheckedApproxOrthogonalizer eps
    (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)

/-! ### Newton-Schulz reduction lemmas -/

/-- Expanded form of the left Gram matrix. -/
theorem leftGram_eq {m n : Nat} (X : MatrixTensor α m n) :
    leftGram X = matMulSpec X (Spec.Tensor.matrixTransposeSpec X) := by
  rfl

/-- Expanded form of the right/column Gram matrix. -/
theorem rightGram_eq {m n : Nat} (X : MatrixTensor α m n) :
    rightGram X = matMulSpec (Spec.Tensor.matrixTransposeSpec X) X := by
  rfl

/-- Expanded form of one row-oriented Newton-Schulz step. -/
theorem newtonSchulzLeftStep_eq {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (X : MatrixTensor α m n) :
    newtonSchulzLeftStep coeffs X =
      let G := leftGram X
      let GX := matMulSpec G X
      let G2X := matMulSpec G GX
      addSpec (addSpec (scaleSpec X coeffs.a) (scaleSpec GX coeffs.b))
        (scaleSpec G2X coeffs.c) := by
  rfl

/-- Expanded form of one column-oriented Newton-Schulz step. -/
theorem newtonSchulzStep_eq {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (X : MatrixTensor α m n) :
    newtonSchulzStep coeffs X =
      let G := rightGram X
      let XG := matMulSpec X G
      let XG2 := matMulSpec XG G
      addSpec (addSpec (scaleSpec X coeffs.a) (scaleSpec XG coeffs.b))
        (scaleSpec XG2 coeffs.c) := by
  rfl

/-- Zero row-oriented Newton-Schulz iterations return the input. -/
theorem newtonSchulzLeftIter_zero {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (X : MatrixTensor α m n) :
    newtonSchulzLeftIter coeffs 0 X = X := by
  rfl

/-- Successor row-oriented Newton-Schulz iteration. -/
theorem newtonSchulzLeftIter_succ {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) (X : MatrixTensor α m n) :
    newtonSchulzLeftIter coeffs (steps + 1) X =
      newtonSchulzLeftIter coeffs steps (newtonSchulzLeftStep coeffs X) := by
  rfl

/-- Zero column-oriented Newton-Schulz iterations return the input. -/
theorem newtonSchulzIter_zero {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (X : MatrixTensor α m n) :
    newtonSchulzIter coeffs 0 X = X := by
  rfl

/-- Successor column-oriented Newton-Schulz iteration. -/
theorem newtonSchulzIter_succ {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) (X : MatrixTensor α m n) :
    newtonSchulzIter coeffs (steps + 1) X =
      newtonSchulzIter coeffs steps (newtonSchulzStep coeffs X) := by
  rfl

/-- Applying the row-oriented Newton-Schulz orthogonalizer runs the row-oriented iterator. -/
theorem newtonSchulzLeftOrthogonalizer_apply {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) (buffer : MatrixTensor α m n) :
    (newtonSchulzLeftOrthogonalizer (α := α) (m := m) (n := n) coeffs steps).apply buffer =
      newtonSchulzLeftIter coeffs steps buffer := by
  rfl

/-- Zero row-oriented Newton-Schulz steps make the orthogonalizer the identity on buffers. -/
theorem newtonSchulzLeftOrthogonalizer_zero_apply {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (buffer : MatrixTensor α m n) :
    (newtonSchulzLeftOrthogonalizer (α := α) (m := m) (n := n) coeffs 0).apply buffer =
      buffer := by
  rfl

/-- Applying the column-oriented Newton-Schulz orthogonalizer runs the column-oriented iterator. -/
theorem newtonSchulzOrthogonalizer_apply {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) (buffer : MatrixTensor α m n) :
    (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps).apply buffer =
      newtonSchulzIter coeffs steps buffer := by
  rfl

/-- Zero column-oriented Newton-Schulz steps make the orthogonalizer the identity on buffers. -/
theorem newtonSchulzOrthogonalizer_zero_apply {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (buffer : MatrixTensor α m n) :
    (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs 0).apply buffer =
      buffer := by
  rfl

/--
With zero Newton-Schulz steps, the Muon backend leaves the fresh momentum buffer unchanged, so the
stored buffer agrees with momentum SGD's stored buffer.
-/
theorem update_newtonSchulz_zero_buffer_eq_momentumSGD {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α)
    (lr momentum : α) (buf params grads : MatrixTensor α m n) :
    (update
        ({ lr := lr, momentum := momentum, buf := buf,
           orthogonalizer := newtonSchulzOrthogonalizer
            (α := α) (m := m) (n := n) coeffs 0 } :
          State α (.dim m (.dim n .scalar)))
        params grads).1.buf =
      (MomentumSGD.update
        ({ lr := lr, momentum := momentum, buf := buf } :
          MomentumSGD.State α (.dim m (.dim n .scalar)))
        params grads).1.buf := by
  rfl

/--
With zero Newton-Schulz steps, Muon's parameter update is exactly momentum SGD's parameter update.

This is a backend sanity law: enabling the Newton-Schulz-shaped path with no iterations cannot
silently change the optimizer semantics.
-/
theorem update_newtonSchulz_zero_params_eq_momentumSGD {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α)
    (lr momentum : α) (buf params grads : MatrixTensor α m n) :
    (update
        ({ lr := lr, momentum := momentum, buf := buf,
           orthogonalizer := newtonSchulzOrthogonalizer
            (α := α) (m := m) (n := n) coeffs 0 } :
          State α (.dim m (.dim n .scalar)))
        params grads).2 =
      (MomentumSGD.update
        ({ lr := lr, momentum := momentum, buf := buf } :
          MomentumSGD.State α (.dim m (.dim n .scalar)))
        params grads).2 := by
  rfl

/--
Starting from initialized states, a zero-step Newton-Schulz Muon backend stores the same next buffer
as momentum SGD.
-/
theorem init_newtonSchulz_zero_update_buffer_eq_momentumSGD {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α)
    (lr momentum : α) (params grads : MatrixTensor α m n) :
    (update
        (init lr momentum
          (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs 0)
          params)
        params grads).1.buf =
      (MomentumSGD.update (MomentumSGD.init lr momentum params) params grads).1.buf := by
  rfl

/--
Starting from initialized states, a zero-step Newton-Schulz Muon backend has the same parameter
update as momentum SGD.
-/
theorem init_newtonSchulz_zero_update_params_eq_momentumSGD {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α)
    (lr momentum : α) (params grads : MatrixTensor α m n) :
    (update
        (init lr momentum
          (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs 0)
          params)
        params grads).2 =
      (MomentumSGD.update (MomentumSGD.init lr momentum params) params grads).2 := by
  rfl

/--
The success predicate of the Newton-Schulz residual-checked backend is the approximate Gram
certificate for the Newton-Schulz output.
-/
theorem newtonSchulzResidualChecked_success_iff {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) (eps : α)
    (buffer : MatrixTensor α m n) :
    (newtonSchulzResidualCheckedOrthogonalizer
      (α := α) (m := m) (n := n) coeffs steps eps).Success buffer ↔
      HasApproxColumnGram eps
        ((newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps).apply buffer) :=
  Iff.rfl

/--
For a zero-step Newton-Schulz backend, the residual checker is checking the raw momentum buffer.
-/
theorem newtonSchulzResidualChecked_zero_success_iff {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (eps : α) (buffer : MatrixTensor α m n) :
    (newtonSchulzResidualCheckedOrthogonalizer
      (α := α) (m := m) (n := n) coeffs 0 eps).Success buffer ↔
      HasApproxColumnGram eps buffer :=
  Iff.rfl

/--
A buffer is a fixed point of one column-oriented Newton-Schulz step.

This is a deliberately local condition.  It does not assert convergence from arbitrary inputs; it
records the exact algebraic fact needed when a backend has already reached a stable direction.
-/
def NewtonSchulzFixedPoint {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (buffer : MatrixTensor α m n) : Prop :=
  newtonSchulzStep coeffs buffer = buffer

/-- The fixed-point predicate is exactly equality with one Newton-Schulz step. -/
theorem newtonSchulzFixedPoint_iff {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (buffer : MatrixTensor α m n) :
    NewtonSchulzFixedPoint coeffs buffer ↔ newtonSchulzStep coeffs buffer = buffer :=
  Iff.rfl

/--
If one Newton-Schulz step fixes a buffer, then any finite number of Newton-Schulz iterations fixes
the same buffer.
-/
theorem newtonSchulzIter_fixed_of_step_fixed {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) (buffer : MatrixTensor α m n)
    (hfixed : NewtonSchulzFixedPoint coeffs buffer) :
    newtonSchulzIter coeffs steps buffer = buffer := by
  induction steps with
  | zero =>
      rfl
  | succ steps ih =>
      rw [newtonSchulzIter_succ]
      rw [hfixed]
      exact ih

/-- A Newton-Schulz fixed point is returned unchanged by the corresponding orthogonalizer. -/
theorem newtonSchulzOrthogonalizer_fixed_apply {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) (buffer : MatrixTensor α m n)
    (hfixed : NewtonSchulzFixedPoint coeffs buffer) :
    (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps).apply buffer =
      buffer := by
  exact newtonSchulzIter_fixed_of_step_fixed coeffs steps buffer hfixed

/--
If a buffer already has exact column Gram and is fixed by one Newton-Schulz step, then the
finite-iteration Newton-Schulz backend exactly orthogonalizes that buffer.
-/
theorem newtonSchulzFixedPoint_exactOrthogonalizesBuffer {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) (buffer : MatrixTensor α m n)
    (hgram : HasExactColumnGram buffer)
    (hfixed : NewtonSchulzFixedPoint coeffs buffer) :
    ExactOrthogonalizesBuffer
      (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps) buffer := by
  unfold ExactOrthogonalizesBuffer
  rw [newtonSchulzOrthogonalizer_fixed_apply coeffs steps buffer hfixed]
  exact hgram

/--
Newton-Schulz packaged as a checked exact backend for already-stable directions.

The success predicate says that the input buffer already has exact column Gram and is a fixed point
of one Newton-Schulz step.  Under that explicit condition, every finite Newton-Schulz iteration
returns the same exact-column-orthogonal direction.
-/
def newtonSchulzFixedPointCheckedExactOrthogonalizer {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) :
    CheckedExactOrthogonalizer α m n :=
  { orthogonalizer := newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps
    Success := fun buffer => HasExactColumnGram buffer ∧ NewtonSchulzFixedPoint coeffs buffer
    certified := fun buffer hsuccess =>
      newtonSchulzFixedPoint_exactOrthogonalizesBuffer coeffs steps buffer
        hsuccess.1 hsuccess.2 }

/-- Success for the fixed-point Newton-Schulz exact backend is the explicit Gram/fixed predicate. -/
theorem newtonSchulzFixedPointCheckedExact_success_iff {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) (buffer : MatrixTensor α m n) :
    (newtonSchulzFixedPointCheckedExactOrthogonalizer
      (α := α) (m := m) (n := n) coeffs steps).Success buffer ↔
      HasExactColumnGram buffer ∧ NewtonSchulzFixedPoint coeffs buffer :=
  Iff.rfl

/-- A direction is certified when it is exactly the backend output and has exact column Gram `I`. -/
structure ExactCertifiedDirection {m n : Nat}
    (orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar)))
    (buffer direction : MatrixTensor α m n) : Prop where
  /-- The direction is the one returned by the Muon orthogonalizer. -/
  direction_eq : direction = orthogonalizer.apply buffer
  /-- The returned direction has exact column Gram `I`. -/
  exact_column_gram : HasExactColumnGram direction

/--
A direction is approximately certified when it is exactly the backend output and its Gram residual
is entrywise bounded by `eps`.
-/
structure ApproxCertifiedDirection {m n : Nat} (eps : α)
    (orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar)))
    (buffer direction : MatrixTensor α m n) : Prop where
  /-- The direction is the one returned by the Muon orthogonalizer. -/
  direction_eq : direction = orthogonalizer.apply buffer
  /-- The returned direction has an entrywise Gram residual bound. -/
  approx_column_gram : HasApproxColumnGram eps direction

/--
An exact certified Muon step packages the whole update fact: the backend-produced direction is
exactly certified, the next state is the old state with the fresh momentum buffer installed, and the
parameter update subtracts the scaled certified direction.
-/
structure ExactCertifiedStep {m n : Nat}
    (state : State α (.dim m (.dim n .scalar)))
    (params grads direction : MatrixTensor α m n) : Prop where
  /-- The update direction is exactly certified for the fresh momentum buffer. -/
  direction_cert :
    ExactCertifiedDirection state.orthogonalizer (update state params grads).1.buf direction
  /-- The next optimizer state only replaces the momentum buffer. -/
  state_eq :
    (update state params grads).1 =
      { state with buf := OptimizerUtils.updateMomentumBuf state.buf state.momentum grads }
  /-- The parameter update subtracts the scaled certified direction. -/
  params_eq :
    (update state params grads).2 = subSpec params (scaleSpec direction state.lr)

/--
An approximate certified Muon step packages the same update fact as `ExactCertifiedStep`, with the
direction carrying an entrywise Gram-residual bound.
-/
structure ApproxCertifiedStep {m n : Nat} (eps : α)
    (state : State α (.dim m (.dim n .scalar)))
    (params grads direction : MatrixTensor α m n) : Prop where
  /-- The update direction is approximately certified for the fresh momentum buffer. -/
  direction_cert :
    ApproxCertifiedDirection eps state.orthogonalizer (update state params grads).1.buf direction
  /-- The next optimizer state only replaces the momentum buffer. -/
  state_eq :
    (update state params grads).1 =
      { state with buf := OptimizerUtils.updateMomentumBuf state.buf state.momentum grads }
  /-- The parameter update subtracts the scaled certified direction. -/
  params_eq :
    (update state params grads).2 = subSpec params (scaleSpec direction state.lr)

/-- An exact orthogonalizer certifies its own output direction. -/
theorem exact_certified_direction_of_orthogonalizer {m n : Nat}
    {orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar))}
    (horth : ExactMatrixOrthogonalizer orthogonalizer)
    (buffer : MatrixTensor α m n) :
    ExactCertifiedDirection orthogonalizer buffer (orthogonalizer.apply buffer) := by
  refine ⟨rfl, ?_⟩
  exact horth buffer

/-- An approximate orthogonalizer certifies its own output direction. -/
theorem approx_certified_direction_of_orthogonalizer {m n : Nat} {eps : α}
    {orthogonalizer : Orthogonalizer α (.dim m (.dim n .scalar))}
    (horth : ApproxMatrixOrthogonalizer eps orthogonalizer)
    (buffer : MatrixTensor α m n) :
    ApproxCertifiedDirection eps orthogonalizer buffer (orthogonalizer.apply buffer) := by
  refine ⟨rfl, ?_⟩
  exact horth buffer

/--
A residual proof for the Newton-Schulz output directly gives the approximate direction certificate
for that buffer.
-/
theorem newtonSchulz_residual_certifies_direction {m n : Nat} {eps : α}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) (buffer : MatrixTensor α m n)
    (hresidual :
      ResidualApproxSuccess eps
        (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
        buffer) :
    ApproxCertifiedDirection eps
      (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
      buffer
      ((newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps).apply buffer) := by
  exact ⟨rfl, hresidual⟩

/-- A certified exact direction for the fresh buffer gives a certified exact Muon step. -/
theorem exact_certified_step_of_direction {m n : Nat}
    (state : State α (.dim m (.dim n .scalar)))
    (params grads direction : MatrixTensor α m n)
    (hdir :
      ExactCertifiedDirection state.orthogonalizer
        (update state params grads).1.buf direction) :
    ExactCertifiedStep state params grads direction := by
  refine ⟨hdir, rfl, ?_⟩
  rw [hdir.direction_eq]
  rfl

/-- A certified approximate direction for the fresh buffer gives a certified approximate Muon step. -/
theorem approx_certified_step_of_direction {m n : Nat} {eps : α}
    (state : State α (.dim m (.dim n .scalar)))
    (params grads direction : MatrixTensor α m n)
    (hdir :
      ApproxCertifiedDirection eps state.orthogonalizer
        (update state params grads).1.buf direction) :
    ApproxCertifiedStep eps state params grads direction := by
  refine ⟨hdir, rfl, ?_⟩
  rw [hdir.direction_eq]
  rfl

/-- An exact certified step gives the exact column-Gram fact for its certified direction. -/
theorem exactCertifiedStep_direction_has_exact_column_gram {m n : Nat}
    {state : State α (.dim m (.dim n .scalar))}
    {params grads direction : MatrixTensor α m n}
    (hstep : ExactCertifiedStep state params grads direction) :
    HasExactColumnGram direction :=
  hstep.direction_cert.exact_column_gram

/-- An approximate certified step gives the residual bound for its certified direction. -/
theorem approxCertifiedStep_direction_has_approx_column_gram {m n : Nat} {eps : α}
    {state : State α (.dim m (.dim n .scalar))}
    {params grads direction : MatrixTensor α m n}
    (hstep : ApproxCertifiedStep eps state params grads direction) :
    HasApproxColumnGram eps direction :=
  hstep.direction_cert.approx_column_gram

/--
An exact certified step identifies its certified direction with the backend output on the fresh
momentum buffer.
-/
theorem exactCertifiedStep_direction_eq {m n : Nat}
    {state : State α (.dim m (.dim n .scalar))}
    {params grads direction : MatrixTensor α m n}
    (hstep : ExactCertifiedStep state params grads direction) :
    direction = state.orthogonalizer.apply (update state params grads).1.buf :=
  hstep.direction_cert.direction_eq

/--
An approximate certified step identifies its certified direction with the backend output on the
fresh momentum buffer.
-/
theorem approxCertifiedStep_direction_eq {m n : Nat} {eps : α}
    {state : State α (.dim m (.dim n .scalar))}
    {params grads direction : MatrixTensor α m n}
    (hstep : ApproxCertifiedStep eps state params grads direction) :
    direction = state.orthogonalizer.apply (update state params grads).1.buf :=
  hstep.direction_cert.direction_eq

/--
An exact certified step gives the exact column-Gram fact for the actual backend output on the fresh
momentum buffer.
-/
theorem exactCertifiedStep_backend_output_has_exact_column_gram {m n : Nat}
    {state : State α (.dim m (.dim n .scalar))}
    {params grads direction : MatrixTensor α m n}
    (hstep : ExactCertifiedStep state params grads direction) :
    HasExactColumnGram (state.orthogonalizer.apply (update state params grads).1.buf) := by
  rw [← hstep.direction_cert.direction_eq]
  exact hstep.direction_cert.exact_column_gram

/--
An approximate certified step gives the residual bound for the actual backend output on the fresh
momentum buffer.
-/
theorem approxCertifiedStep_backend_output_has_approx_column_gram {m n : Nat} {eps : α}
    {state : State α (.dim m (.dim n .scalar))}
    {params grads direction : MatrixTensor α m n}
    (hstep : ApproxCertifiedStep eps state params grads direction) :
    HasApproxColumnGram eps (state.orthogonalizer.apply (update state params grads).1.buf) := by
  rw [← hstep.direction_cert.direction_eq]
  exact hstep.direction_cert.approx_column_gram

/-- An exact certified step gives the next-state equation for the Muon update. -/
theorem exactCertifiedStep_state_eq {m n : Nat}
    {state : State α (.dim m (.dim n .scalar))}
    {params grads direction : MatrixTensor α m n}
    (hstep : ExactCertifiedStep state params grads direction) :
    (update state params grads).1 =
      { state with buf := OptimizerUtils.updateMomentumBuf state.buf state.momentum grads } :=
  hstep.state_eq

/-- An approximate certified step gives the next-state equation for the Muon update. -/
theorem approxCertifiedStep_state_eq {m n : Nat} {eps : α}
    {state : State α (.dim m (.dim n .scalar))}
    {params grads direction : MatrixTensor α m n}
    (hstep : ApproxCertifiedStep eps state params grads direction) :
    (update state params grads).1 =
      { state with buf := OptimizerUtils.updateMomentumBuf state.buf state.momentum grads } :=
  hstep.state_eq

/-- An exact certified step gives the parameter-update equation for the certified direction. -/
theorem exactCertifiedStep_params_eq {m n : Nat}
    {state : State α (.dim m (.dim n .scalar))}
    {params grads direction : MatrixTensor α m n}
    (hstep : ExactCertifiedStep state params grads direction) :
    (update state params grads).2 = subSpec params (scaleSpec direction state.lr) :=
  hstep.params_eq

/-- An approximate certified step gives the parameter-update equation for the certified direction. -/
theorem approxCertifiedStep_params_eq {m n : Nat} {eps : α}
    {state : State α (.dim m (.dim n .scalar))}
    {params grads direction : MatrixTensor α m n}
    (hstep : ApproxCertifiedStep eps state params grads direction) :
    (update state params grads).2 = subSpec params (scaleSpec direction state.lr) :=
  hstep.params_eq

/--
If the Muon backend is an exact matrix orthogonalizer, then one executable update uses a certified
direction.  The existential direction is not merely postulated: it is the backend applied to the
fresh momentum buffer stored by the update.
-/
theorem update_has_exact_certified_direction {m n : Nat}
    (state : State α (.dim m (.dim n .scalar)))
    (params grads : MatrixTensor α m n)
    (horth : ExactMatrixOrthogonalizer state.orthogonalizer) :
    ∃ direction : MatrixTensor α m n,
      ExactCertifiedDirection state.orthogonalizer (update state params grads).1.buf direction ∧
      (update state params grads).2 = subSpec params (scaleSpec direction state.lr) := by
  refine ⟨state.orthogonalizer.apply (update state params grads).1.buf, ?_, ?_⟩
  · exact exact_certified_direction_of_orthogonalizer horth (update state params grads).1.buf
  · rfl

/--
If the Muon backend is an exact matrix orthogonalizer, then one executable update has a certified
step: certified direction, exact next-state equation, and exact parameter equation.
-/
theorem update_has_exact_certified_step {m n : Nat}
    (state : State α (.dim m (.dim n .scalar)))
    (params grads : MatrixTensor α m n)
    (horth : ExactMatrixOrthogonalizer state.orthogonalizer) :
    ∃ direction : MatrixTensor α m n,
      ExactCertifiedStep state params grads direction := by
  refine ⟨state.orthogonalizer.apply (update state params grads).1.buf, ?_⟩
  exact exact_certified_step_of_direction state params grads _
    (exact_certified_direction_of_orthogonalizer horth (update state params grads).1.buf)

/-- Same as `update_has_exact_certified_direction`, but consumes a certified backend record. -/
theorem update_has_exact_certified_direction_of_backend {m n : Nat}
    (backend : ExactCertifiedOrthogonalizer α m n)
    (lr momentum : α) (buf params grads : MatrixTensor α m n) :
    ∃ direction : MatrixTensor α m n,
      ExactCertifiedDirection backend.orthogonalizer
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          params grads).1.buf direction ∧
      (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          params grads).2 =
        subSpec params (scaleSpec direction lr) := by
  let state : State α (.dim m (.dim n .scalar)) :=
    { lr := lr, momentum := momentum, buf := buf, orthogonalizer := backend.orthogonalizer }
  exact update_has_exact_certified_direction
    (state := state) (params := params) (grads := grads) backend.certified

/-- Same as `update_has_exact_certified_step`, but consumes a certified backend record. -/
theorem update_has_exact_certified_step_of_backend {m n : Nat}
    (backend : ExactCertifiedOrthogonalizer α m n)
    (lr momentum : α) (buf params grads : MatrixTensor α m n) :
    ∃ direction : MatrixTensor α m n,
      ExactCertifiedStep
        ({ lr := lr, momentum := momentum, buf := buf,
           orthogonalizer := backend.orthogonalizer } :
          State α (.dim m (.dim n .scalar)))
        params grads direction := by
  let state : State α (.dim m (.dim n .scalar)) :=
    { lr := lr, momentum := momentum, buf := buf, orthogonalizer := backend.orthogonalizer }
  exact update_has_exact_certified_step
    (state := state) (params := params) (grads := grads) backend.certified

/--
Local-buffer version of the exact Muon certificate. This is useful for concrete orthogonalizers such
as QR, where exactness may require a success condition on the particular fresh momentum buffer.
-/
theorem update_has_exact_certified_direction_of_buffer {m n : Nat}
    (state : State α (.dim m (.dim n .scalar)))
    (params grads : MatrixTensor α m n)
    (horth :
      ExactOrthogonalizesBuffer state.orthogonalizer (update state params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ExactCertifiedDirection state.orthogonalizer (update state params grads).1.buf direction ∧
      (update state params grads).2 = subSpec params (scaleSpec direction state.lr) := by
  refine ⟨state.orthogonalizer.apply (update state params grads).1.buf, ?_, ?_⟩
  · exact ⟨rfl, horth⟩
  · rfl

/-- Local-buffer version of the exact certified Muon step theorem. -/
theorem update_has_exact_certified_step_of_buffer {m n : Nat}
    (state : State α (.dim m (.dim n .scalar)))
    (params grads : MatrixTensor α m n)
    (horth :
      ExactOrthogonalizesBuffer state.orthogonalizer (update state params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ExactCertifiedStep state params grads direction := by
  refine ⟨state.orthogonalizer.apply (update state params grads).1.buf, ?_⟩
  exact exact_certified_step_of_direction state params grads _ ⟨rfl, horth⟩

/--
Checked-backend version of the exact Muon certificate. The success predicate is checked on the fresh
momentum buffer actually used by the step.
-/
theorem update_has_exact_certified_direction_of_checked_backend {m n : Nat}
    (backend : CheckedExactOrthogonalizer α m n)
    (lr momentum : α) (buf params grads : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ExactCertifiedDirection backend.orthogonalizer
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          params grads).1.buf direction ∧
      (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          params grads).2 =
        subSpec params (scaleSpec direction lr) := by
  let state : State α (.dim m (.dim n .scalar)) :=
    { lr := lr, momentum := momentum, buf := buf, orthogonalizer := backend.orthogonalizer }
  exact update_has_exact_certified_direction_of_buffer
    (state := state) (params := params) (grads := grads)
    (backend.certified (update state params grads).1.buf hsuccess)

/--
Checked-backend version of the exact certified Muon step. The success predicate is checked on the
fresh momentum buffer actually used by the step.
-/
theorem update_has_exact_certified_step_of_checked_backend {m n : Nat}
    (backend : CheckedExactOrthogonalizer α m n)
    (lr momentum : α) (buf params grads : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ExactCertifiedStep
        ({ lr := lr, momentum := momentum, buf := buf,
           orthogonalizer := backend.orthogonalizer } :
          State α (.dim m (.dim n .scalar)))
        params grads direction := by
  let state : State α (.dim m (.dim n .scalar)) :=
    { lr := lr, momentum := momentum, buf := buf, orthogonalizer := backend.orthogonalizer }
  exact update_has_exact_certified_step_of_buffer
    (state := state) (params := params) (grads := grads)
    (backend.certified (update state params grads).1.buf hsuccess)

/--
Checked exact backends give the concrete `QᵀQ = I` fact for the direction used by this update.
-/
theorem update_direction_has_exact_column_gram_of_checked_backend {m n : Nat}
    (backend : CheckedExactOrthogonalizer α m n)
    (lr momentum : α) (buf params grads : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    HasExactColumnGram
      (backend.orthogonalizer.apply
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          params grads).1.buf) := by
  let state : State α (.dim m (.dim n .scalar)) :=
    { lr := lr, momentum := momentum, buf := buf, orthogonalizer := backend.orthogonalizer }
  exact backend.certified (update state params grads).1.buf hsuccess

/--
Initialized checked exact backend theorem: after Muon is initialized with a checked exact backend,
success on the fresh momentum buffer certifies the direction used by the first update.
-/
theorem init_has_exact_certified_direction_of_checked_backend {m n : Nat}
    (backend : CheckedExactOrthogonalizer α m n)
    (lr momentum : α) (params grads : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (update (init lr momentum backend.orthogonalizer params) params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ExactCertifiedDirection backend.orthogonalizer
        (update (init lr momentum backend.orthogonalizer params) params grads).1.buf direction ∧
      (update (init lr momentum backend.orthogonalizer params) params grads).2 =
        subSpec params (scaleSpec direction lr) := by
  exact update_has_exact_certified_direction_of_checked_backend
    (backend := backend) (lr := lr) (momentum := momentum) (buf := fill 0 (.dim m (.dim n .scalar)))
    (params := params) (grads := grads) hsuccess

/--
Initialized checked exact backend theorem: after Muon is initialized with a checked exact backend,
success on the fresh momentum buffer gives a certified first step.
-/
theorem init_has_exact_certified_step_of_checked_backend {m n : Nat}
    (backend : CheckedExactOrthogonalizer α m n)
    (lr momentum : α) (params grads : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (update (init lr momentum backend.orthogonalizer params) params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ExactCertifiedStep
        (init lr momentum backend.orthogonalizer params)
        params grads direction := by
  exact update_has_exact_certified_step_of_checked_backend
    (backend := backend) (lr := lr) (momentum := momentum)
    (buf := fill 0 (.dim m (.dim n .scalar)))
    (params := params) (grads := grads) hsuccess

/--
Initialized checked exact backend direction theorem: after initialization, backend success gives
`QᵀQ = I` for the actual direction used by the first Muon update.
-/
theorem init_direction_has_exact_column_gram_of_checked_backend {m n : Nat}
    (backend : CheckedExactOrthogonalizer α m n)
    (lr momentum : α) (params grads : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (update (init lr momentum backend.orthogonalizer params) params grads).1.buf) :
    HasExactColumnGram
      (backend.orthogonalizer.apply
        (update (init lr momentum backend.orthogonalizer params) params grads).1.buf) := by
  exact update_direction_has_exact_column_gram_of_checked_backend
    (backend := backend) (lr := lr) (momentum := momentum)
    (buf := fill 0 (.dim m (.dim n .scalar))) (params := params) (grads := grads) hsuccess

/--
A fixed-point checked Newton-Schulz backend gives an exact certified direction for one Muon update.
-/
theorem update_has_exact_certified_direction_newtonSchulz_fixed_checked {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat)
    (lr momentum : α) (buf params grads : MatrixTensor α m n)
    (hsuccess :
      (newtonSchulzFixedPointCheckedExactOrthogonalizer
        (α := α) (m := m) (n := n) coeffs steps).Success
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ExactCertifiedDirection
        (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            State α (.dim m (.dim n .scalar)))
          params grads).1.buf direction ∧
      (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            State α (.dim m (.dim n .scalar)))
          params grads).2 =
        subSpec params (scaleSpec direction lr) := by
  exact update_has_exact_certified_direction_of_checked_backend
    (backend := newtonSchulzFixedPointCheckedExactOrthogonalizer
      (α := α) (m := m) (n := n) coeffs steps)
    lr momentum buf params grads hsuccess

/--
A fixed-point checked Newton-Schulz backend gives an exact certified step for one Muon update.
-/
theorem update_has_exact_certified_step_newtonSchulz_fixed_checked {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat)
    (lr momentum : α) (buf params grads : MatrixTensor α m n)
    (hsuccess :
      (newtonSchulzFixedPointCheckedExactOrthogonalizer
        (α := α) (m := m) (n := n) coeffs steps).Success
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ExactCertifiedStep
        ({ lr := lr, momentum := momentum, buf := buf,
           orthogonalizer :=
            newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
          State α (.dim m (.dim n .scalar)))
        params grads direction := by
  exact update_has_exact_certified_step_of_checked_backend
    (backend := newtonSchulzFixedPointCheckedExactOrthogonalizer
      (α := α) (m := m) (n := n) coeffs steps)
    lr momentum buf params grads hsuccess

/-- Fixed-point checked Newton-Schulz gives `QᵀQ = I` for the actual update direction. -/
theorem update_newtonSchulz_fixed_direction_has_exact_column_gram_checked {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat)
    (lr momentum : α) (buf params grads : MatrixTensor α m n)
    (hsuccess :
      (newtonSchulzFixedPointCheckedExactOrthogonalizer
        (α := α) (m := m) (n := n) coeffs steps).Success
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    HasExactColumnGram
      ((newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps).apply
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            State α (.dim m (.dim n .scalar)))
          params grads).1.buf) := by
  exact update_direction_has_exact_column_gram_of_checked_backend
    (backend := newtonSchulzFixedPointCheckedExactOrthogonalizer
      (α := α) (m := m) (n := n) coeffs steps)
    lr momentum buf params grads hsuccess

/--
Initialized fixed-point checked Newton-Schulz gives an exact certified first Muon step.
-/
theorem init_has_exact_certified_step_newtonSchulz_fixed_checked {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat)
    (lr momentum : α) (params grads : MatrixTensor α m n)
    (hsuccess :
      (newtonSchulzFixedPointCheckedExactOrthogonalizer
        (α := α) (m := m) (n := n) coeffs steps).Success
        (update
          (init lr momentum
            (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
            params)
          params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ExactCertifiedStep
        (init lr momentum
          (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
          params)
        params grads direction := by
  exact init_has_exact_certified_step_of_checked_backend
    (backend := newtonSchulzFixedPointCheckedExactOrthogonalizer
      (α := α) (m := m) (n := n) coeffs steps)
    lr momentum params grads hsuccess

/-- Initialized fixed-point checked Newton-Schulz gives `QᵀQ = I` for the first update direction. -/
theorem init_newtonSchulz_fixed_direction_has_exact_column_gram_checked {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat)
    (lr momentum : α) (params grads : MatrixTensor α m n)
    (hsuccess :
      (newtonSchulzFixedPointCheckedExactOrthogonalizer
        (α := α) (m := m) (n := n) coeffs steps).Success
        (update
          (init lr momentum
            (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
            params)
          params grads).1.buf) :
    HasExactColumnGram
      ((newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps).apply
        (update
          (init lr momentum
            (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
            params)
          params grads).1.buf) := by
  exact init_direction_has_exact_column_gram_of_checked_backend
    (backend := newtonSchulzFixedPointCheckedExactOrthogonalizer
      (α := α) (m := m) (n := n) coeffs steps)
    lr momentum params grads hsuccess

/--
If the Muon backend is an approximate matrix orthogonalizer, then one executable update uses a
direction whose Gram residual has the same entrywise bound.
-/
theorem update_has_approx_certified_direction {m n : Nat} {eps : α}
    (state : State α (.dim m (.dim n .scalar)))
    (params grads : MatrixTensor α m n)
    (horth : ApproxMatrixOrthogonalizer eps state.orthogonalizer) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedDirection eps state.orthogonalizer (update state params grads).1.buf direction ∧
      (update state params grads).2 = subSpec params (scaleSpec direction state.lr) := by
  refine ⟨state.orthogonalizer.apply (update state params grads).1.buf, ?_, ?_⟩
  · exact approx_certified_direction_of_orthogonalizer horth (update state params grads).1.buf
  · rfl

/--
If the Muon backend is an approximate matrix orthogonalizer, then one executable update has a
certified approximate step.
-/
theorem update_has_approx_certified_step {m n : Nat} {eps : α}
    (state : State α (.dim m (.dim n .scalar)))
    (params grads : MatrixTensor α m n)
    (horth : ApproxMatrixOrthogonalizer eps state.orthogonalizer) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedStep eps state params grads direction := by
  refine ⟨state.orthogonalizer.apply (update state params grads).1.buf, ?_⟩
  exact approx_certified_step_of_direction state params grads _
    (approx_certified_direction_of_orthogonalizer horth (update state params grads).1.buf)

/-- Same as `update_has_approx_certified_direction`, but consumes a certified backend record. -/
theorem update_has_approx_certified_direction_of_backend {m n : Nat} {eps : α}
    (backend : ApproxCertifiedOrthogonalizer α m n eps)
    (lr momentum : α) (buf params grads : MatrixTensor α m n) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedDirection eps backend.orthogonalizer
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          params grads).1.buf direction ∧
      (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          params grads).2 =
        subSpec params (scaleSpec direction lr) := by
  let state : State α (.dim m (.dim n .scalar)) :=
    { lr := lr, momentum := momentum, buf := buf, orthogonalizer := backend.orthogonalizer }
  exact update_has_approx_certified_direction
    (state := state) (params := params) (grads := grads) backend.certified

/-- Same as `update_has_approx_certified_step`, but consumes a certified backend record. -/
theorem update_has_approx_certified_step_of_backend {m n : Nat} {eps : α}
    (backend : ApproxCertifiedOrthogonalizer α m n eps)
    (lr momentum : α) (buf params grads : MatrixTensor α m n) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedStep eps
        ({ lr := lr, momentum := momentum, buf := buf,
           orthogonalizer := backend.orthogonalizer } :
          State α (.dim m (.dim n .scalar)))
        params grads direction := by
  let state : State α (.dim m (.dim n .scalar)) :=
    { lr := lr, momentum := momentum, buf := buf, orthogonalizer := backend.orthogonalizer }
  exact update_has_approx_certified_step
    (state := state) (params := params) (grads := grads) backend.certified

/--
Local-buffer version of the approximate Muon certificate. This lets a backend certify only the
fresh momentum buffer used by the current step.
-/
theorem update_has_approx_certified_direction_of_buffer {m n : Nat} {eps : α}
    (state : State α (.dim m (.dim n .scalar)))
    (params grads : MatrixTensor α m n)
    (horth :
      ApproxOrthogonalizesBuffer eps state.orthogonalizer (update state params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedDirection eps state.orthogonalizer (update state params grads).1.buf direction ∧
      (update state params grads).2 = subSpec params (scaleSpec direction state.lr) := by
  refine ⟨state.orthogonalizer.apply (update state params grads).1.buf, ?_, ?_⟩
  · exact ⟨rfl, horth⟩
  · rfl

/-- Local-buffer version of the approximate certified Muon step theorem. -/
theorem update_has_approx_certified_step_of_buffer {m n : Nat} {eps : α}
    (state : State α (.dim m (.dim n .scalar)))
    (params grads : MatrixTensor α m n)
    (horth :
      ApproxOrthogonalizesBuffer eps state.orthogonalizer (update state params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedStep eps state params grads direction := by
  refine ⟨state.orthogonalizer.apply (update state params grads).1.buf, ?_⟩
  exact approx_certified_step_of_direction state params grads _ ⟨rfl, horth⟩

/--
Checked-backend version of the approximate Muon certificate. This is the main theorem a
Newton-Schulz/CUDA backend should target: once the checker establishes the backend success predicate
on the fresh momentum buffer, the Muon update is known to use an approximately column-orthogonal
direction.
-/
theorem update_has_approx_certified_direction_of_checked_backend {m n : Nat} {eps : α}
    (backend : CheckedApproxOrthogonalizer α m n eps)
    (lr momentum : α) (buf params grads : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedDirection eps backend.orthogonalizer
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          params grads).1.buf direction ∧
      (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          params grads).2 =
        subSpec params (scaleSpec direction lr) := by
  let state : State α (.dim m (.dim n .scalar)) :=
    { lr := lr, momentum := momentum, buf := buf, orthogonalizer := backend.orthogonalizer }
  exact update_has_approx_certified_direction_of_buffer
    (state := state) (params := params) (grads := grads)
    (backend.certified (update state params grads).1.buf hsuccess)

/--
Checked-backend version of the approximate certified Muon step. This is the step-level theorem a
residual-checked Newton-Schulz/CUDA backend should target.
-/
theorem update_has_approx_certified_step_of_checked_backend {m n : Nat} {eps : α}
    (backend : CheckedApproxOrthogonalizer α m n eps)
    (lr momentum : α) (buf params grads : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedStep eps
        ({ lr := lr, momentum := momentum, buf := buf,
           orthogonalizer := backend.orthogonalizer } :
          State α (.dim m (.dim n .scalar)))
        params grads direction := by
  let state : State α (.dim m (.dim n .scalar)) :=
    { lr := lr, momentum := momentum, buf := buf, orthogonalizer := backend.orthogonalizer }
  exact update_has_approx_certified_step_of_buffer
    (state := state) (params := params) (grads := grads)
    (backend.certified (update state params grads).1.buf hsuccess)

/--
Checked approximate backends give the concrete residual bound for the direction used by this
update.
-/
theorem update_direction_has_approx_column_gram_of_checked_backend {m n : Nat} {eps : α}
    (backend : CheckedApproxOrthogonalizer α m n eps)
    (lr momentum : α) (buf params grads : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    HasApproxColumnGram eps
      (backend.orthogonalizer.apply
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            State α (.dim m (.dim n .scalar)))
          params grads).1.buf) := by
  let state : State α (.dim m (.dim n .scalar)) :=
    { lr := lr, momentum := momentum, buf := buf, orthogonalizer := backend.orthogonalizer }
  exact backend.certified (update state params grads).1.buf hsuccess

/--
Initialized checked approximate backend theorem: after Muon is initialized with a checked
approximate backend, success on the fresh momentum buffer certifies the direction used by the first
update.
-/
theorem init_has_approx_certified_direction_of_checked_backend {m n : Nat} {eps : α}
    (backend : CheckedApproxOrthogonalizer α m n eps)
    (lr momentum : α) (params grads : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (update (init lr momentum backend.orthogonalizer params) params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedDirection eps backend.orthogonalizer
        (update (init lr momentum backend.orthogonalizer params) params grads).1.buf direction ∧
      (update (init lr momentum backend.orthogonalizer params) params grads).2 =
        subSpec params (scaleSpec direction lr) := by
  exact update_has_approx_certified_direction_of_checked_backend
    (backend := backend) (lr := lr) (momentum := momentum) (buf := fill 0 (.dim m (.dim n .scalar)))
    (params := params) (grads := grads) hsuccess

/--
Initialized checked approximate backend theorem: after Muon is initialized with a checked
approximate backend, success on the fresh momentum buffer gives a certified first step.
-/
theorem init_has_approx_certified_step_of_checked_backend {m n : Nat} {eps : α}
    (backend : CheckedApproxOrthogonalizer α m n eps)
    (lr momentum : α) (params grads : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (update (init lr momentum backend.orthogonalizer params) params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedStep eps
        (init lr momentum backend.orthogonalizer params)
        params grads direction := by
  exact update_has_approx_certified_step_of_checked_backend
    (backend := backend) (lr := lr) (momentum := momentum)
    (buf := fill 0 (.dim m (.dim n .scalar)))
    (params := params) (grads := grads) hsuccess

/--
Initialized checked approximate backend direction theorem: after initialization, backend success
gives the residual bound for the actual direction used by the first Muon update.
-/
theorem init_direction_has_approx_column_gram_of_checked_backend {m n : Nat} {eps : α}
    (backend : CheckedApproxOrthogonalizer α m n eps)
    (lr momentum : α) (params grads : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (update (init lr momentum backend.orthogonalizer params) params grads).1.buf) :
    HasApproxColumnGram eps
      (backend.orthogonalizer.apply
        (update (init lr momentum backend.orthogonalizer params) params grads).1.buf) := by
  exact update_direction_has_approx_column_gram_of_checked_backend
    (backend := backend) (lr := lr) (momentum := momentum)
    (buf := fill 0 (.dim m (.dim n .scalar))) (params := params) (grads := grads) hsuccess

/--
Checked Newton-Schulz Muon theorem: if the Newton-Schulz output for the fresh momentum buffer passes
the residual check, then the executable Muon update uses an approximately column-orthogonal
direction.
-/
theorem update_has_approx_certified_direction_newtonSchulz_checked {m n : Nat}
    {eps : α} (coeffs : NewtonSchulzCoeffs α) (steps : Nat)
    (lr momentum : α) (buf params grads : MatrixTensor α m n)
    (hresidual :
      ResidualApproxSuccess eps
        (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedDirection eps
        (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            State α (.dim m (.dim n .scalar)))
          params grads).1.buf direction ∧
      (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            State α (.dim m (.dim n .scalar)))
          params grads).2 =
        subSpec params (scaleSpec direction lr) := by
  exact update_has_approx_certified_direction_of_checked_backend
    (backend := newtonSchulzResidualCheckedOrthogonalizer
      (α := α) (m := m) (n := n) coeffs steps eps)
    (lr := lr) (momentum := momentum) (buf := buf) (params := params) (grads := grads)
    hresidual

/--
Checked Newton-Schulz Muon step theorem: if the Newton-Schulz output for the fresh momentum buffer
passes the residual check, then the executable Muon update has a certified approximate step.
-/
theorem update_has_approx_certified_step_newtonSchulz_checked {m n : Nat}
    {eps : α} (coeffs : NewtonSchulzCoeffs α) (steps : Nat)
    (lr momentum : α) (buf params grads : MatrixTensor α m n)
    (hresidual :
      ResidualApproxSuccess eps
        (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedStep eps
        ({ lr := lr, momentum := momentum, buf := buf,
           orthogonalizer :=
            newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
          State α (.dim m (.dim n .scalar)))
        params grads direction := by
  exact update_has_approx_certified_step_of_checked_backend
    (backend := newtonSchulzResidualCheckedOrthogonalizer
      (α := α) (m := m) (n := n) coeffs steps eps)
    (lr := lr) (momentum := momentum) (buf := buf) (params := params) (grads := grads)
    hresidual

/--
Checked Newton-Schulz direction theorem: if the fresh momentum buffer passes the residual check,
then the actual direction used by the Muon update has the requested Gram-residual bound.
-/
theorem update_newtonSchulz_direction_has_approx_column_gram_checked {m n : Nat}
    {eps : α} (coeffs : NewtonSchulzCoeffs α) (steps : Nat)
    (lr momentum : α) (buf params grads : MatrixTensor α m n)
    (hresidual :
      ResidualApproxSuccess eps
        (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    HasApproxColumnGram eps
      ((newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps).apply
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            State α (.dim m (.dim n .scalar)))
          params grads).1.buf) := by
  exact hresidual

/--
Initialized checked Newton-Schulz theorem: if the first fresh momentum buffer passes the residual
check, then the first initialized Muon update uses an approximately certified direction.
-/
theorem init_has_approx_certified_direction_newtonSchulz_checked {m n : Nat}
    {eps : α} (coeffs : NewtonSchulzCoeffs α) (steps : Nat)
    (lr momentum : α) (params grads : MatrixTensor α m n)
    (hresidual :
      ResidualApproxSuccess eps
        (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
        (update
          (init lr momentum
            (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
            params)
          params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedDirection eps
        (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
        (update
          (init lr momentum
            (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
            params)
          params grads).1.buf direction ∧
      (update
          (init lr momentum
            (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
            params)
          params grads).2 =
        subSpec params (scaleSpec direction lr) := by
  exact init_has_approx_certified_direction_of_checked_backend
    (backend := newtonSchulzResidualCheckedOrthogonalizer
      (α := α) (m := m) (n := n) coeffs steps eps)
    (lr := lr) (momentum := momentum) (params := params) (grads := grads)
    hresidual

/--
Initialized checked Newton-Schulz step theorem: if the first fresh momentum buffer passes the
residual check, then the first initialized Muon update has a certified approximate step.
-/
theorem init_has_approx_certified_step_newtonSchulz_checked {m n : Nat}
    {eps : α} (coeffs : NewtonSchulzCoeffs α) (steps : Nat)
    (lr momentum : α) (params grads : MatrixTensor α m n)
    (hresidual :
      ResidualApproxSuccess eps
        (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
        (update
          (init lr momentum
            (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
            params)
          params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedStep eps
        (init lr momentum
          (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
          params)
        params grads direction := by
  exact init_has_approx_certified_step_of_checked_backend
    (backend := newtonSchulzResidualCheckedOrthogonalizer
      (α := α) (m := m) (n := n) coeffs steps eps)
    (lr := lr) (momentum := momentum) (params := params) (grads := grads)
    hresidual

/--
Initialized checked Newton-Schulz direction theorem: if the first fresh momentum buffer passes the
residual check, then the first initialized update direction satisfies the Gram-residual bound.
-/
theorem init_newtonSchulz_direction_has_approx_column_gram_checked {m n : Nat}
    {eps : α} (coeffs : NewtonSchulzCoeffs α) (steps : Nat)
    (lr momentum : α) (params grads : MatrixTensor α m n)
    (hresidual :
      ResidualApproxSuccess eps
        (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
        (update
          (init lr momentum
            (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
            params)
          params grads).1.buf) :
    HasApproxColumnGram eps
      ((newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps).apply
        (update
          (init lr momentum
            (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
            params)
          params grads).1.buf) := by
  exact hresidual

/-- The exact certificate gives the concrete `QᵀQ = I` fact for the direction used by the update. -/
theorem update_direction_has_exact_column_gram {m n : Nat}
    (state : State α (.dim m (.dim n .scalar)))
    (params grads : MatrixTensor α m n)
    (horth : ExactMatrixOrthogonalizer state.orthogonalizer) :
    HasExactColumnGram (state.orthogonalizer.apply (update state params grads).1.buf) := by
  exact horth (update state params grads).1.buf

/--
The approximate certificate gives the concrete residual bound for the direction used by the
update.
-/
theorem update_direction_has_approx_column_gram {m n : Nat} {eps : α}
    (state : State α (.dim m (.dim n .scalar)))
    (params grads : MatrixTensor α m n)
    (horth : ApproxMatrixOrthogonalizer eps state.orthogonalizer) :
    HasApproxColumnGram eps (state.orthogonalizer.apply (update state params grads).1.buf) := by
  exact horth (update state params grads).1.buf

/-! ## QR as a concrete exact orthogonalizer over `ℝ` -/

/-- QR/Gram-Schmidt orthogonalizer: return the `Q` factor of the fresh matrix buffer. -/
noncomputable def qrOrthogonalizer {m n : Nat} :
    Orthogonalizer ℝ (.dim m (.dim n .scalar)) :=
  { apply := fun buffer => qrQSpec buffer }

/-- The success condition for TorchLean's executable QR orthogonalizer. -/
def HasPositiveQRPivots {m n : Nat} (buffer : MatrixTensor ℝ m n) : Prop :=
  ∀ j : Fin n, 0 < get2 (qrRSpec buffer) j j

lemma get2_identityTensorSpec_real {n : Nat} (i j : Fin n) :
    get2 (identityTensorSpec (α := ℝ) n) i j = if i = j then 1 else 0 := by
  cases n with
  | zero => exact Fin.elim0 i
  | succ n =>
      by_cases h : i = j
      · subst j
        simp [identityTensorSpec, get2_eq, get_eq]
      · have hval : i.val ≠ j.val := by
          intro hv
          exact h (Fin.ext hv)
        simp [identityTensorSpec, get2_eq, get_eq, h, hval]

/-- Entry rule for matrix-shaped tensor addition over `ℝ`. -/
lemma get2_addSpec_real {m n : Nat} (A B : MatrixTensor ℝ m n) (i : Fin m) (j : Fin n) :
    get2 (addSpec A B) i j = get2 A i j + get2 B i j := by
  cases A with
  | dim rowsA =>
    cases B with
    | dim rowsB =>
      cases hA : rowsA i with
      | dim colsA =>
        cases hB : rowsB i with
        | dim colsB =>
          cases hAj : colsA j with
          | scalar a =>
            cases hBj : colsB j with
            | scalar b =>
              simp [addSpec, map2Spec, get2_eq, get_eq, hA, hB, hAj, hBj]

/-- Entry rule for matrix-shaped tensor scaling over `ℝ`. -/
lemma get2_scaleSpec_real {m n : Nat} (A : MatrixTensor ℝ m n) (c : ℝ)
    (i : Fin m) (j : Fin n) :
    get2 (scaleSpec A c) i j = get2 A i j * c := by
  cases A with
  | dim rows =>
    cases hrow : rows i with
    | dim cols =>
      cases hcol : cols j with
      | scalar a =>
        simp [scaleSpec, mapSpec, get2_eq, get_eq, hrow, hcol]

/-- Entry rule for matrix-shaped tensor subtraction over `ℝ`. -/
lemma get2_subSpec_real {m n : Nat} (A B : MatrixTensor ℝ m n) (i : Fin m) (j : Fin n) :
    get2 (subSpec A B) i j = get2 A i j - get2 B i j := by
  cases A with
  | dim rowsA =>
    cases B with
    | dim rowsB =>
      cases hA : rowsA i with
      | dim colsA =>
        cases hB : rowsB i with
        | dim colsB =>
          cases hAj : colsA j with
          | scalar a =>
            cases hBj : colsB j with
            | scalar b =>
              simp [subSpec, map2Spec, get2_eq, get_eq, hA, hB, hAj, hBj]

/-- Right multiplication by the identity matrix leaves a real matrix unchanged. -/
theorem matMul_right_identity_real {m n : Nat} (A : MatrixTensor ℝ m n) :
    matMulSpec A (identityTensorSpec (α := ℝ) n) = A := by
  classical
  apply matrix_ext
  intro i j
  calc
    get2 (matMulSpec A (identityTensorSpec (α := ℝ) n)) i j
        = ∑ k : Fin n, get2 A i k * get2 (identityTensorSpec (α := ℝ) n) k j := by
          simpa using
            (get2_mat_mul_spec (A := A) (B := identityTensorSpec (α := ℝ) n) (i := i) (j := j))
    _ = ∑ k : Fin n, get2 A i k * (if k = j then 1 else 0) := by
          refine Finset.sum_congr rfl ?_
          intro k _
          rw [get2_identityTensorSpec_real]
    _ = get2 A i j := by
          simp

/--
Three scaled copies of the same real matrix combine into one scaled copy using the sum of the
coefficients.
-/
theorem add_scaled_three_eq_scale_sum {m n : Nat}
    (Q : MatrixTensor ℝ m n) (a b c : ℝ) :
    addSpec (addSpec (scaleSpec Q a) (scaleSpec Q b)) (scaleSpec Q c) =
      scaleSpec Q (a + b + c) := by
  apply matrix_ext
  intro i j
  calc
    get2 (addSpec (addSpec (scaleSpec Q a) (scaleSpec Q b)) (scaleSpec Q c)) i j
        = (get2 Q i j * a + get2 Q i j * b) + get2 Q i j * c := by
          rw [get2_addSpec_real, get2_addSpec_real, get2_scaleSpec_real,
            get2_scaleSpec_real, get2_scaleSpec_real]
    _ = get2 Q i j * (a + b + c) := by
          ring
    _ = get2 (scaleSpec Q (a + b + c)) i j := by
          rw [get2_scaleSpec_real]

/--
If three scaled copies of a matrix are added and the coefficients sum to one, the result is the
original matrix.
-/
theorem add_scaled_three_eq_self_of_coeff_sum_one {m n : Nat}
    (Q : MatrixTensor ℝ m n) (a b c : ℝ) (hsum : a + b + c = 1) :
    addSpec (addSpec (scaleSpec Q a) (scaleSpec Q b)) (scaleSpec Q c) = Q := by
  rw [add_scaled_three_eq_scale_sum]
  apply matrix_ext
  intro i j
  calc
    get2 (scaleSpec Q (a + b + c)) i j = get2 Q i j * (a + b + c) := by
      rw [get2_scaleSpec_real]
    _ = get2 Q i j * 1 := by
          rw [hsum]
    _ = get2 Q i j := by
      ring

/--
Scaling an exact-column-orthogonal real matrix by a scalar whose square is one preserves exact
column Gram.
-/
theorem scale_hasExactColumnGram_of_square_eq_one {m n : Nat}
    (Q : MatrixTensor ℝ m n) (k : ℝ)
    (hgram : HasExactColumnGram Q) (hk : k * k = 1) :
    HasExactColumnGram (scaleSpec Q k) := by
  unfold HasExactColumnGram columnGram
  apply matrix_ext
  intro i j
  have hgram' : matMulSpec (Spec.Tensor.matrixTransposeSpec Q) Q =
      identityTensorSpec (α := ℝ) n := by
    simpa [HasExactColumnGram, columnGram] using hgram
  have hentry :
      (∑ r : Fin m, get2 Q r i * get2 Q r j) =
        get2 (identityTensorSpec (α := ℝ) n) i j := by
    calc
      (∑ r : Fin m, get2 Q r i * get2 Q r j)
          = get2 (matMulSpec (Spec.Tensor.matrixTransposeSpec Q) Q) i j := by
            symm
            calc
              get2 (matMulSpec (Spec.Tensor.matrixTransposeSpec Q) Q) i j
                  = ∑ r : Fin m,
                      get2 (Spec.Tensor.matrixTransposeSpec Q) i r * get2 Q r j := by
                    simpa using
                      (get2_mat_mul_spec
                        (A := Spec.Tensor.matrixTransposeSpec Q) (B := Q) (i := i) (j := j))
              _ = ∑ r : Fin m, get2 Q r i * get2 Q r j := by
                    refine Finset.sum_congr rfl ?_
                    intro r _
                    rw [get2_matrix_transpose_spec]
      _ = get2 (identityTensorSpec (α := ℝ) n) i j := by
            exact congrArg (fun M => get2 M i j) hgram'
  calc
    get2
        (matMulSpec (Spec.Tensor.matrixTransposeSpec (scaleSpec Q k)) (scaleSpec Q k)) i j
        = ∑ r : Fin m,
            get2 (Spec.Tensor.matrixTransposeSpec (scaleSpec Q k)) i r *
              get2 (scaleSpec Q k) r j := by
          simpa using
            (get2_mat_mul_spec
              (A := Spec.Tensor.matrixTransposeSpec (scaleSpec Q k)) (B := scaleSpec Q k)
              (i := i) (j := j))
    _ = ∑ r : Fin m, (get2 Q r i * k) * (get2 Q r j * k) := by
          refine Finset.sum_congr rfl ?_
          intro r _
          rw [get2_matrix_transpose_spec, get2_scaleSpec_real, get2_scaleSpec_real]
    _ = ∑ r : Fin m, (get2 Q r i * get2 Q r j) * (k * k) := by
          refine Finset.sum_congr rfl ?_
          intro r _
          ring
    _ = (∑ r : Fin m, get2 Q r i * get2 Q r j) * (k * k) := by
          rw [Finset.sum_mul]
    _ = (∑ r : Fin m, get2 Q r i * get2 Q r j) * 1 := by
          rw [hk]
    _ = get2 (identityTensorSpec (α := ℝ) n) i j := by
          rw [mul_one, hentry]

/--
Scaling an exact-column-orthogonal real matrix gives an approximate Gram certificate whenever
`|k^2 - 1|` is bounded by the requested tolerance.
-/
theorem scale_hasApproxColumnGram_of_exact_column_gram_of_square_error {m n : Nat}
    (Q : MatrixTensor ℝ m n) (k eps : ℝ)
    (hgram : HasExactColumnGram Q)
    (herr : MathFunctions.abs (k * k - 1) ≤ eps)
    (heps : 0 ≤ eps) :
    HasApproxColumnGram eps (scaleSpec Q k) := by
  intro i j
  have hgram' : matMulSpec (Spec.Tensor.matrixTransposeSpec Q) Q =
      identityTensorSpec (α := ℝ) n := by
    simpa [HasExactColumnGram, columnGram] using hgram
  have hentry :
      (∑ r : Fin m, get2 Q r i * get2 Q r j) =
        get2 (identityTensorSpec (α := ℝ) n) i j := by
    calc
      (∑ r : Fin m, get2 Q r i * get2 Q r j)
          = get2 (matMulSpec (Spec.Tensor.matrixTransposeSpec Q) Q) i j := by
            symm
            calc
              get2 (matMulSpec (Spec.Tensor.matrixTransposeSpec Q) Q) i j
                  = ∑ r : Fin m,
                      get2 (Spec.Tensor.matrixTransposeSpec Q) i r * get2 Q r j := by
                    simpa using
                      (get2_mat_mul_spec
                        (A := Spec.Tensor.matrixTransposeSpec Q) (B := Q) (i := i) (j := j))
              _ = ∑ r : Fin m, get2 Q r i * get2 Q r j := by
                    refine Finset.sum_congr rfl ?_
                    intro r _
                    rw [get2_matrix_transpose_spec]
      _ = get2 (identityTensorSpec (α := ℝ) n) i j := by
            exact congrArg (fun M => get2 M i j) hgram'
  have hscaledGram :
      get2 (columnGram (scaleSpec Q k)) i j =
        get2 (identityTensorSpec (α := ℝ) n) i j * (k * k) := by
    calc
      get2 (columnGram (scaleSpec Q k)) i j
          = get2
              (matMulSpec (Spec.Tensor.matrixTransposeSpec (scaleSpec Q k)) (scaleSpec Q k))
              i j := by
            rfl
      _ = ∑ r : Fin m,
            get2 (Spec.Tensor.matrixTransposeSpec (scaleSpec Q k)) i r *
              get2 (scaleSpec Q k) r j := by
            simpa using
              (get2_mat_mul_spec
                (A := Spec.Tensor.matrixTransposeSpec (scaleSpec Q k)) (B := scaleSpec Q k)
                (i := i) (j := j))
      _ = ∑ r : Fin m, (get2 Q r i * k) * (get2 Q r j * k) := by
            refine Finset.sum_congr rfl ?_
            intro r _
            rw [get2_matrix_transpose_spec, get2_scaleSpec_real, get2_scaleSpec_real]
      _ = ∑ r : Fin m, (get2 Q r i * get2 Q r j) * (k * k) := by
            refine Finset.sum_congr rfl ?_
            intro r _
            ring
      _ = (∑ r : Fin m, get2 Q r i * get2 Q r j) * (k * k) := by
            rw [Finset.sum_mul]
      _ = get2 (identityTensorSpec (α := ℝ) n) i j * (k * k) := by
            rw [hentry]
  by_cases hij : i = j
  · subst j
    have hdiag : get2 (identityTensorSpec (α := ℝ) n) i i = 1 := by
      simp [get2_identityTensorSpec_real]
    calc
      MathFunctions.abs (get2 (columnGramResidual (scaleSpec Q k)) i i)
          = MathFunctions.abs (k * k - 1) := by
            rw [columnGramResidual_eq, get2_subSpec_real, hscaledGram, hdiag]
            ring_nf
      _ ≤ eps := herr
  · have hoff : get2 (identityTensorSpec (α := ℝ) n) i j = 0 := by
      rw [get2_identityTensorSpec_real]
      simp [hij]
    calc
      MathFunctions.abs (get2 (columnGramResidual (scaleSpec Q k)) i j)
          = 0 := by
            rw [columnGramResidual_eq, get2_subSpec_real, hscaledGram, hoff]
            simp [MathFunctions.abs]
      _ ≤ eps := heps

/--
If `QᵀQ = I`, then one column-oriented Newton-Schulz step returns
`(a + b + c) Q`.
-/
theorem newtonSchulzStep_eq_scale_sum_of_exact_column_gram {m n : Nat}
    (coeffs : NewtonSchulzCoeffs ℝ) (Q : MatrixTensor ℝ m n)
    (hgram : HasExactColumnGram Q) :
    newtonSchulzStep coeffs Q = scaleSpec Q (coeffs.a + coeffs.b + coeffs.c) := by
  unfold newtonSchulzStep
  have hright : rightGram Q = identityTensorSpec (α := ℝ) n := by
    simpa [HasExactColumnGram, columnGram, rightGram] using hgram
  have hXG : matMulSpec Q (rightGram Q) = Q := by
    rw [hright]
    exact matMul_right_identity_real Q
  have hXG2 : matMulSpec (matMulSpec Q (rightGram Q)) (rightGram Q) = Q := by
    rw [hright]
    rw [matMul_right_identity_real Q]
    exact matMul_right_identity_real Q
  simpa [hXG, hXG2] using
    add_scaled_three_eq_scale_sum Q coeffs.a coeffs.b coeffs.c

/--
For real coefficients whose sum is one, an exact-column-orthogonal matrix is a fixed point of one
column-oriented Newton-Schulz step.
-/
theorem newtonSchulzFixedPoint_of_exact_column_gram_of_coeff_sum_one {m n : Nat}
    (coeffs : NewtonSchulzCoeffs ℝ) (Q : MatrixTensor ℝ m n)
    (hgram : HasExactColumnGram Q)
    (hsum : coeffs.a + coeffs.b + coeffs.c = 1) :
    NewtonSchulzFixedPoint coeffs Q := by
  unfold NewtonSchulzFixedPoint
  rw [newtonSchulzStep_eq_scale_sum_of_exact_column_gram coeffs Q hgram]
  rw [hsum]
  apply matrix_ext
  intro i j
  calc
    get2 (scaleSpec Q 1) i j = get2 Q i j * 1 := by
      rw [get2_scaleSpec_real]
    _ = get2 Q i j := by
      ring

/--
If `QᵀQ = I` and `(a + b + c)^2 = 1`, then one column-oriented Newton-Schulz step still has exact
column Gram.
-/
theorem newtonSchulzStep_hasExactColumnGram_of_exact_column_gram_of_sum_square_one {m n : Nat}
    (coeffs : NewtonSchulzCoeffs ℝ) (Q : MatrixTensor ℝ m n)
    (hgram : HasExactColumnGram Q)
    (hsquare : (coeffs.a + coeffs.b + coeffs.c) * (coeffs.a + coeffs.b + coeffs.c) = 1) :
    HasExactColumnGram (newtonSchulzStep coeffs Q) := by
  rw [newtonSchulzStep_eq_scale_sum_of_exact_column_gram coeffs Q hgram]
  exact scale_hasExactColumnGram_of_square_eq_one Q (coeffs.a + coeffs.b + coeffs.c) hgram hsquare

/--
If `QᵀQ = I` and `|(a + b + c)^2 - 1| ≤ eps`, then one column-oriented Newton-Schulz step has
entrywise Gram residual bounded by `eps`.
-/
theorem newtonSchulzStep_hasApproxColumnGram_of_exact_column_gram_of_sum_square_error {m n : Nat}
    (coeffs : NewtonSchulzCoeffs ℝ) (Q : MatrixTensor ℝ m n) (eps : ℝ)
    (hgram : HasExactColumnGram Q)
    (herr :
      MathFunctions.abs
        ((coeffs.a + coeffs.b + coeffs.c) * (coeffs.a + coeffs.b + coeffs.c) - 1) ≤ eps)
    (heps : 0 ≤ eps) :
    HasApproxColumnGram eps (newtonSchulzStep coeffs Q) := by
  rw [newtonSchulzStep_eq_scale_sum_of_exact_column_gram coeffs Q hgram]
  exact scale_hasApproxColumnGram_of_exact_column_gram_of_square_error
    Q (coeffs.a + coeffs.b + coeffs.c) eps hgram herr heps

/--
If the Newton-Schulz coefficients sum to one, exact column Gram is enough to satisfy the exact
fixed-point backend's success predicate.
-/
theorem newtonSchulzFixedPointCheckedExact_success_of_coeff_sum_one {m n : Nat}
    (coeffs : NewtonSchulzCoeffs ℝ) (steps : Nat) (buffer : MatrixTensor ℝ m n)
    (hgram : HasExactColumnGram buffer)
    (hsum : coeffs.a + coeffs.b + coeffs.c = 1) :
    (newtonSchulzFixedPointCheckedExactOrthogonalizer
      (α := ℝ) (m := m) (n := n) coeffs steps).Success buffer := by
  exact ⟨hgram, newtonSchulzFixedPoint_of_exact_column_gram_of_coeff_sum_one coeffs buffer hgram hsum⟩

/--
For real coefficients with `a + b + c = 1`, exact column Gram of the fresh momentum buffer is
enough to certify a Newton-Schulz Muon update exactly.
-/
theorem update_has_exact_certified_step_newtonSchulz_exact_gram_checked {m n : Nat}
    (coeffs : NewtonSchulzCoeffs ℝ) (steps : Nat)
    (lr momentum : ℝ) (buf params grads : MatrixTensor ℝ m n)
    (hgram :
      HasExactColumnGram
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := ℝ) (m := m) (n := n) coeffs steps } :
            State ℝ (.dim m (.dim n .scalar)))
          params grads).1.buf)
    (hsum : coeffs.a + coeffs.b + coeffs.c = 1) :
    ∃ direction : MatrixTensor ℝ m n,
      ExactCertifiedStep
        ({ lr := lr, momentum := momentum, buf := buf,
           orthogonalizer :=
            newtonSchulzOrthogonalizer (α := ℝ) (m := m) (n := n) coeffs steps } :
          State ℝ (.dim m (.dim n .scalar)))
        params grads direction := by
  exact update_has_exact_certified_step_newtonSchulz_fixed_checked
    (coeffs := coeffs) (steps := steps) (lr := lr) (momentum := momentum)
    (buf := buf) (params := params) (grads := grads)
    (newtonSchulzFixedPointCheckedExact_success_of_coeff_sum_one
      coeffs steps
      (update
        ({ lr := lr, momentum := momentum, buf := buf,
           orthogonalizer :=
            newtonSchulzOrthogonalizer (α := ℝ) (m := m) (n := n) coeffs steps } :
          State ℝ (.dim m (.dim n .scalar)))
        params grads).1.buf hgram hsum)

/--
For real coefficients with `a + b + c = 1`, exact column Gram of the fresh momentum buffer gives
`QᵀQ = I` for the actual Newton-Schulz update direction.
-/
theorem update_newtonSchulz_exact_gram_direction_has_exact_column_gram_checked {m n : Nat}
    (coeffs : NewtonSchulzCoeffs ℝ) (steps : Nat)
    (lr momentum : ℝ) (buf params grads : MatrixTensor ℝ m n)
    (hgram :
      HasExactColumnGram
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := ℝ) (m := m) (n := n) coeffs steps } :
            State ℝ (.dim m (.dim n .scalar)))
          params grads).1.buf)
    (hsum : coeffs.a + coeffs.b + coeffs.c = 1) :
    HasExactColumnGram
      ((newtonSchulzOrthogonalizer (α := ℝ) (m := m) (n := n) coeffs steps).apply
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := ℝ) (m := m) (n := n) coeffs steps } :
            State ℝ (.dim m (.dim n .scalar)))
          params grads).1.buf) := by
  exact update_newtonSchulz_fixed_direction_has_exact_column_gram_checked
    (coeffs := coeffs) (steps := steps) (lr := lr) (momentum := momentum)
    (buf := buf) (params := params) (grads := grads)
    (newtonSchulzFixedPointCheckedExact_success_of_coeff_sum_one
      coeffs steps
      (update
        ({ lr := lr, momentum := momentum, buf := buf,
           orthogonalizer :=
            newtonSchulzOrthogonalizer (α := ℝ) (m := m) (n := n) coeffs steps } :
          State ℝ (.dim m (.dim n .scalar)))
        params grads).1.buf hgram hsum)

/--
Initialized version: exact column Gram of the first fresh momentum buffer and `a + b + c = 1`
certify the first Newton-Schulz Muon step exactly.
-/
theorem init_has_exact_certified_step_newtonSchulz_exact_gram_checked {m n : Nat}
    (coeffs : NewtonSchulzCoeffs ℝ) (steps : Nat)
    (lr momentum : ℝ) (params grads : MatrixTensor ℝ m n)
    (hgram :
      HasExactColumnGram
        (update
          (init lr momentum
            (newtonSchulzOrthogonalizer (α := ℝ) (m := m) (n := n) coeffs steps)
            params)
          params grads).1.buf)
    (hsum : coeffs.a + coeffs.b + coeffs.c = 1) :
    ∃ direction : MatrixTensor ℝ m n,
      ExactCertifiedStep
        (init lr momentum
          (newtonSchulzOrthogonalizer (α := ℝ) (m := m) (n := n) coeffs steps)
          params)
        params grads direction := by
  exact init_has_exact_certified_step_newtonSchulz_fixed_checked
    (coeffs := coeffs) (steps := steps) (lr := lr) (momentum := momentum)
    (params := params) (grads := grads)
    (newtonSchulzFixedPointCheckedExact_success_of_coeff_sum_one
      coeffs steps
      (update
        (init lr momentum
          (newtonSchulzOrthogonalizer (α := ℝ) (m := m) (n := n) coeffs steps)
          params)
        params grads).1.buf hgram hsum)

/--
The QR orthogonalizer satisfies the exact Muon direction contract whenever the executable QR pivots
of the input buffer are positive.
-/
theorem qrOrthogonalizer_exact_of_positive_pivots {m n : Nat}
    (buffer : MatrixTensor ℝ m n)
    (hpivots : HasPositiveQRPivots buffer) :
    ExactOrthogonalizesBuffer (qrOrthogonalizer (m := m) (n := n)) buffer := by
  unfold ExactOrthogonalizesBuffer HasExactColumnGram columnGram qrOrthogonalizer
  apply matrix_ext
  intro i j
  calc
    get2 (matMulSpec (Spec.Tensor.matrixTransposeSpec (qrQSpec buffer)) (qrQSpec buffer)) i j
        = ∑ k : Fin m, get2 (Spec.Tensor.matrixTransposeSpec (qrQSpec buffer)) i k *
            get2 (qrQSpec buffer) k j := by
          simpa using
            (get2_mat_mul_spec
              (A := Spec.Tensor.matrixTransposeSpec (qrQSpec buffer))
              (B := qrQSpec buffer) (i := i) (j := j))
    _ = ∑ k : Fin m, get2 (qrQSpec buffer) k i * get2 (qrQSpec buffer) k j := by
          refine Finset.sum_congr rfl ?_
          intro k _
          rw [get2_matrix_transpose_spec]
    _ = if i = j then 1 else 0 := by
          exact Spec.Factorization.Reconstruction.qrSpec_orthonormal buffer hpivots i j
    _ = get2 (identityTensorSpec (α := ℝ) n) i j := by
          rw [get2_identityTensorSpec_real]

/-- QR packaged as a checked exact Muon backend. -/
noncomputable def qrCheckedExactOrthogonalizer {m n : Nat} :
    CheckedExactOrthogonalizer ℝ m n :=
  { orthogonalizer := qrOrthogonalizer (m := m) (n := n)
    Success := HasPositiveQRPivots
    certified := fun buffer hpivots =>
      qrOrthogonalizer_exact_of_positive_pivots buffer hpivots }

/-- The QR checked backend succeeds exactly when the executable QR pivots are positive. -/
theorem qrCheckedExactOrthogonalizer_success_iff {m n : Nat}
    (buffer : MatrixTensor ℝ m n) :
    (qrCheckedExactOrthogonalizer (m := m) (n := n)).Success buffer ↔
      HasPositiveQRPivots buffer :=
  Iff.rfl

/--
Concrete QR-backed Muon theorem: if the fresh momentum buffer has positive QR pivots, the executable
Muon update uses a direction with exact column Gram `I`.
-/
theorem update_has_exact_certified_direction_qr {m n : Nat}
    (lr momentum : ℝ) (buf params grads : MatrixTensor ℝ m n)
    (hpivots :
      HasPositiveQRPivots
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := qrOrthogonalizer (m := m) (n := n) } :
            State ℝ (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    ∃ direction : MatrixTensor ℝ m n,
      ExactCertifiedDirection (qrOrthogonalizer (m := m) (n := n))
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := qrOrthogonalizer (m := m) (n := n) } :
            State ℝ (.dim m (.dim n .scalar)))
          params grads).1.buf direction ∧
      (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := qrOrthogonalizer (m := m) (n := n) } :
            State ℝ (.dim m (.dim n .scalar)))
          params grads).2 =
        subSpec params (scaleSpec direction lr) := by
  exact update_has_exact_certified_direction_of_checked_backend
    (backend := qrCheckedExactOrthogonalizer (m := m) (n := n))
    (lr := lr) (momentum := momentum) (buf := buf) (params := params) (grads := grads)
    hpivots

/--
Concrete QR-backed Muon step theorem: if the fresh momentum buffer has positive QR pivots, the
executable Muon update has a certified exact step.
-/
theorem update_has_exact_certified_step_qr {m n : Nat}
    (lr momentum : ℝ) (buf params grads : MatrixTensor ℝ m n)
    (hpivots :
      HasPositiveQRPivots
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := qrOrthogonalizer (m := m) (n := n) } :
            State ℝ (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    ∃ direction : MatrixTensor ℝ m n,
      ExactCertifiedStep
        ({ lr := lr, momentum := momentum, buf := buf,
           orthogonalizer := qrOrthogonalizer (m := m) (n := n) } :
          State ℝ (.dim m (.dim n .scalar)))
        params grads direction := by
  exact update_has_exact_certified_step_of_checked_backend
    (backend := qrCheckedExactOrthogonalizer (m := m) (n := n))
    (lr := lr) (momentum := momentum) (buf := buf) (params := params) (grads := grads)
    hpivots

/--
Concrete QR-backed direction theorem: if the fresh momentum buffer has positive QR pivots, the
actual direction used by the Muon update has column Gram `I`.
-/
theorem update_qr_direction_has_exact_column_gram {m n : Nat}
    (lr momentum : ℝ) (buf params grads : MatrixTensor ℝ m n)
    (hpivots :
      HasPositiveQRPivots
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := qrOrthogonalizer (m := m) (n := n) } :
            State ℝ (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    HasExactColumnGram
      ((qrOrthogonalizer (m := m) (n := n)).apply
        (update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := qrOrthogonalizer (m := m) (n := n) } :
            State ℝ (.dim m (.dim n .scalar)))
          params grads).1.buf) := by
  exact update_direction_has_exact_column_gram_of_checked_backend
    (backend := qrCheckedExactOrthogonalizer (m := m) (n := n))
    (lr := lr) (momentum := momentum) (buf := buf) (params := params) (grads := grads)
    hpivots

/--
Initialized QR-backed Muon theorem: if the first fresh momentum buffer has positive QR pivots, the
first initialized Muon update uses a direction with exact column Gram `I`.
-/
theorem init_has_exact_certified_direction_qr {m n : Nat}
    (lr momentum : ℝ) (params grads : MatrixTensor ℝ m n)
    (hpivots :
      HasPositiveQRPivots
        (update
          (init lr momentum (qrOrthogonalizer (m := m) (n := n)) params)
          params grads).1.buf) :
    ∃ direction : MatrixTensor ℝ m n,
      ExactCertifiedDirection (qrOrthogonalizer (m := m) (n := n))
        (update
          (init lr momentum (qrOrthogonalizer (m := m) (n := n)) params)
          params grads).1.buf direction ∧
      (update
          (init lr momentum (qrOrthogonalizer (m := m) (n := n)) params)
          params grads).2 =
        subSpec params (scaleSpec direction lr) := by
  exact init_has_exact_certified_direction_of_checked_backend
    (backend := qrCheckedExactOrthogonalizer (m := m) (n := n))
    (lr := lr) (momentum := momentum) (params := params) (grads := grads)
    hpivots

/--
Initialized QR-backed Muon step theorem: if the first fresh momentum buffer has positive QR pivots,
the first initialized Muon update has a certified exact step.
-/
theorem init_has_exact_certified_step_qr {m n : Nat}
    (lr momentum : ℝ) (params grads : MatrixTensor ℝ m n)
    (hpivots :
      HasPositiveQRPivots
        (update
          (init lr momentum (qrOrthogonalizer (m := m) (n := n)) params)
          params grads).1.buf) :
    ∃ direction : MatrixTensor ℝ m n,
      ExactCertifiedStep
        (init lr momentum (qrOrthogonalizer (m := m) (n := n)) params)
        params grads direction := by
  exact init_has_exact_certified_step_of_checked_backend
    (backend := qrCheckedExactOrthogonalizer (m := m) (n := n))
    (lr := lr) (momentum := momentum) (params := params) (grads := grads)
    hpivots

/--
Initialized QR-backed direction theorem: if the first fresh momentum buffer has positive QR pivots,
the first initialized Muon update direction has column Gram `I`.
-/
theorem init_qr_direction_has_exact_column_gram {m n : Nat}
    (lr momentum : ℝ) (params grads : MatrixTensor ℝ m n)
    (hpivots :
      HasPositiveQRPivots
        (update
          (init lr momentum (qrOrthogonalizer (m := m) (n := n)) params)
          params grads).1.buf) :
    HasExactColumnGram
      ((qrOrthogonalizer (m := m) (n := n)).apply
        (update
          (init lr momentum (qrOrthogonalizer (m := m) (n := n)) params)
          params grads).1.buf) := by
  exact init_direction_has_exact_column_gram_of_checked_backend
    (backend := qrCheckedExactOrthogonalizer (m := m) (n := n))
    (lr := lr) (momentum := momentum) (params := params) (grads := grads)
    hpivots

end Muon

end Optim

namespace TorchLean

namespace optim

namespace muon

/-- Matrix-shaped tensor used by the proof layer Muon contracts. -/
abbrev MatrixTensor (α : Type) (m n : Nat) := _root_.Optim.Muon.MatrixTensor α m n

/-- Muon optimizer state for a tensor of shape `s`. -/
abbrev State (α : Type) (s : _root_.Spec.Shape) := _root_.Optim.Muon.State α s

/-- Orthogonalization backend used by Muon. -/
abbrev Orthogonalizer (α : Type) (s : _root_.Spec.Shape) :=
  _root_.Optim.Muon.Orthogonalizer α s

/-- The identity Muon orthogonalizer. -/
def identityOrthogonalizer {α : Type} {s : _root_.Spec.Shape} :
    Orthogonalizer α s :=
  _root_.Optim.Muon.identityOrthogonalizer (α := α) (s := s)

/-- The proof layer Muon initialization equation. -/
abbrev initSpec {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : _root_.Spec.Shape}
    (lr momentum : α) (orthogonalizer : Orthogonalizer α s)
    (params : _root_.Spec.Tensor α s) : State α s :=
  _root_.Optim.Muon.initSpec lr momentum orthogonalizer params

/-- The executable Muon initializer follows `initSpec`. -/
theorem init_eq_spec {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : _root_.Spec.Shape}
    (lr momentum : α) (orthogonalizer : Orthogonalizer α s)
    (params : _root_.Spec.Tensor α s) :
    _root_.Optim.Muon.init lr momentum orthogonalizer params =
      initSpec lr momentum orthogonalizer params :=
  _root_.Optim.Muon.init_eq_spec lr momentum orthogonalizer params

/-- Muon initialization stores the requested learning rate. -/
theorem init_lr_eq {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : _root_.Spec.Shape}
    (lr momentum : α) (orthogonalizer : Orthogonalizer α s)
    (params : _root_.Spec.Tensor α s) :
    (_root_.Optim.Muon.init lr momentum orthogonalizer params).lr = lr :=
  _root_.Optim.Muon.init_lr_eq lr momentum orthogonalizer params

/-- Muon initialization stores the requested momentum coefficient. -/
theorem init_momentum_eq {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : _root_.Spec.Shape}
    (lr momentum : α) (orthogonalizer : Orthogonalizer α s)
    (params : _root_.Spec.Tensor α s) :
    (_root_.Optim.Muon.init lr momentum orthogonalizer params).momentum = momentum :=
  _root_.Optim.Muon.init_momentum_eq lr momentum orthogonalizer params

/-- Muon initialization starts from the all-zero momentum buffer. -/
theorem init_buffer_eq_zero {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : _root_.Spec.Shape}
    (lr momentum : α) (orthogonalizer : Orthogonalizer α s)
    (params : _root_.Spec.Tensor α s) :
    (_root_.Optim.Muon.init lr momentum orthogonalizer params).buf =
      _root_.Spec.fill 0 s :=
  _root_.Optim.Muon.init_buffer_eq_zero lr momentum orthogonalizer params

/-- Muon initialization stores exactly the orthogonalizer backend supplied by the caller. -/
theorem init_orthogonalizer_eq {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : _root_.Spec.Shape}
    (lr momentum : α) (orthogonalizer : Orthogonalizer α s)
    (params : _root_.Spec.Tensor α s) :
    (_root_.Optim.Muon.init lr momentum orthogonalizer params).orthogonalizer = orthogonalizer :=
  _root_.Optim.Muon.init_orthogonalizer_eq lr momentum orthogonalizer params

/-- After initialization, Muon's first step stores the momentum update from the zero buffer. -/
theorem init_update_buffer_eq_spec {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : _root_.Spec.Shape}
    (lr momentum : α) (orthogonalizer : Orthogonalizer α s)
    (params grads : _root_.Spec.Tensor α s) :
    (_root_.Optim.Muon.update
        (_root_.Optim.Muon.init lr momentum orthogonalizer params) params grads).1.buf =
      _root_.Optim.OptimizerUtils.updateMomentumBuf (_root_.Spec.fill 0 s) momentum grads :=
  _root_.Optim.Muon.init_update_buffer_eq_spec lr momentum orthogonalizer params grads

/--
After initialization, Muon's first parameter update applies the supplied backend to the fresh
momentum buffer computed from the zero buffer.
-/
theorem init_update_params_eq_spec {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : _root_.Spec.Shape}
    (lr momentum : α) (orthogonalizer : Orthogonalizer α s)
    (params grads : _root_.Spec.Tensor α s) :
    (_root_.Optim.Muon.update
        (_root_.Optim.Muon.init lr momentum orthogonalizer params) params grads).2 =
      let newBuf := _root_.Optim.OptimizerUtils.updateMomentumBuf (_root_.Spec.fill 0 s) momentum grads
      _root_.Spec.Tensor.subSpec params
        (_root_.Spec.Tensor.scaleSpec (orthogonalizer.apply newBuf) lr) :=
  _root_.Optim.Muon.init_update_params_eq_spec lr momentum orthogonalizer params grads

/-- The proof layer Muon update equation. -/
abbrev updateSpec {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : _root_.Spec.Shape}
    (state : State α s) (params grads : _root_.Spec.Tensor α s) :
    State α s × _root_.Spec.Tensor α s :=
  _root_.Optim.Muon.updateSpec state params grads

/-- The executable Muon update follows `updateSpec`. -/
theorem update_eq_spec {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : _root_.Spec.Shape}
    (state : State α s) (params grads : _root_.Spec.Tensor α s) :
    _root_.Optim.Muon.update state params grads = updateSpec state params grads :=
  _root_.Optim.Muon.update_eq_spec state params grads

/-- Muon's next state is the old state with only the momentum buffer replaced. -/
theorem update_state_eq_spec {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : _root_.Spec.Shape}
    (state : State α s) (params grads : _root_.Spec.Tensor α s) :
    (_root_.Optim.Muon.update state params grads).1 =
      { state with
        buf := _root_.Optim.OptimizerUtils.updateMomentumBuf state.buf state.momentum grads } :=
  _root_.Optim.Muon.update_state_eq_spec state params grads

/-- After initialization, Muon's first next state has the requested scalars/backend and fresh buffer. -/
theorem init_update_state_eq_spec {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : _root_.Spec.Shape}
    (lr momentum : α) (orthogonalizer : Orthogonalizer α s)
    (params grads : _root_.Spec.Tensor α s) :
    (_root_.Optim.Muon.update
        (_root_.Optim.Muon.init lr momentum orthogonalizer params) params grads).1 =
      ({ lr := lr, momentum := momentum,
         buf := _root_.Optim.OptimizerUtils.updateMomentumBuf (_root_.Spec.fill 0 s) momentum grads,
         orthogonalizer := orthogonalizer } : State α s) :=
  _root_.Optim.Muon.init_update_state_eq_spec lr momentum orthogonalizer params grads

/-- Muon's next buffer is the momentum update `momentum * buf + grad`. -/
theorem update_buffer_eq_spec {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : _root_.Spec.Shape}
    (state : State α s) (params grads : _root_.Spec.Tensor α s) :
    (_root_.Optim.Muon.update state params grads).1.buf =
      _root_.Optim.OptimizerUtils.updateMomentumBuf state.buf state.momentum grads :=
  _root_.Optim.Muon.update_buffer_eq_spec state params grads

/-- A Muon update preserves the learning rate stored in the optimizer state. -/
theorem update_lr_eq {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : _root_.Spec.Shape}
    (state : State α s) (params grads : _root_.Spec.Tensor α s) :
    (_root_.Optim.Muon.update state params grads).1.lr = state.lr :=
  _root_.Optim.Muon.update_lr_eq state params grads

/-- A Muon update preserves the momentum coefficient stored in the optimizer state. -/
theorem update_momentum_eq {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : _root_.Spec.Shape}
    (state : State α s) (params grads : _root_.Spec.Tensor α s) :
    (_root_.Optim.Muon.update state params grads).1.momentum = state.momentum :=
  _root_.Optim.Muon.update_momentum_eq state params grads

/-- A Muon update preserves the configured orthogonalizer backend. -/
theorem update_orthogonalizer_eq {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : _root_.Spec.Shape}
    (state : State α s) (params grads : _root_.Spec.Tensor α s) :
    (_root_.Optim.Muon.update state params grads).1.orthogonalizer = state.orthogonalizer :=
  _root_.Optim.Muon.update_orthogonalizer_eq state params grads

/-- Muon's parameter update uses the orthogonalizer applied to the fresh momentum buffer. -/
theorem update_params_eq_spec {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : _root_.Spec.Shape}
    (state : State α s) (params grads : _root_.Spec.Tensor α s) :
    (_root_.Optim.Muon.update state params grads).2 =
      let newBuf :=
        _root_.Optim.OptimizerUtils.updateMomentumBuf state.buf state.momentum grads
      _root_.Spec.Tensor.subSpec params
        (_root_.Spec.Tensor.scaleSpec (state.orthogonalizer.apply newBuf) state.lr) :=
  _root_.Optim.Muon.update_params_eq_spec state params grads

/--
For any orthogonalizer backend, Muon's stored momentum buffer evolves exactly like momentum SGD.
The backend changes the parameter direction, not the buffer recurrence.
-/
theorem update_buffer_eq_momentumSGD
    {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    {s : _root_.Spec.Shape}
    (state : State α s) (params grads : _root_.Spec.Tensor α s) :
    (_root_.Optim.Muon.update state params grads).1.buf =
      (_root_.Optim.MomentumSGD.update
        ({ lr := state.lr, momentum := state.momentum, buf := state.buf } :
          _root_.Optim.MomentumSGD.State α s)
        params grads).1.buf :=
  _root_.Optim.Muon.update_buffer_eq_momentumSGD state params grads

/--
Starting from initialized states, Muon's first stored momentum buffer agrees with momentum SGD for
any orthogonalizer backend.
-/
theorem init_update_buffer_eq_momentumSGD
    {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    {s : _root_.Spec.Shape} (lr momentum : α)
    (orthogonalizer : Orthogonalizer α s) (params grads : _root_.Spec.Tensor α s) :
    (_root_.Optim.Muon.update
        (_root_.Optim.Muon.init lr momentum orthogonalizer params) params grads).1.buf =
      (_root_.Optim.MomentumSGD.update
        (_root_.Optim.MomentumSGD.init lr momentum params) params grads).1.buf :=
  _root_.Optim.Muon.init_update_buffer_eq_momentumSGD
    lr momentum orthogonalizer params grads

/--
If a Muon backend returns the fresh momentum buffer unchanged on this step, then the parameter
update agrees with momentum SGD for this step.
-/
theorem update_params_eq_momentumSGD_of_apply_eq
    {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    {s : _root_.Spec.Shape}
    (state : State α s) (params grads : _root_.Spec.Tensor α s)
    (happly :
      state.orthogonalizer.apply
        (_root_.Optim.OptimizerUtils.updateMomentumBuf state.buf state.momentum grads) =
        _root_.Optim.OptimizerUtils.updateMomentumBuf state.buf state.momentum grads) :
    (_root_.Optim.Muon.update state params grads).2 =
      (_root_.Optim.MomentumSGD.update
        ({ lr := state.lr, momentum := state.momentum, buf := state.buf } :
          _root_.Optim.MomentumSGD.State α s)
        params grads).2 :=
  _root_.Optim.Muon.update_params_eq_momentumSGD_of_apply_eq
    state params grads happly

/-- Initialized version of `update_params_eq_momentumSGD_of_apply_eq`. -/
theorem init_update_params_eq_momentumSGD_of_apply_eq
    {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    {s : _root_.Spec.Shape} (lr momentum : α)
    (orthogonalizer : Orthogonalizer α s) (params grads : _root_.Spec.Tensor α s)
    (happly :
      orthogonalizer.apply (_root_.Optim.OptimizerUtils.updateMomentumBuf (_root_.Spec.fill 0 s) momentum grads) =
        _root_.Optim.OptimizerUtils.updateMomentumBuf (_root_.Spec.fill 0 s) momentum grads) :
    (_root_.Optim.Muon.update
        (_root_.Optim.Muon.init lr momentum orthogonalizer params) params grads).2 =
      (_root_.Optim.MomentumSGD.update
        (_root_.Optim.MomentumSGD.init lr momentum params) params grads).2 :=
  _root_.Optim.Muon.init_update_params_eq_momentumSGD_of_apply_eq
    lr momentum orthogonalizer params grads happly

/-- With the identity orthogonalizer, Muon has the same parameter update as momentum SGD. -/
theorem update_identity_params_eq_momentumSGD_spec
    {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    {s : _root_.Spec.Shape} (lr momentum : α)
    (buf params grads : _root_.Spec.Tensor α s) :
    (_root_.Optim.Muon.update
        ({ lr := lr, momentum := momentum, buf := buf,
           orthogonalizer := identityOrthogonalizer (α := α) (s := s) } : State α s)
        params grads).2 =
      (_root_.Optim.MomentumSGD.update
        ({ lr := lr, momentum := momentum, buf := buf } :
          _root_.Optim.MomentumSGD.State α s)
        params grads).2 :=
  _root_.Optim.Muon.update_identity_params_eq_momentumSGD_spec lr momentum buf params grads

/--
Starting from initialized states, identity-backend Muon stores the same next buffer as momentum SGD.
-/
theorem init_identity_update_buffer_eq_momentumSGD
    {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    {s : _root_.Spec.Shape} (lr momentum : α)
    (params grads : _root_.Spec.Tensor α s) :
    (_root_.Optim.Muon.update
        (_root_.Optim.Muon.init lr momentum
          (identityOrthogonalizer (α := α) (s := s)) params)
        params grads).1.buf =
      (_root_.Optim.MomentumSGD.update
        (_root_.Optim.MomentumSGD.init lr momentum params) params grads).1.buf :=
  _root_.Optim.Muon.init_identity_update_buffer_eq_momentumSGD lr momentum params grads

/--
Starting from initialized states, identity-backend Muon has the same parameter update as momentum
SGD.
-/
theorem init_identity_update_params_eq_momentumSGD
    {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    {s : _root_.Spec.Shape} (lr momentum : α)
    (params grads : _root_.Spec.Tensor α s) :
    (_root_.Optim.Muon.update
        (_root_.Optim.Muon.init lr momentum
          (identityOrthogonalizer (α := α) (s := s)) params)
        params grads).2 =
      (_root_.Optim.MomentumSGD.update
        (_root_.Optim.MomentumSGD.init lr momentum params) params grads).2 :=
  _root_.Optim.Muon.init_identity_update_params_eq_momentumSGD lr momentum params grads

/-- Column Gram matrix `QᵀQ` for a matrix update direction. -/
abbrev columnGram {α : Type} [Context α] {m n : Nat}
    (Q : MatrixTensor α m n) : MatrixTensor α n n :=
  _root_.Optim.Muon.columnGram Q

/-- Residual matrix `QᵀQ - I` for a matrix update direction. -/
abbrev columnGramResidual {α : Type} [Context α] {m n : Nat}
    (Q : MatrixTensor α m n) : MatrixTensor α n n :=
  _root_.Optim.Muon.columnGramResidual Q

@[inherit_doc _root_.Optim.Muon.columnGramResidual_eq]
theorem columnGramResidual_eq {α : Type} [Context α] {m n : Nat}
    (Q : MatrixTensor α m n) :
    columnGramResidual Q =
      _root_.Spec.Tensor.subSpec (columnGram Q) (_root_.Spec.identityTensorSpec n) :=
  _root_.Optim.Muon.columnGramResidual_eq Q

/-- Exact Muon direction contract: the direction has column Gram matrix `I`. -/
abbrev HasExactColumnGram {α : Type} [Context α] {m n : Nat}
    (Q : MatrixTensor α m n) : Prop :=
  _root_.Optim.Muon.HasExactColumnGram Q

@[inherit_doc _root_.Optim.Muon.hasExactColumnGram_iff]
theorem hasExactColumnGram_iff {α : Type} [Context α] {m n : Nat}
    (Q : MatrixTensor α m n) :
    HasExactColumnGram Q ↔ columnGram Q = _root_.Spec.identityTensorSpec n :=
  _root_.Optim.Muon.hasExactColumnGram_iff Q

/-- Approximate Muon direction contract: `QᵀQ - I` is entrywise bounded. -/
abbrev HasApproxColumnGram {α : Type} [Context α] {m n : Nat}
    (eps : α) (Q : MatrixTensor α m n) : Prop :=
  _root_.Optim.Muon.HasApproxColumnGram eps Q

@[inherit_doc _root_.Optim.Muon.hasApproxColumnGram_iff]
theorem hasApproxColumnGram_iff {α : Type} [Context α] {m n : Nat}
    (eps : α) (Q : MatrixTensor α m n) :
    HasApproxColumnGram eps Q ↔
      ∀ i : Fin n, ∀ j : Fin n,
        _root_.MathFunctions.abs (_root_.Spec.get2 (columnGramResidual Q) i j) ≤ eps :=
  _root_.Optim.Muon.hasApproxColumnGram_iff eps Q

/-- Exact orthogonalizer contract for every matrix buffer of a fixed shape. -/
abbrev ExactMatrixOrthogonalizer {α : Type} [Context α] {m n : Nat}
    (orthogonalizer : _root_.Optim.Muon.Orthogonalizer α (.dim m (.dim n .scalar))) : Prop :=
  _root_.Optim.Muon.ExactMatrixOrthogonalizer orthogonalizer

/-- Exact orthogonalization contract for one concrete momentum buffer. -/
abbrev ExactOrthogonalizesBuffer {α : Type} [Context α] {m n : Nat}
    (orthogonalizer : _root_.Optim.Muon.Orthogonalizer α (.dim m (.dim n .scalar)))
    (buffer : MatrixTensor α m n) : Prop :=
  _root_.Optim.Muon.ExactOrthogonalizesBuffer orthogonalizer buffer

@[inherit_doc _root_.Optim.Muon.exactOrthogonalizesBuffer_iff]
theorem exactOrthogonalizesBuffer_iff {α : Type} [Context α] {m n : Nat}
    (orthogonalizer : _root_.Optim.Muon.Orthogonalizer α (.dim m (.dim n .scalar)))
    (buffer : MatrixTensor α m n) :
    ExactOrthogonalizesBuffer orthogonalizer buffer ↔
      HasExactColumnGram (orthogonalizer.apply buffer) :=
  _root_.Optim.Muon.exactOrthogonalizesBuffer_iff orthogonalizer buffer

/-- Approximate orthogonalizer contract for every matrix buffer of a fixed shape. -/
abbrev ApproxMatrixOrthogonalizer {α : Type} [Context α] {m n : Nat} (eps : α)
    (orthogonalizer : _root_.Optim.Muon.Orthogonalizer α (.dim m (.dim n .scalar))) : Prop :=
  _root_.Optim.Muon.ApproxMatrixOrthogonalizer eps orthogonalizer

/-- Approximate orthogonalization contract for one concrete momentum buffer. -/
abbrev ApproxOrthogonalizesBuffer {α : Type} [Context α] {m n : Nat} (eps : α)
    (orthogonalizer : _root_.Optim.Muon.Orthogonalizer α (.dim m (.dim n .scalar)))
    (buffer : MatrixTensor α m n) : Prop :=
  _root_.Optim.Muon.ApproxOrthogonalizesBuffer eps orthogonalizer buffer

@[inherit_doc _root_.Optim.Muon.approxOrthogonalizesBuffer_iff]
theorem approxOrthogonalizesBuffer_iff {α : Type} [Context α] {m n : Nat} (eps : α)
    (orthogonalizer : _root_.Optim.Muon.Orthogonalizer α (.dim m (.dim n .scalar)))
    (buffer : MatrixTensor α m n) :
    ApproxOrthogonalizesBuffer eps orthogonalizer buffer ↔
      HasApproxColumnGram eps (orthogonalizer.apply buffer) :=
  _root_.Optim.Muon.approxOrthogonalizesBuffer_iff eps orthogonalizer buffer

@[inherit_doc _root_.Optim.Muon.exactMatrixOrthogonalizer_iff]
theorem exactMatrixOrthogonalizer_iff {α : Type} [Context α] {m n : Nat}
    (orthogonalizer : _root_.Optim.Muon.Orthogonalizer α (.dim m (.dim n .scalar))) :
    ExactMatrixOrthogonalizer (m := m) (n := n) orthogonalizer ↔
      ∀ buffer : MatrixTensor α m n, HasExactColumnGram (orthogonalizer.apply buffer) :=
  _root_.Optim.Muon.exactMatrixOrthogonalizer_iff orthogonalizer

@[inherit_doc _root_.Optim.Muon.approxMatrixOrthogonalizer_iff]
theorem approxMatrixOrthogonalizer_iff {α : Type} [Context α] {m n : Nat} (eps : α)
    (orthogonalizer : _root_.Optim.Muon.Orthogonalizer α (.dim m (.dim n .scalar))) :
    ApproxMatrixOrthogonalizer (m := m) (n := n) eps orthogonalizer ↔
      ∀ buffer : MatrixTensor α m n, HasApproxColumnGram eps (orthogonalizer.apply buffer) :=
  _root_.Optim.Muon.approxMatrixOrthogonalizer_iff eps orthogonalizer

/-- Unconditionally certified exact Muon backend. -/
abbrev ExactCertifiedOrthogonalizer :=
  _root_.Optim.Muon.ExactCertifiedOrthogonalizer

/-- Unconditionally certified approximate Muon backend. -/
abbrev ApproxCertifiedOrthogonalizer :=
  _root_.Optim.Muon.ApproxCertifiedOrthogonalizer

/-- Checked exact Muon backend with a backend-specific success predicate. -/
abbrev CheckedExactOrthogonalizer :=
  _root_.Optim.Muon.CheckedExactOrthogonalizer

/-- Checked approximate Muon backend with a backend-specific success predicate. -/
abbrev CheckedApproxOrthogonalizer :=
  _root_.Optim.Muon.CheckedApproxOrthogonalizer

/-- Coefficients for the Newton-Schulz polynomial backend. -/
abbrev NewtonSchulzCoeffs := _root_.Optim.Muon.NewtonSchulzCoeffs

/-- Left Gram matrix `XXᵀ`, used by the Newton-Schulz backend. -/
abbrev leftGram {α : Type} [Context α] {m n : Nat}
    (X : MatrixTensor α m n) : MatrixTensor α m m :=
  _root_.Optim.Muon.leftGram X

@[inherit_doc _root_.Optim.Muon.leftGram_eq]
theorem leftGram_eq {α : Type} [Context α] {m n : Nat}
    (X : MatrixTensor α m n) :
    leftGram X = _root_.Spec.matMulSpec X (_root_.Spec.Tensor.matrixTransposeSpec X) :=
  _root_.Optim.Muon.leftGram_eq X

/-- Right/column Gram matrix `XᵀX`, matching the Muon column-Gram certificate. -/
abbrev rightGram {α : Type} [Context α] {m n : Nat}
    (X : MatrixTensor α m n) : MatrixTensor α n n :=
  _root_.Optim.Muon.rightGram X

@[inherit_doc _root_.Optim.Muon.rightGram_eq]
theorem rightGram_eq {α : Type} [Context α] {m n : Nat}
    (X : MatrixTensor α m n) :
    rightGram X = _root_.Spec.matMulSpec (_root_.Spec.Tensor.matrixTransposeSpec X) X :=
  _root_.Optim.Muon.rightGram_eq X

/-- One row-oriented Newton-Schulz polynomial step using `XXᵀ`. -/
abbrev newtonSchulzLeftStep {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (X : MatrixTensor α m n) : MatrixTensor α m n :=
  _root_.Optim.Muon.newtonSchulzLeftStep coeffs X

@[inherit_doc _root_.Optim.Muon.newtonSchulzLeftStep_eq]
theorem newtonSchulzLeftStep_eq {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (X : MatrixTensor α m n) :
    newtonSchulzLeftStep coeffs X =
      let G := leftGram X
      let GX := _root_.Spec.matMulSpec G X
      let G2X := _root_.Spec.matMulSpec G GX
      _root_.Spec.Tensor.addSpec
        (_root_.Spec.Tensor.addSpec (_root_.Spec.Tensor.scaleSpec X coeffs.a)
          (_root_.Spec.Tensor.scaleSpec GX coeffs.b))
        (_root_.Spec.Tensor.scaleSpec G2X coeffs.c) :=
  _root_.Optim.Muon.newtonSchulzLeftStep_eq coeffs X

/-- One column-oriented Newton-Schulz polynomial step using `XᵀX`. -/
abbrev newtonSchulzStep {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (X : MatrixTensor α m n) : MatrixTensor α m n :=
  _root_.Optim.Muon.newtonSchulzStep coeffs X

@[inherit_doc _root_.Optim.Muon.newtonSchulzStep_eq]
theorem newtonSchulzStep_eq {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (X : MatrixTensor α m n) :
    newtonSchulzStep coeffs X =
      let G := rightGram X
      let XG := _root_.Spec.matMulSpec X G
      let XG2 := _root_.Spec.matMulSpec XG G
      _root_.Spec.Tensor.addSpec
        (_root_.Spec.Tensor.addSpec (_root_.Spec.Tensor.scaleSpec X coeffs.a)
          (_root_.Spec.Tensor.scaleSpec XG coeffs.b))
        (_root_.Spec.Tensor.scaleSpec XG2 coeffs.c) :=
  _root_.Optim.Muon.newtonSchulzStep_eq coeffs X

/-- Fixed-count row-oriented Newton-Schulz polynomial iteration. -/
abbrev newtonSchulzLeftIter {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat)
    (X : MatrixTensor α m n) : MatrixTensor α m n :=
  _root_.Optim.Muon.newtonSchulzLeftIter coeffs steps X

@[inherit_doc _root_.Optim.Muon.newtonSchulzLeftIter_zero]
theorem newtonSchulzLeftIter_zero {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (X : MatrixTensor α m n) :
    newtonSchulzLeftIter coeffs 0 X = X :=
  _root_.Optim.Muon.newtonSchulzLeftIter_zero coeffs X

@[inherit_doc _root_.Optim.Muon.newtonSchulzLeftIter_succ]
theorem newtonSchulzLeftIter_succ {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) (X : MatrixTensor α m n) :
    newtonSchulzLeftIter coeffs (steps + 1) X =
      newtonSchulzLeftIter coeffs steps (newtonSchulzLeftStep coeffs X) :=
  _root_.Optim.Muon.newtonSchulzLeftIter_succ coeffs steps X

/-- Fixed-count column-oriented Newton-Schulz polynomial iteration. -/
abbrev newtonSchulzIter {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat)
    (X : MatrixTensor α m n) : MatrixTensor α m n :=
  _root_.Optim.Muon.newtonSchulzIter coeffs steps X

@[inherit_doc _root_.Optim.Muon.newtonSchulzIter_zero]
theorem newtonSchulzIter_zero {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (X : MatrixTensor α m n) :
    newtonSchulzIter coeffs 0 X = X :=
  _root_.Optim.Muon.newtonSchulzIter_zero coeffs X

@[inherit_doc _root_.Optim.Muon.newtonSchulzIter_succ]
theorem newtonSchulzIter_succ {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) (X : MatrixTensor α m n) :
    newtonSchulzIter coeffs (steps + 1) X =
      newtonSchulzIter coeffs steps (newtonSchulzStep coeffs X) :=
  _root_.Optim.Muon.newtonSchulzIter_succ coeffs steps X

/-- Row-oriented Newton-Schulz-shaped orthogonalizer backend. -/
abbrev newtonSchulzLeftOrthogonalizer {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) :
    _root_.Optim.Muon.Orthogonalizer α (.dim m (.dim n .scalar)) :=
  _root_.Optim.Muon.newtonSchulzLeftOrthogonalizer (α := α) (m := m) (n := n) coeffs steps

@[inherit_doc _root_.Optim.Muon.newtonSchulzLeftOrthogonalizer_apply]
theorem newtonSchulzLeftOrthogonalizer_apply {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) (buffer : MatrixTensor α m n) :
    (newtonSchulzLeftOrthogonalizer (α := α) (m := m) (n := n) coeffs steps).apply buffer =
      newtonSchulzLeftIter coeffs steps buffer :=
  _root_.Optim.Muon.newtonSchulzLeftOrthogonalizer_apply coeffs steps buffer

@[inherit_doc _root_.Optim.Muon.newtonSchulzLeftOrthogonalizer_zero_apply]
theorem newtonSchulzLeftOrthogonalizer_zero_apply
    {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (buffer : MatrixTensor α m n) :
    (newtonSchulzLeftOrthogonalizer (α := α) (m := m) (n := n) coeffs 0).apply buffer =
      buffer :=
  _root_.Optim.Muon.newtonSchulzLeftOrthogonalizer_zero_apply coeffs buffer

/-- Column-oriented Newton-Schulz-shaped orthogonalizer backend. -/
abbrev newtonSchulzOrthogonalizer {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) :
    _root_.Optim.Muon.Orthogonalizer α (.dim m (.dim n .scalar)) :=
  _root_.Optim.Muon.newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps

@[inherit_doc _root_.Optim.Muon.newtonSchulzOrthogonalizer_apply]
theorem newtonSchulzOrthogonalizer_apply {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) (buffer : MatrixTensor α m n) :
    (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps).apply buffer =
      newtonSchulzIter coeffs steps buffer :=
  _root_.Optim.Muon.newtonSchulzOrthogonalizer_apply coeffs steps buffer

@[inherit_doc _root_.Optim.Muon.newtonSchulzOrthogonalizer_zero_apply]
theorem newtonSchulzOrthogonalizer_zero_apply
    {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (buffer : MatrixTensor α m n) :
    (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs 0).apply buffer =
      buffer :=
  _root_.Optim.Muon.newtonSchulzOrthogonalizer_zero_apply coeffs buffer

/--
With zero Newton-Schulz steps, the Muon backend stores the same next buffer as momentum SGD.
-/
theorem update_newtonSchulz_zero_buffer_eq_momentumSGD
    {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α)
    (lr momentum : α) (buf params grads : MatrixTensor α m n) :
    (_root_.Optim.Muon.update
        ({ lr := lr, momentum := momentum, buf := buf,
           orthogonalizer := newtonSchulzOrthogonalizer
            (α := α) (m := m) (n := n) coeffs 0 } :
          _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
        params grads).1.buf =
      (_root_.Optim.MomentumSGD.update
        ({ lr := lr, momentum := momentum, buf := buf } :
          _root_.Optim.MomentumSGD.State α (.dim m (.dim n .scalar)))
        params grads).1.buf :=
  _root_.Optim.Muon.update_newtonSchulz_zero_buffer_eq_momentumSGD
    coeffs lr momentum buf params grads

/--
With zero Newton-Schulz steps, the Muon backend gives the same parameter update as momentum SGD.
-/
theorem update_newtonSchulz_zero_params_eq_momentumSGD
    {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α)
    (lr momentum : α) (buf params grads : MatrixTensor α m n) :
    (_root_.Optim.Muon.update
        ({ lr := lr, momentum := momentum, buf := buf,
           orthogonalizer := newtonSchulzOrthogonalizer
            (α := α) (m := m) (n := n) coeffs 0 } :
          _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
        params grads).2 =
      (_root_.Optim.MomentumSGD.update
        ({ lr := lr, momentum := momentum, buf := buf } :
          _root_.Optim.MomentumSGD.State α (.dim m (.dim n .scalar)))
        params grads).2 :=
  _root_.Optim.Muon.update_newtonSchulz_zero_params_eq_momentumSGD
    coeffs lr momentum buf params grads

/--
Starting from initialized states, a zero-step Newton-Schulz Muon backend stores the same next buffer
as momentum SGD.
-/
theorem init_newtonSchulz_zero_update_buffer_eq_momentumSGD
    {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α)
    (lr momentum : α) (params grads : MatrixTensor α m n) :
    (_root_.Optim.Muon.update
        (_root_.Optim.Muon.init lr momentum
          (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs 0)
          params)
        params grads).1.buf =
      (_root_.Optim.MomentumSGD.update
        (_root_.Optim.MomentumSGD.init lr momentum params) params grads).1.buf :=
  _root_.Optim.Muon.init_newtonSchulz_zero_update_buffer_eq_momentumSGD
    coeffs lr momentum params grads

/--
Starting from initialized states, a zero-step Newton-Schulz Muon backend has the same parameter
update as momentum SGD.
-/
theorem init_newtonSchulz_zero_update_params_eq_momentumSGD
    {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α)
    (lr momentum : α) (params grads : MatrixTensor α m n) :
    (_root_.Optim.Muon.update
        (_root_.Optim.Muon.init lr momentum
          (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs 0)
          params)
        params grads).2 =
      (_root_.Optim.MomentumSGD.update
        (_root_.Optim.MomentumSGD.init lr momentum params) params grads).2 :=
  _root_.Optim.Muon.init_newtonSchulz_zero_update_params_eq_momentumSGD
    coeffs lr momentum params grads

/-- Residual-check success predicate for approximate Muon backends. -/
abbrev ResidualApproxSuccess {α : Type} [Context α] {m n : Nat} (eps : α)
    (orthogonalizer : _root_.Optim.Muon.Orthogonalizer α (.dim m (.dim n .scalar)))
    (buffer : MatrixTensor α m n) : Prop :=
  _root_.Optim.Muon.ResidualApproxSuccess eps orthogonalizer buffer

/-- The residual-check success predicate is exactly the approximate Gram certificate. -/
theorem residualApproxSuccess_iff {α : Type} [Context α] {m n : Nat} (eps : α)
    (orthogonalizer : _root_.Optim.Muon.Orthogonalizer α (.dim m (.dim n .scalar)))
    (buffer : MatrixTensor α m n) :
    ResidualApproxSuccess eps orthogonalizer buffer ↔
      HasApproxColumnGram eps (orthogonalizer.apply buffer) :=
  _root_.Optim.Muon.residualApproxSuccess_iff eps orthogonalizer buffer

/-- Package any orthogonalizer as a residual-checked approximate backend. -/
abbrev residualCheckedApproxOrthogonalizer {α : Type} [Context α] {m n : Nat} (eps : α)
    (orthogonalizer : _root_.Optim.Muon.Orthogonalizer α (.dim m (.dim n .scalar))) :
    CheckedApproxOrthogonalizer α m n eps :=
  _root_.Optim.Muon.residualCheckedApproxOrthogonalizer eps orthogonalizer

/-- The success predicate of a residual-checked backend is the approximate Gram certificate. -/
theorem residualCheckedApproxOrthogonalizer_success_iff
    {α : Type} [Context α] {m n : Nat} (eps : α)
    (orthogonalizer : _root_.Optim.Muon.Orthogonalizer α (.dim m (.dim n .scalar)))
    (buffer : MatrixTensor α m n) :
    (residualCheckedApproxOrthogonalizer eps orthogonalizer).Success buffer ↔
      HasApproxColumnGram eps (orthogonalizer.apply buffer) :=
  _root_.Optim.Muon.residualCheckedApproxOrthogonalizer_success_iff eps orthogonalizer buffer

/-- Newton-Schulz packaged as a residual-checked approximate backend. -/
abbrev newtonSchulzResidualCheckedOrthogonalizer
    {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) (eps : α) :
    CheckedApproxOrthogonalizer α m n eps :=
  _root_.Optim.Muon.newtonSchulzResidualCheckedOrthogonalizer
    (α := α) (m := m) (n := n) coeffs steps eps

/--
The success predicate of the Newton-Schulz residual-checked backend is the approximate Gram
certificate for the Newton-Schulz output.
-/
theorem newtonSchulzResidualChecked_success_iff
    {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) (eps : α)
    (buffer : MatrixTensor α m n) :
    (newtonSchulzResidualCheckedOrthogonalizer
      (α := α) (m := m) (n := n) coeffs steps eps).Success buffer ↔
      HasApproxColumnGram eps
        ((newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps).apply buffer) :=
  _root_.Optim.Muon.newtonSchulzResidualChecked_success_iff coeffs steps eps buffer

/--
For a zero-step Newton-Schulz backend, the residual checker is checking the raw momentum buffer.
-/
theorem newtonSchulzResidualChecked_zero_success_iff
    {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (eps : α) (buffer : MatrixTensor α m n) :
    (newtonSchulzResidualCheckedOrthogonalizer
      (α := α) (m := m) (n := n) coeffs 0 eps).Success buffer ↔
      HasApproxColumnGram eps buffer :=
  _root_.Optim.Muon.newtonSchulzResidualChecked_zero_success_iff coeffs eps buffer

/-- Predicate saying one column-oriented Newton-Schulz step fixes a buffer. -/
abbrev NewtonSchulzFixedPoint {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (buffer : MatrixTensor α m n) : Prop :=
  _root_.Optim.Muon.NewtonSchulzFixedPoint coeffs buffer

/-- A Newton-Schulz fixed point is exactly equality with one Newton-Schulz step. -/
theorem newtonSchulzFixedPoint_iff
    {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (buffer : MatrixTensor α m n) :
    NewtonSchulzFixedPoint coeffs buffer ↔
      _root_.Optim.Muon.newtonSchulzStep coeffs buffer = buffer :=
  _root_.Optim.Muon.newtonSchulzFixedPoint_iff coeffs buffer

/-- A one-step Newton-Schulz fixed point is fixed by every finite number of iterations. -/
theorem newtonSchulzIter_fixed_of_step_fixed
    {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) (buffer : MatrixTensor α m n)
    (hfixed : NewtonSchulzFixedPoint coeffs buffer) :
    _root_.Optim.Muon.newtonSchulzIter coeffs steps buffer = buffer :=
  _root_.Optim.Muon.newtonSchulzIter_fixed_of_step_fixed coeffs steps buffer hfixed

/-- A Newton-Schulz fixed point is returned unchanged by the corresponding orthogonalizer. -/
theorem newtonSchulzOrthogonalizer_fixed_apply
    {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) (buffer : MatrixTensor α m n)
    (hfixed : NewtonSchulzFixedPoint coeffs buffer) :
    (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps).apply buffer =
      buffer :=
  _root_.Optim.Muon.newtonSchulzOrthogonalizer_fixed_apply coeffs steps buffer hfixed

/--
If a buffer already has exact column Gram and is fixed by one Newton-Schulz step, then the
finite-iteration Newton-Schulz backend exactly orthogonalizes that buffer.
-/
theorem newtonSchulzFixedPoint_exactOrthogonalizesBuffer
    {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) (buffer : MatrixTensor α m n)
    (hgram : HasExactColumnGram buffer)
    (hfixed : NewtonSchulzFixedPoint coeffs buffer) :
    ExactOrthogonalizesBuffer
      (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps) buffer :=
  _root_.Optim.Muon.newtonSchulzFixedPoint_exactOrthogonalizesBuffer
    coeffs steps buffer hgram hfixed

/-- Newton-Schulz as a checked exact backend for already-stable directions. -/
abbrev newtonSchulzFixedPointCheckedExactOrthogonalizer
    {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) :
    CheckedExactOrthogonalizer α m n :=
  _root_.Optim.Muon.newtonSchulzFixedPointCheckedExactOrthogonalizer
    (α := α) (m := m) (n := n) coeffs steps

/-- The fixed-point exact backend succeeds exactly under the explicit Gram/fixed-point predicate. -/
theorem newtonSchulzFixedPointCheckedExact_success_iff
    {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) (buffer : MatrixTensor α m n) :
    (newtonSchulzFixedPointCheckedExactOrthogonalizer
      (α := α) (m := m) (n := n) coeffs steps).Success buffer ↔
      HasExactColumnGram buffer ∧ NewtonSchulzFixedPoint coeffs buffer :=
  _root_.Optim.Muon.newtonSchulzFixedPointCheckedExact_success_iff coeffs steps buffer

/--
If `QᵀQ = I`, then one real column-oriented Newton-Schulz step returns
`(a + b + c) Q`.
-/
theorem newtonSchulzStep_eq_scale_sum_of_exact_column_gram
    {m n : Nat}
    (coeffs : NewtonSchulzCoeffs ℝ) (Q : MatrixTensor ℝ m n)
    (hgram : HasExactColumnGram Q) :
    _root_.Optim.Muon.newtonSchulzStep coeffs Q =
      _root_.Spec.Tensor.scaleSpec Q (coeffs.a + coeffs.b + coeffs.c) :=
  _root_.Optim.Muon.newtonSchulzStep_eq_scale_sum_of_exact_column_gram
    coeffs Q hgram

/-- Scaling an exact-column-orthogonal real matrix by `k` preserves exact Gram when `k * k = 1`. -/
theorem scale_hasExactColumnGram_of_square_eq_one
    {m n : Nat} (Q : MatrixTensor ℝ m n) (k : ℝ)
    (hgram : HasExactColumnGram Q) (hk : k * k = 1) :
    HasExactColumnGram (_root_.Spec.Tensor.scaleSpec Q k) :=
  _root_.Optim.Muon.scale_hasExactColumnGram_of_square_eq_one Q k hgram hk

/--
Scaling an exact-column-orthogonal real matrix gives an approximate Gram certificate when
`|k^2 - 1|` is bounded by `eps`.
-/
theorem scale_hasApproxColumnGram_of_exact_column_gram_of_square_error
    {m n : Nat} (Q : MatrixTensor ℝ m n) (k eps : ℝ)
    (hgram : HasExactColumnGram Q)
    (herr : MathFunctions.abs (k * k - 1) ≤ eps)
    (heps : 0 ≤ eps) :
    HasApproxColumnGram eps (_root_.Spec.Tensor.scaleSpec Q k) :=
  _root_.Optim.Muon.scale_hasApproxColumnGram_of_exact_column_gram_of_square_error
    Q k eps hgram herr heps

/--
If `QᵀQ = I` and `(a + b + c)^2 = 1`, then one real Newton-Schulz step still has exact column
Gram.
-/
theorem newtonSchulzStep_hasExactColumnGram_of_exact_column_gram_of_sum_square_one
    {m n : Nat}
    (coeffs : NewtonSchulzCoeffs ℝ) (Q : MatrixTensor ℝ m n)
    (hgram : HasExactColumnGram Q)
    (hsquare : (coeffs.a + coeffs.b + coeffs.c) * (coeffs.a + coeffs.b + coeffs.c) = 1) :
    HasExactColumnGram (_root_.Optim.Muon.newtonSchulzStep coeffs Q) :=
  _root_.Optim.Muon.newtonSchulzStep_hasExactColumnGram_of_exact_column_gram_of_sum_square_one
    coeffs Q hgram hsquare

/--
If `QᵀQ = I` and `|(a + b + c)^2 - 1| ≤ eps`, then one real Newton-Schulz step has entrywise Gram
residual bounded by `eps`.
-/
theorem newtonSchulzStep_hasApproxColumnGram_of_exact_column_gram_of_sum_square_error
    {m n : Nat}
    (coeffs : NewtonSchulzCoeffs ℝ) (Q : MatrixTensor ℝ m n) (eps : ℝ)
    (hgram : HasExactColumnGram Q)
    (herr :
      MathFunctions.abs
        ((coeffs.a + coeffs.b + coeffs.c) * (coeffs.a + coeffs.b + coeffs.c) - 1) ≤ eps)
    (heps : 0 ≤ eps) :
    HasApproxColumnGram eps (_root_.Optim.Muon.newtonSchulzStep coeffs Q) :=
  _root_.Optim.Muon.newtonSchulzStep_hasApproxColumnGram_of_exact_column_gram_of_sum_square_error
    coeffs Q eps hgram herr heps

/--
For real coefficients whose sum is one, an exact-column-orthogonal matrix is a fixed point of one
column-oriented Newton-Schulz step.
-/
theorem newtonSchulzFixedPoint_of_exact_column_gram_of_coeff_sum_one
    {m n : Nat}
    (coeffs : NewtonSchulzCoeffs ℝ) (Q : MatrixTensor ℝ m n)
    (hgram : HasExactColumnGram Q)
    (hsum : coeffs.a + coeffs.b + coeffs.c = 1) :
    NewtonSchulzFixedPoint coeffs Q :=
  _root_.Optim.Muon.newtonSchulzFixedPoint_of_exact_column_gram_of_coeff_sum_one
    coeffs Q hgram hsum

/--
For real coefficients whose sum is one, exact column Gram is enough to satisfy the fixed-point
checked backend's success predicate.
-/
theorem newtonSchulzFixedPointCheckedExact_success_of_coeff_sum_one
    {m n : Nat}
    (coeffs : NewtonSchulzCoeffs ℝ) (steps : Nat) (buffer : MatrixTensor ℝ m n)
    (hgram : HasExactColumnGram buffer)
    (hsum : coeffs.a + coeffs.b + coeffs.c = 1) :
    (newtonSchulzFixedPointCheckedExactOrthogonalizer
      (α := ℝ) (m := m) (n := n) coeffs steps).Success buffer :=
  _root_.Optim.Muon.newtonSchulzFixedPointCheckedExact_success_of_coeff_sum_one
    coeffs steps buffer hgram hsum

/-- Exact certificate carried by the direction used in a Muon step. -/
abbrev ExactCertifiedDirection {α : Type} [Context α] {m n : Nat}
    (orthogonalizer : _root_.Optim.Muon.Orthogonalizer α (.dim m (.dim n .scalar)))
    (buffer direction : MatrixTensor α m n) : Prop :=
  _root_.Optim.Muon.ExactCertifiedDirection orthogonalizer buffer direction

/-- Approximate certificate carried by the direction used in a Muon step. -/
abbrev ApproxCertifiedDirection {α : Type} [Context α] {m n : Nat} (eps : α)
    (orthogonalizer : _root_.Optim.Muon.Orthogonalizer α (.dim m (.dim n .scalar)))
    (buffer direction : MatrixTensor α m n) : Prop :=
  _root_.Optim.Muon.ApproxCertifiedDirection eps orthogonalizer buffer direction

/--
Exact certified Muon step: certified direction, next-state equation, and parameter-update equation.
-/
abbrev ExactCertifiedStep {α : Type} [Context α] {m n : Nat}
    (state : _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
    (params grads direction : MatrixTensor α m n) : Prop :=
  _root_.Optim.Muon.ExactCertifiedStep state params grads direction

/--
Approximate certified Muon step: residual-certified direction, next-state equation, and
parameter-update equation.
-/
abbrev ApproxCertifiedStep {α : Type} [Context α] {m n : Nat} (eps : α)
    (state : _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
    (params grads direction : MatrixTensor α m n) : Prop :=
  _root_.Optim.Muon.ApproxCertifiedStep eps state params grads direction

/-- An exact certified step gives the exact column-Gram fact for its certified direction. -/
theorem exactCertifiedStep_direction_has_exact_column_gram
    {α : Type} [Context α] {m n : Nat}
    {state : _root_.Optim.Muon.State α (.dim m (.dim n .scalar))}
    {params grads direction : MatrixTensor α m n}
    (hstep : ExactCertifiedStep state params grads direction) :
    HasExactColumnGram direction :=
  _root_.Optim.Muon.exactCertifiedStep_direction_has_exact_column_gram hstep

/-- An approximate certified step gives the residual bound for its certified direction. -/
theorem approxCertifiedStep_direction_has_approx_column_gram
    {α : Type} [Context α] {m n : Nat} {eps : α}
    {state : _root_.Optim.Muon.State α (.dim m (.dim n .scalar))}
    {params grads direction : MatrixTensor α m n}
    (hstep : ApproxCertifiedStep eps state params grads direction) :
    HasApproxColumnGram eps direction :=
  _root_.Optim.Muon.approxCertifiedStep_direction_has_approx_column_gram hstep

/--
An exact certified step identifies its certified direction with the backend output on the fresh
momentum buffer.
-/
theorem exactCertifiedStep_direction_eq
    {α : Type} [Context α] {m n : Nat}
    {state : _root_.Optim.Muon.State α (.dim m (.dim n .scalar))}
    {params grads direction : MatrixTensor α m n}
    (hstep : ExactCertifiedStep state params grads direction) :
    direction = state.orthogonalizer.apply (_root_.Optim.Muon.update state params grads).1.buf :=
  _root_.Optim.Muon.exactCertifiedStep_direction_eq hstep

/--
An approximate certified step identifies its certified direction with the backend output on the
fresh momentum buffer.
-/
theorem approxCertifiedStep_direction_eq
    {α : Type} [Context α] {m n : Nat} {eps : α}
    {state : _root_.Optim.Muon.State α (.dim m (.dim n .scalar))}
    {params grads direction : MatrixTensor α m n}
    (hstep : ApproxCertifiedStep eps state params grads direction) :
    direction = state.orthogonalizer.apply (_root_.Optim.Muon.update state params grads).1.buf :=
  _root_.Optim.Muon.approxCertifiedStep_direction_eq hstep

/--
An exact certified step gives the exact column-Gram fact for the actual backend output on the fresh
momentum buffer.
-/
theorem exactCertifiedStep_backend_output_has_exact_column_gram
    {α : Type} [Context α] {m n : Nat}
    {state : _root_.Optim.Muon.State α (.dim m (.dim n .scalar))}
    {params grads direction : MatrixTensor α m n}
    (hstep : ExactCertifiedStep state params grads direction) :
    HasExactColumnGram (state.orthogonalizer.apply (_root_.Optim.Muon.update state params grads).1.buf) :=
  _root_.Optim.Muon.exactCertifiedStep_backend_output_has_exact_column_gram hstep

/--
An approximate certified step gives the residual bound for the actual backend output on the fresh
momentum buffer.
-/
theorem approxCertifiedStep_backend_output_has_approx_column_gram
    {α : Type} [Context α] {m n : Nat} {eps : α}
    {state : _root_.Optim.Muon.State α (.dim m (.dim n .scalar))}
    {params grads direction : MatrixTensor α m n}
    (hstep : ApproxCertifiedStep eps state params grads direction) :
    HasApproxColumnGram eps
      (state.orthogonalizer.apply (_root_.Optim.Muon.update state params grads).1.buf) :=
  _root_.Optim.Muon.approxCertifiedStep_backend_output_has_approx_column_gram hstep

/-- An exact certified step gives the next-state equation for the Muon update. -/
theorem exactCertifiedStep_state_eq
    {α : Type} [Context α] {m n : Nat}
    {state : _root_.Optim.Muon.State α (.dim m (.dim n .scalar))}
    {params grads direction : MatrixTensor α m n}
    (hstep : ExactCertifiedStep state params grads direction) :
    (_root_.Optim.Muon.update state params grads).1 =
      { state with
        buf := _root_.Optim.OptimizerUtils.updateMomentumBuf state.buf state.momentum grads } :=
  _root_.Optim.Muon.exactCertifiedStep_state_eq hstep

/-- An approximate certified step gives the next-state equation for the Muon update. -/
theorem approxCertifiedStep_state_eq
    {α : Type} [Context α] {m n : Nat} {eps : α}
    {state : _root_.Optim.Muon.State α (.dim m (.dim n .scalar))}
    {params grads direction : MatrixTensor α m n}
    (hstep : ApproxCertifiedStep eps state params grads direction) :
    (_root_.Optim.Muon.update state params grads).1 =
      { state with
        buf := _root_.Optim.OptimizerUtils.updateMomentumBuf state.buf state.momentum grads } :=
  _root_.Optim.Muon.approxCertifiedStep_state_eq hstep

/-- An exact certified step gives the parameter-update equation for the certified direction. -/
theorem exactCertifiedStep_params_eq
    {α : Type} [Context α] {m n : Nat}
    {state : _root_.Optim.Muon.State α (.dim m (.dim n .scalar))}
    {params grads direction : MatrixTensor α m n}
    (hstep : ExactCertifiedStep state params grads direction) :
    (_root_.Optim.Muon.update state params grads).2 =
      _root_.Spec.Tensor.subSpec params (_root_.Spec.Tensor.scaleSpec direction state.lr) :=
  _root_.Optim.Muon.exactCertifiedStep_params_eq hstep

/-- An approximate certified step gives the parameter-update equation for the certified direction. -/
theorem approxCertifiedStep_params_eq
    {α : Type} [Context α] {m n : Nat} {eps : α}
    {state : _root_.Optim.Muon.State α (.dim m (.dim n .scalar))}
    {params grads direction : MatrixTensor α m n}
    (hstep : ApproxCertifiedStep eps state params grads direction) :
    (_root_.Optim.Muon.update state params grads).2 =
      _root_.Spec.Tensor.subSpec params (_root_.Spec.Tensor.scaleSpec direction state.lr) :=
  _root_.Optim.Muon.approxCertifiedStep_params_eq hstep

/-- A residual proof for the Newton-Schulz output gives the approximate direction certificate. -/
theorem newtonSchulz_residual_certifies_direction
    {α : Type} [Context α] {m n : Nat} {eps : α}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat) (buffer : MatrixTensor α m n)
    (hresidual :
      ResidualApproxSuccess eps
        (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
        buffer) :
    ApproxCertifiedDirection eps
      (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
      buffer
      ((newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps).apply buffer) :=
  _root_.Optim.Muon.newtonSchulz_residual_certifies_direction coeffs steps buffer hresidual

/-- An exact orthogonalizer certifies its own output direction. -/
theorem exact_certified_direction_of_orthogonalizer
    {α : Type} [Context α] {m n : Nat}
    {orthogonalizer : _root_.Optim.Muon.Orthogonalizer α (.dim m (.dim n .scalar))}
    (horth : ExactMatrixOrthogonalizer orthogonalizer)
    (buffer : MatrixTensor α m n) :
    ExactCertifiedDirection orthogonalizer buffer (orthogonalizer.apply buffer) :=
  _root_.Optim.Muon.exact_certified_direction_of_orthogonalizer horth buffer

/-- An approximate orthogonalizer certifies its own output direction. -/
theorem approx_certified_direction_of_orthogonalizer
    {α : Type} [Context α] {m n : Nat} {eps : α}
    {orthogonalizer : _root_.Optim.Muon.Orthogonalizer α (.dim m (.dim n .scalar))}
    (horth : ApproxMatrixOrthogonalizer eps orthogonalizer)
    (buffer : MatrixTensor α m n) :
    ApproxCertifiedDirection eps orthogonalizer buffer (orthogonalizer.apply buffer) :=
  _root_.Optim.Muon.approx_certified_direction_of_orthogonalizer horth buffer

/-- Proof-level QR/Gram-Schmidt orthogonalizer over `ℝ`. -/
noncomputable abbrev qrOrthogonalizer {m n : Nat} :
    _root_.Optim.Muon.Orthogonalizer ℝ (.dim m (.dim n .scalar)) :=
  _root_.Optim.Muon.qrOrthogonalizer (m := m) (n := n)

/-- Positive executable QR pivots, the success condition for the exact QR-backed Muon theorem. -/
abbrev HasPositiveQRPivots {m n : Nat} (buffer : MatrixTensor ℝ m n) : Prop :=
  _root_.Optim.Muon.HasPositiveQRPivots buffer

/-- QR packaged as a checked exact Muon backend. -/
noncomputable abbrev qrCheckedExactOrthogonalizer {m n : Nat} :
    CheckedExactOrthogonalizer ℝ m n :=
  _root_.Optim.Muon.qrCheckedExactOrthogonalizer (m := m) (n := n)

/-- The QR checked backend succeeds exactly when the executable QR pivots are positive. -/
theorem qrCheckedExactOrthogonalizer_success_iff {m n : Nat}
    (buffer : MatrixTensor ℝ m n) :
    (qrCheckedExactOrthogonalizer (m := m) (n := n)).Success buffer ↔
      HasPositiveQRPivots buffer :=
  _root_.Optim.Muon.qrCheckedExactOrthogonalizer_success_iff buffer

/-- QR orthogonalization exactly satisfies the Muon buffer contract under positive QR pivots. -/
theorem qrOrthogonalizer_exact_of_positive_pivots {m n : Nat}
    (buffer : MatrixTensor ℝ m n)
    (hpivots : HasPositiveQRPivots buffer) :
    _root_.Optim.Muon.ExactOrthogonalizesBuffer
      (qrOrthogonalizer (m := m) (n := n)) buffer :=
  _root_.Optim.Muon.qrOrthogonalizer_exact_of_positive_pivots buffer hpivots

/--
Generic exact-backend Muon theorem: if the state's backend is exact for every matrix buffer, the
update uses an exactly certified direction.
-/
theorem update_has_exact_certified_direction
    {α : Type} [Context α] {m n : Nat}
    (state : _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
    (params grads : MatrixTensor α m n)
    (horth : ExactMatrixOrthogonalizer state.orthogonalizer) :
    ∃ direction : MatrixTensor α m n,
      ExactCertifiedDirection state.orthogonalizer
        (_root_.Optim.Muon.update state params grads).1.buf direction ∧
      (_root_.Optim.Muon.update state params grads).2 =
        _root_.Spec.Tensor.subSpec params (_root_.Spec.Tensor.scaleSpec direction state.lr) :=
  _root_.Optim.Muon.update_has_exact_certified_direction state params grads horth

/--
Generic exact-backend Muon theorem for packaged backends.
-/
theorem update_has_exact_certified_direction_of_backend
    {α : Type} [Context α] {m n : Nat}
    (backend : ExactCertifiedOrthogonalizer α m n)
    (lr momentum : α) (buf params grads : MatrixTensor α m n) :
    ∃ direction : MatrixTensor α m n,
      ExactCertifiedDirection backend.orthogonalizer
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).1.buf direction ∧
      (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).2 =
        _root_.Spec.Tensor.subSpec params (_root_.Spec.Tensor.scaleSpec direction lr) :=
  _root_.Optim.Muon.update_has_exact_certified_direction_of_backend
    backend lr momentum buf params grads

/--
Generic exact-backend Muon theorem: if the state's backend is exact for every matrix buffer, the
update has a certified step containing the direction certificate, next-state equation, and
parameter-update equation.
-/
theorem update_has_exact_certified_step
    {α : Type} [Context α] {m n : Nat}
    (state : _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
    (params grads : MatrixTensor α m n)
    (horth : ExactMatrixOrthogonalizer state.orthogonalizer) :
    ∃ direction : MatrixTensor α m n,
      ExactCertifiedStep state params grads direction :=
  _root_.Optim.Muon.update_has_exact_certified_step state params grads horth

/--
Generic exact-backend Muon theorem for packaged backends, returning the whole certified step.
-/
theorem update_has_exact_certified_step_of_backend
    {α : Type} [Context α] {m n : Nat}
    (backend : ExactCertifiedOrthogonalizer α m n)
    (lr momentum : α) (buf params grads : MatrixTensor α m n) :
    ∃ direction : MatrixTensor α m n,
      ExactCertifiedStep
        ({ lr := lr, momentum := momentum, buf := buf,
           orthogonalizer := backend.orthogonalizer } :
          _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
        params grads direction :=
  _root_.Optim.Muon.update_has_exact_certified_step_of_backend
    backend lr momentum buf params grads

/--
Local exact-backend Muon theorem: it is enough to certify the fresh momentum buffer used by this
one update.
-/
theorem update_has_exact_certified_direction_of_buffer
    {α : Type} [Context α] {m n : Nat}
    (state : _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
    (params grads : MatrixTensor α m n)
    (horth :
      ExactOrthogonalizesBuffer state.orthogonalizer
        (_root_.Optim.Muon.update state params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ExactCertifiedDirection state.orthogonalizer
        (_root_.Optim.Muon.update state params grads).1.buf direction ∧
      (_root_.Optim.Muon.update state params grads).2 =
        _root_.Spec.Tensor.subSpec params (_root_.Spec.Tensor.scaleSpec direction state.lr) :=
  _root_.Optim.Muon.update_has_exact_certified_direction_of_buffer state params grads horth

/--
Local exact-backend Muon theorem returning the whole certified step from a certificate for the
fresh momentum buffer used by this one update.
-/
theorem update_has_exact_certified_step_of_buffer
    {α : Type} [Context α] {m n : Nat}
    (state : _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
    (params grads : MatrixTensor α m n)
    (horth :
      ExactOrthogonalizesBuffer state.orthogonalizer
        (_root_.Optim.Muon.update state params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ExactCertifiedStep state params grads direction :=
  _root_.Optim.Muon.update_has_exact_certified_step_of_buffer state params grads horth

/--
Generic checked-backend theorem for exact Muon: once the backend success predicate is established
on the fresh momentum buffer, the update uses an exactly certified direction.
-/
theorem update_has_exact_certified_direction_of_checked_backend
    {α : Type} [Context α] {m n : Nat}
    (backend : CheckedExactOrthogonalizer α m n)
    (lr momentum : α) (buf params grads : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ExactCertifiedDirection backend.orthogonalizer
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).1.buf direction ∧
      (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).2 =
        _root_.Spec.Tensor.subSpec params (_root_.Spec.Tensor.scaleSpec direction lr) :=
  _root_.Optim.Muon.update_has_exact_certified_direction_of_checked_backend
    backend lr momentum buf params grads hsuccess

/--
Generic checked-backend theorem for exact Muon: backend success gives a certified step containing
the direction certificate, next-state equation, and parameter-update equation.
-/
theorem update_has_exact_certified_step_of_checked_backend
    {α : Type} [Context α] {m n : Nat}
    (backend : CheckedExactOrthogonalizer α m n)
    (lr momentum : α) (buf params grads : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ExactCertifiedStep
        ({ lr := lr, momentum := momentum, buf := buf,
           orthogonalizer := backend.orthogonalizer } :
          _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
        params grads direction :=
  _root_.Optim.Muon.update_has_exact_certified_step_of_checked_backend
    backend lr momentum buf params grads hsuccess

/--
Generic checked-backend theorem for exact Muon: backend success gives `QᵀQ = I` for the actual
direction used by this update.
-/
theorem update_direction_has_exact_column_gram_of_checked_backend
    {α : Type} [Context α] {m n : Nat}
    (backend : CheckedExactOrthogonalizer α m n)
    (lr momentum : α) (buf params grads : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    HasExactColumnGram
      (backend.orthogonalizer.apply
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :=
  _root_.Optim.Muon.update_direction_has_exact_column_gram_of_checked_backend
    backend lr momentum buf params grads hsuccess

/--
Initialized checked-backend theorem for exact Muon: after `init`, backend success on the fresh
momentum buffer certifies the direction used by the first update.
-/
theorem init_has_exact_certified_direction_of_checked_backend
    {α : Type} [Context α] {m n : Nat}
    (backend : CheckedExactOrthogonalizer α m n)
    (lr momentum : α) (params grads : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum backend.orthogonalizer params)
          params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ExactCertifiedDirection backend.orthogonalizer
        (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum backend.orthogonalizer params)
          params grads).1.buf direction ∧
      (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum backend.orthogonalizer params)
          params grads).2 =
        _root_.Spec.Tensor.subSpec params (_root_.Spec.Tensor.scaleSpec direction lr) :=
  _root_.Optim.Muon.init_has_exact_certified_direction_of_checked_backend
    backend lr momentum params grads hsuccess

/--
Initialized checked-backend theorem for exact Muon: after `init`, backend success gives a certified
first step.
-/
theorem init_has_exact_certified_step_of_checked_backend
    {α : Type} [Context α] {m n : Nat}
    (backend : CheckedExactOrthogonalizer α m n)
    (lr momentum : α) (params grads : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum backend.orthogonalizer params)
          params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ExactCertifiedStep
        (_root_.Optim.Muon.init lr momentum backend.orthogonalizer params)
        params grads direction :=
  _root_.Optim.Muon.init_has_exact_certified_step_of_checked_backend
    backend lr momentum params grads hsuccess

/--
Initialized checked-backend theorem for exact Muon: after `init`, backend success gives `QᵀQ = I`
for the direction used by the first update.
-/
theorem init_direction_has_exact_column_gram_of_checked_backend
    {α : Type} [Context α] {m n : Nat}
    (backend : CheckedExactOrthogonalizer α m n)
    (lr momentum : α) (params grads : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum backend.orthogonalizer params)
          params grads).1.buf) :
    HasExactColumnGram
      (backend.orthogonalizer.apply
        (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum backend.orthogonalizer params)
          params grads).1.buf) :=
  _root_.Optim.Muon.init_direction_has_exact_column_gram_of_checked_backend
    backend lr momentum params grads hsuccess

/--
Fixed-point checked Newton-Schulz theorem: backend success gives an exact certified direction for
one Muon update.
-/
theorem update_has_exact_certified_direction_newtonSchulz_fixed_checked
    {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat)
    (lr momentum : α) (buf params grads : MatrixTensor α m n)
    (hsuccess :
      (newtonSchulzFixedPointCheckedExactOrthogonalizer
        (α := α) (m := m) (n := n) coeffs steps).Success
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ExactCertifiedDirection
        (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).1.buf direction ∧
      (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).2 =
        _root_.Spec.Tensor.subSpec params (_root_.Spec.Tensor.scaleSpec direction lr) :=
  _root_.Optim.Muon.update_has_exact_certified_direction_newtonSchulz_fixed_checked
    coeffs steps lr momentum buf params grads hsuccess

/--
Fixed-point checked Newton-Schulz theorem: backend success gives an exact certified Muon step.
-/
theorem update_has_exact_certified_step_newtonSchulz_fixed_checked
    {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat)
    (lr momentum : α) (buf params grads : MatrixTensor α m n)
    (hsuccess :
      (newtonSchulzFixedPointCheckedExactOrthogonalizer
        (α := α) (m := m) (n := n) coeffs steps).Success
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ExactCertifiedStep
        ({ lr := lr, momentum := momentum, buf := buf,
           orthogonalizer :=
            newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
          _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
        params grads direction :=
  _root_.Optim.Muon.update_has_exact_certified_step_newtonSchulz_fixed_checked
    coeffs steps lr momentum buf params grads hsuccess

/-- Fixed-point checked Newton-Schulz gives `QᵀQ = I` for the actual update direction. -/
theorem update_newtonSchulz_fixed_direction_has_exact_column_gram_checked
    {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat)
    (lr momentum : α) (buf params grads : MatrixTensor α m n)
    (hsuccess :
      (newtonSchulzFixedPointCheckedExactOrthogonalizer
        (α := α) (m := m) (n := n) coeffs steps).Success
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    HasExactColumnGram
      ((newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps).apply
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :=
  _root_.Optim.Muon.update_newtonSchulz_fixed_direction_has_exact_column_gram_checked
    coeffs steps lr momentum buf params grads hsuccess

/-- Initialized fixed-point checked Newton-Schulz gives an exact certified first Muon step. -/
theorem init_has_exact_certified_step_newtonSchulz_fixed_checked
    {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat)
    (lr momentum : α) (params grads : MatrixTensor α m n)
    (hsuccess :
      (newtonSchulzFixedPointCheckedExactOrthogonalizer
        (α := α) (m := m) (n := n) coeffs steps).Success
        (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum
            (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
            params)
          params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ExactCertifiedStep
        (_root_.Optim.Muon.init lr momentum
          (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
          params)
        params grads direction :=
  _root_.Optim.Muon.init_has_exact_certified_step_newtonSchulz_fixed_checked
    coeffs steps lr momentum params grads hsuccess

/-- Initialized fixed-point checked Newton-Schulz gives `QᵀQ = I` for the first update direction. -/
theorem init_newtonSchulz_fixed_direction_has_exact_column_gram_checked
    {α : Type} [Context α] {m n : Nat}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat)
    (lr momentum : α) (params grads : MatrixTensor α m n)
    (hsuccess :
      (newtonSchulzFixedPointCheckedExactOrthogonalizer
        (α := α) (m := m) (n := n) coeffs steps).Success
        (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum
            (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
            params)
          params grads).1.buf) :
    HasExactColumnGram
      ((newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps).apply
        (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum
            (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
            params)
          params grads).1.buf) :=
  _root_.Optim.Muon.init_newtonSchulz_fixed_direction_has_exact_column_gram_checked
    coeffs steps lr momentum params grads hsuccess

/--
For real coefficients with `a + b + c = 1`, exact column Gram of the fresh momentum buffer is
enough to certify a Newton-Schulz Muon update exactly.
-/
theorem update_has_exact_certified_step_newtonSchulz_exact_gram_checked
    {m n : Nat}
    (coeffs : NewtonSchulzCoeffs ℝ) (steps : Nat)
    (lr momentum : ℝ) (buf params grads : MatrixTensor ℝ m n)
    (hgram :
      HasExactColumnGram
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := ℝ) (m := m) (n := n) coeffs steps } :
            _root_.Optim.Muon.State ℝ (.dim m (.dim n .scalar)))
          params grads).1.buf)
    (hsum : coeffs.a + coeffs.b + coeffs.c = 1) :
    ∃ direction : MatrixTensor ℝ m n,
      ExactCertifiedStep
        ({ lr := lr, momentum := momentum, buf := buf,
           orthogonalizer :=
            newtonSchulzOrthogonalizer (α := ℝ) (m := m) (n := n) coeffs steps } :
          _root_.Optim.Muon.State ℝ (.dim m (.dim n .scalar)))
        params grads direction :=
  _root_.Optim.Muon.update_has_exact_certified_step_newtonSchulz_exact_gram_checked
    coeffs steps lr momentum buf params grads hgram hsum

/--
For real coefficients with `a + b + c = 1`, exact column Gram of the fresh momentum buffer gives
`QᵀQ = I` for the actual Newton-Schulz update direction.
-/
theorem update_newtonSchulz_exact_gram_direction_has_exact_column_gram_checked
    {m n : Nat}
    (coeffs : NewtonSchulzCoeffs ℝ) (steps : Nat)
    (lr momentum : ℝ) (buf params grads : MatrixTensor ℝ m n)
    (hgram :
      HasExactColumnGram
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := ℝ) (m := m) (n := n) coeffs steps } :
            _root_.Optim.Muon.State ℝ (.dim m (.dim n .scalar)))
          params grads).1.buf)
    (hsum : coeffs.a + coeffs.b + coeffs.c = 1) :
    HasExactColumnGram
      ((newtonSchulzOrthogonalizer (α := ℝ) (m := m) (n := n) coeffs steps).apply
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := ℝ) (m := m) (n := n) coeffs steps } :
            _root_.Optim.Muon.State ℝ (.dim m (.dim n .scalar)))
          params grads).1.buf) :=
  _root_.Optim.Muon.update_newtonSchulz_exact_gram_direction_has_exact_column_gram_checked
    coeffs steps lr momentum buf params grads hgram hsum

/--
Initialized version: exact column Gram of the first fresh momentum buffer and `a + b + c = 1`
certify the first Newton-Schulz Muon step exactly.
-/
theorem init_has_exact_certified_step_newtonSchulz_exact_gram_checked
    {m n : Nat}
    (coeffs : NewtonSchulzCoeffs ℝ) (steps : Nat)
    (lr momentum : ℝ) (params grads : MatrixTensor ℝ m n)
    (hgram :
      HasExactColumnGram
        (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum
            (newtonSchulzOrthogonalizer (α := ℝ) (m := m) (n := n) coeffs steps)
            params)
          params grads).1.buf)
    (hsum : coeffs.a + coeffs.b + coeffs.c = 1) :
    ∃ direction : MatrixTensor ℝ m n,
      ExactCertifiedStep
        (_root_.Optim.Muon.init lr momentum
          (newtonSchulzOrthogonalizer (α := ℝ) (m := m) (n := n) coeffs steps)
          params)
        params grads direction :=
  _root_.Optim.Muon.init_has_exact_certified_step_newtonSchulz_exact_gram_checked
    coeffs steps lr momentum params grads hgram hsum

/--
Generic approximate-backend Muon theorem: if the state's backend has an entrywise Gram-residual
bound for every matrix buffer, the update uses an approximately certified direction.
-/
theorem update_has_approx_certified_direction
    {α : Type} [Context α] {m n : Nat} {eps : α}
    (state : _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
    (params grads : MatrixTensor α m n)
    (horth : ApproxMatrixOrthogonalizer eps state.orthogonalizer) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedDirection eps state.orthogonalizer
        (_root_.Optim.Muon.update state params grads).1.buf direction ∧
      (_root_.Optim.Muon.update state params grads).2 =
        _root_.Spec.Tensor.subSpec params (_root_.Spec.Tensor.scaleSpec direction state.lr) :=
  _root_.Optim.Muon.update_has_approx_certified_direction state params grads horth

/--
Generic approximate-backend Muon theorem for packaged backends.
-/
theorem update_has_approx_certified_direction_of_backend
    {α : Type} [Context α] {m n : Nat} {eps : α}
    (backend : ApproxCertifiedOrthogonalizer α m n eps)
    (lr momentum : α) (buf params grads : MatrixTensor α m n) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedDirection eps backend.orthogonalizer
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).1.buf direction ∧
      (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).2 =
        _root_.Spec.Tensor.subSpec params (_root_.Spec.Tensor.scaleSpec direction lr) :=
  _root_.Optim.Muon.update_has_approx_certified_direction_of_backend
    backend lr momentum buf params grads

/--
Generic approximate-backend Muon theorem: if the state's backend is approximately orthogonalizing
for every matrix buffer, the update has a certified step containing the residual-bounded direction,
next-state equation, and parameter-update equation.
-/
theorem update_has_approx_certified_step
    {α : Type} [Context α] {m n : Nat} {eps : α}
    (state : _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
    (params grads : MatrixTensor α m n)
    (horth : ApproxMatrixOrthogonalizer eps state.orthogonalizer) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedStep eps state params grads direction :=
  _root_.Optim.Muon.update_has_approx_certified_step state params grads horth

/--
Generic approximate-backend Muon theorem for packaged backends, returning the whole certified step.
-/
theorem update_has_approx_certified_step_of_backend
    {α : Type} [Context α] {m n : Nat} {eps : α}
    (backend : ApproxCertifiedOrthogonalizer α m n eps)
    (lr momentum : α) (buf params grads : MatrixTensor α m n) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedStep eps
        ({ lr := lr, momentum := momentum, buf := buf,
           orthogonalizer := backend.orthogonalizer } :
          _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
        params grads direction :=
  _root_.Optim.Muon.update_has_approx_certified_step_of_backend
    backend lr momentum buf params grads

/--
Local approximate-backend Muon theorem: it is enough to certify the fresh momentum buffer used by
this one update.
-/
theorem update_has_approx_certified_direction_of_buffer
    {α : Type} [Context α] {m n : Nat} {eps : α}
    (state : _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
    (params grads : MatrixTensor α m n)
    (horth :
      ApproxOrthogonalizesBuffer eps state.orthogonalizer
        (_root_.Optim.Muon.update state params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedDirection eps state.orthogonalizer
        (_root_.Optim.Muon.update state params grads).1.buf direction ∧
      (_root_.Optim.Muon.update state params grads).2 =
        _root_.Spec.Tensor.subSpec params (_root_.Spec.Tensor.scaleSpec direction state.lr) :=
  _root_.Optim.Muon.update_has_approx_certified_direction_of_buffer state params grads horth

/--
Local approximate-backend Muon theorem returning the whole certified step from a residual
certificate for the fresh momentum buffer used by this one update.
-/
theorem update_has_approx_certified_step_of_buffer
    {α : Type} [Context α] {m n : Nat} {eps : α}
    (state : _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
    (params grads : MatrixTensor α m n)
    (horth :
      ApproxOrthogonalizesBuffer eps state.orthogonalizer
        (_root_.Optim.Muon.update state params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedStep eps state params grads direction :=
  _root_.Optim.Muon.update_has_approx_certified_step_of_buffer state params grads horth

/--
Generic checked-backend theorem for approximate Muon: once the backend success predicate is
established on the fresh momentum buffer, the update uses an approximately certified direction.
-/
theorem update_has_approx_certified_direction_of_checked_backend
    {α : Type} [Context α] {m n : Nat} {eps : α}
    (backend : CheckedApproxOrthogonalizer α m n eps)
    (lr momentum : α) (buf params grads : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedDirection eps backend.orthogonalizer
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).1.buf direction ∧
      (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).2 =
        _root_.Spec.Tensor.subSpec params (_root_.Spec.Tensor.scaleSpec direction lr) :=
  _root_.Optim.Muon.update_has_approx_certified_direction_of_checked_backend
    backend lr momentum buf params grads hsuccess

/--
Generic checked-backend theorem for approximate Muon: backend success gives a certified step
containing the residual-bounded direction, next-state equation, and parameter-update equation.
-/
theorem update_has_approx_certified_step_of_checked_backend
    {α : Type} [Context α] {m n : Nat} {eps : α}
    (backend : CheckedApproxOrthogonalizer α m n eps)
    (lr momentum : α) (buf params grads : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedStep eps
        ({ lr := lr, momentum := momentum, buf := buf,
           orthogonalizer := backend.orthogonalizer } :
          _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
        params grads direction :=
  _root_.Optim.Muon.update_has_approx_certified_step_of_checked_backend
    backend lr momentum buf params grads hsuccess

/--
Generic checked-backend theorem for approximate Muon: backend success gives the residual bound for
the actual direction used by this update.
-/
theorem update_direction_has_approx_column_gram_of_checked_backend
    {α : Type} [Context α] {m n : Nat} {eps : α}
    (backend : CheckedApproxOrthogonalizer α m n eps)
    (lr momentum : α) (buf params grads : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    HasApproxColumnGram eps
      (backend.orthogonalizer.apply
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := backend.orthogonalizer } :
            _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :=
  _root_.Optim.Muon.update_direction_has_approx_column_gram_of_checked_backend
    backend lr momentum buf params grads hsuccess

/--
Initialized checked-backend theorem for approximate Muon: after `init`, backend success on the
fresh momentum buffer certifies the direction used by the first update.
-/
theorem init_has_approx_certified_direction_of_checked_backend
    {α : Type} [Context α] {m n : Nat} {eps : α}
    (backend : CheckedApproxOrthogonalizer α m n eps)
    (lr momentum : α) (params grads : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum backend.orthogonalizer params)
          params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedDirection eps backend.orthogonalizer
        (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum backend.orthogonalizer params)
          params grads).1.buf direction ∧
      (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum backend.orthogonalizer params)
          params grads).2 =
        _root_.Spec.Tensor.subSpec params (_root_.Spec.Tensor.scaleSpec direction lr) :=
  _root_.Optim.Muon.init_has_approx_certified_direction_of_checked_backend
    backend lr momentum params grads hsuccess

/--
Initialized checked-backend theorem for approximate Muon: after `init`, backend success gives a
certified first step.
-/
theorem init_has_approx_certified_step_of_checked_backend
    {α : Type} [Context α] {m n : Nat} {eps : α}
    (backend : CheckedApproxOrthogonalizer α m n eps)
    (lr momentum : α) (params grads : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum backend.orthogonalizer params)
          params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedStep eps
        (_root_.Optim.Muon.init lr momentum backend.orthogonalizer params)
        params grads direction :=
  _root_.Optim.Muon.init_has_approx_certified_step_of_checked_backend
    backend lr momentum params grads hsuccess

/--
Initialized checked-backend theorem for approximate Muon: after `init`, backend success gives the
Gram-residual bound for the direction used by the first update.
-/
theorem init_direction_has_approx_column_gram_of_checked_backend
    {α : Type} [Context α] {m n : Nat} {eps : α}
    (backend : CheckedApproxOrthogonalizer α m n eps)
    (lr momentum : α) (params grads : MatrixTensor α m n)
    (hsuccess :
      backend.Success
        (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum backend.orthogonalizer params)
          params grads).1.buf) :
    HasApproxColumnGram eps
      (backend.orthogonalizer.apply
        (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum backend.orthogonalizer params)
          params grads).1.buf) :=
  _root_.Optim.Muon.init_direction_has_approx_column_gram_of_checked_backend
    backend lr momentum params grads hsuccess

/--
Newton-Schulz checked theorem: if the backend output on the fresh momentum buffer satisfies the
Gram-residual bound, Muon uses an approximately certified direction.
-/
theorem update_has_approx_certified_direction_newtonSchulz_checked
    {α : Type} [Context α] {m n : Nat} {eps : α}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat)
    (lr momentum : α) (buf params grads : MatrixTensor α m n)
    (hresidual :
      ResidualApproxSuccess eps
        (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedDirection eps
        (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).1.buf direction ∧
      (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).2 =
        _root_.Spec.Tensor.subSpec params (_root_.Spec.Tensor.scaleSpec direction lr) :=
  _root_.Optim.Muon.update_has_approx_certified_direction_newtonSchulz_checked
    coeffs steps lr momentum buf params grads hresidual

/--
Newton-Schulz checked step theorem: if the backend output on the fresh momentum buffer satisfies
the Gram-residual bound, Muon has a certified approximate step.
-/
theorem update_has_approx_certified_step_newtonSchulz_checked
    {α : Type} [Context α] {m n : Nat} {eps : α}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat)
    (lr momentum : α) (buf params grads : MatrixTensor α m n)
    (hresidual :
      ResidualApproxSuccess eps
        (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedStep eps
        ({ lr := lr, momentum := momentum, buf := buf,
           orthogonalizer :=
            newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
          _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
        params grads direction :=
  _root_.Optim.Muon.update_has_approx_certified_step_newtonSchulz_checked
    coeffs steps lr momentum buf params grads hresidual

/--
Newton-Schulz checked direction theorem: if the backend output on the fresh momentum buffer
satisfies the Gram-residual bound, the actual update direction satisfies that same bound.
-/
theorem update_newtonSchulz_direction_has_approx_column_gram_checked
    {α : Type} [Context α] {m n : Nat} {eps : α}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat)
    (lr momentum : α) (buf params grads : MatrixTensor α m n)
    (hresidual :
      ResidualApproxSuccess eps
        (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    HasApproxColumnGram eps
      ((newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps).apply
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer :=
              newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps } :
            _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
          params grads).1.buf) :=
  _root_.Optim.Muon.update_newtonSchulz_direction_has_approx_column_gram_checked
    coeffs steps lr momentum buf params grads hresidual

/--
Initialized Newton-Schulz checked theorem: if the first fresh momentum buffer satisfies the
Gram-residual bound, Muon uses an approximately certified direction.
-/
theorem init_has_approx_certified_direction_newtonSchulz_checked
    {α : Type} [Context α] {m n : Nat} {eps : α}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat)
    (lr momentum : α) (params grads : MatrixTensor α m n)
    (hresidual :
      ResidualApproxSuccess eps
        (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
        (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum
            (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
            params)
          params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedDirection eps
        (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
        (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum
            (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
            params)
          params grads).1.buf direction ∧
      (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum
            (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
            params)
          params grads).2 =
        _root_.Spec.Tensor.subSpec params (_root_.Spec.Tensor.scaleSpec direction lr) :=
  _root_.Optim.Muon.init_has_approx_certified_direction_newtonSchulz_checked
    coeffs steps lr momentum params grads hresidual

/--
Initialized Newton-Schulz checked step theorem: if the first fresh momentum buffer satisfies the
Gram-residual bound, Muon has a certified approximate first step.
-/
theorem init_has_approx_certified_step_newtonSchulz_checked
    {α : Type} [Context α] {m n : Nat} {eps : α}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat)
    (lr momentum : α) (params grads : MatrixTensor α m n)
    (hresidual :
      ResidualApproxSuccess eps
        (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
        (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum
            (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
            params)
          params grads).1.buf) :
    ∃ direction : MatrixTensor α m n,
      ApproxCertifiedStep eps
        (_root_.Optim.Muon.init lr momentum
          (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
          params)
        params grads direction :=
  _root_.Optim.Muon.init_has_approx_certified_step_newtonSchulz_checked
    coeffs steps lr momentum params grads hresidual

/--
Initialized Newton-Schulz checked direction theorem: if the first fresh momentum buffer satisfies
the Gram-residual bound, the first update direction satisfies that same bound.
-/
theorem init_newtonSchulz_direction_has_approx_column_gram_checked
    {α : Type} [Context α] {m n : Nat} {eps : α}
    (coeffs : NewtonSchulzCoeffs α) (steps : Nat)
    (lr momentum : α) (params grads : MatrixTensor α m n)
    (hresidual :
      ResidualApproxSuccess eps
        (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
        (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum
            (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
            params)
          params grads).1.buf) :
    HasApproxColumnGram eps
      ((newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps).apply
        (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum
            (newtonSchulzOrthogonalizer (α := α) (m := m) (n := n) coeffs steps)
            params)
          params grads).1.buf) :=
  _root_.Optim.Muon.init_newtonSchulz_direction_has_approx_column_gram_checked
    coeffs steps lr momentum params grads hresidual

/-- The exact backend contract gives `QᵀQ = I` for the direction used by the Muon update. -/
theorem update_direction_has_exact_column_gram
    {α : Type} [Context α] {m n : Nat}
    (state : _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
    (params grads : MatrixTensor α m n)
    (horth : ExactMatrixOrthogonalizer state.orthogonalizer) :
    HasExactColumnGram (state.orthogonalizer.apply
      (_root_.Optim.Muon.update state params grads).1.buf) :=
  _root_.Optim.Muon.update_direction_has_exact_column_gram state params grads horth

/--
The approximate backend contract gives the entrywise Gram-residual bound for the direction used by
the Muon update.
-/
theorem update_direction_has_approx_column_gram
    {α : Type} [Context α] {m n : Nat} {eps : α}
    (state : _root_.Optim.Muon.State α (.dim m (.dim n .scalar)))
    (params grads : MatrixTensor α m n)
    (horth : ApproxMatrixOrthogonalizer eps state.orthogonalizer) :
    HasApproxColumnGram eps (state.orthogonalizer.apply
      (_root_.Optim.Muon.update state params grads).1.buf) :=
  _root_.Optim.Muon.update_direction_has_approx_column_gram state params grads horth

/--
QR-backed Muon theorem: if the fresh momentum buffer has positive QR pivots, the executable Muon
step uses a direction with exact column Gram `I`.
-/
theorem update_has_exact_certified_direction_qr {m n : Nat}
    (lr momentum : ℝ) (buf params grads : MatrixTensor ℝ m n)
    (hpivots :
      HasPositiveQRPivots
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := qrOrthogonalizer (m := m) (n := n) } :
            _root_.Optim.Muon.State ℝ (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    ∃ direction : MatrixTensor ℝ m n,
      ExactCertifiedDirection (qrOrthogonalizer (m := m) (n := n))
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := qrOrthogonalizer (m := m) (n := n) } :
            _root_.Optim.Muon.State ℝ (.dim m (.dim n .scalar)))
          params grads).1.buf direction ∧
      (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := qrOrthogonalizer (m := m) (n := n) } :
            _root_.Optim.Muon.State ℝ (.dim m (.dim n .scalar)))
          params grads).2 =
        _root_.Spec.Tensor.subSpec params (_root_.Spec.Tensor.scaleSpec direction lr) :=
  _root_.Optim.Muon.update_has_exact_certified_direction_qr lr momentum buf params grads hpivots

/--
QR-backed Muon step theorem: if the fresh momentum buffer has positive QR pivots, the executable
Muon step is exactly certified.
-/
theorem update_has_exact_certified_step_qr {m n : Nat}
    (lr momentum : ℝ) (buf params grads : MatrixTensor ℝ m n)
    (hpivots :
      HasPositiveQRPivots
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := qrOrthogonalizer (m := m) (n := n) } :
            _root_.Optim.Muon.State ℝ (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    ∃ direction : MatrixTensor ℝ m n,
      ExactCertifiedStep
        ({ lr := lr, momentum := momentum, buf := buf,
           orthogonalizer := qrOrthogonalizer (m := m) (n := n) } :
          _root_.Optim.Muon.State ℝ (.dim m (.dim n .scalar)))
        params grads direction :=
  _root_.Optim.Muon.update_has_exact_certified_step_qr lr momentum buf params grads hpivots

/--
QR-backed direction theorem: if the fresh momentum buffer has positive QR pivots, the actual Muon
update direction has column Gram `I`.
-/
theorem update_qr_direction_has_exact_column_gram {m n : Nat}
    (lr momentum : ℝ) (buf params grads : MatrixTensor ℝ m n)
    (hpivots :
      HasPositiveQRPivots
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := qrOrthogonalizer (m := m) (n := n) } :
            _root_.Optim.Muon.State ℝ (.dim m (.dim n .scalar)))
          params grads).1.buf) :
    HasExactColumnGram
      ((qrOrthogonalizer (m := m) (n := n)).apply
        (_root_.Optim.Muon.update
          ({ lr := lr, momentum := momentum, buf := buf,
             orthogonalizer := qrOrthogonalizer (m := m) (n := n) } :
            _root_.Optim.Muon.State ℝ (.dim m (.dim n .scalar)))
          params grads).1.buf) :=
  _root_.Optim.Muon.update_qr_direction_has_exact_column_gram
    lr momentum buf params grads hpivots

/--
Initialized QR-backed Muon theorem: if the first fresh momentum buffer has positive QR pivots, the
first update uses a direction with exact column Gram `I`.
-/
theorem init_has_exact_certified_direction_qr {m n : Nat}
    (lr momentum : ℝ) (params grads : MatrixTensor ℝ m n)
    (hpivots :
      HasPositiveQRPivots
        (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum (qrOrthogonalizer (m := m) (n := n)) params)
          params grads).1.buf) :
    ∃ direction : MatrixTensor ℝ m n,
      ExactCertifiedDirection (qrOrthogonalizer (m := m) (n := n))
        (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum (qrOrthogonalizer (m := m) (n := n)) params)
          params grads).1.buf direction ∧
      (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum (qrOrthogonalizer (m := m) (n := n)) params)
          params grads).2 =
        _root_.Spec.Tensor.subSpec params (_root_.Spec.Tensor.scaleSpec direction lr) :=
  _root_.Optim.Muon.init_has_exact_certified_direction_qr lr momentum params grads hpivots

/--
Initialized QR-backed Muon step theorem: if the first fresh momentum buffer has positive QR pivots,
the first update is exactly certified.
-/
theorem init_has_exact_certified_step_qr {m n : Nat}
    (lr momentum : ℝ) (params grads : MatrixTensor ℝ m n)
    (hpivots :
      HasPositiveQRPivots
        (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum (qrOrthogonalizer (m := m) (n := n)) params)
          params grads).1.buf) :
    ∃ direction : MatrixTensor ℝ m n,
      ExactCertifiedStep
        (_root_.Optim.Muon.init lr momentum (qrOrthogonalizer (m := m) (n := n)) params)
        params grads direction :=
  _root_.Optim.Muon.init_has_exact_certified_step_qr lr momentum params grads hpivots

/--
Initialized QR-backed direction theorem: if the first fresh momentum buffer has positive QR pivots,
the first update direction has column Gram `I`.
-/
theorem init_qr_direction_has_exact_column_gram {m n : Nat}
    (lr momentum : ℝ) (params grads : MatrixTensor ℝ m n)
    (hpivots :
      HasPositiveQRPivots
        (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum (qrOrthogonalizer (m := m) (n := n)) params)
          params grads).1.buf) :
    HasExactColumnGram
      ((qrOrthogonalizer (m := m) (n := n)).apply
        (_root_.Optim.Muon.update
          (_root_.Optim.Muon.init lr momentum (qrOrthogonalizer (m := m) (n := n)) params)
          params grads).1.buf) :=
  _root_.Optim.Muon.init_qr_direction_has_exact_column_gram lr momentum params grads hpivots

end muon

end optim

end TorchLean
