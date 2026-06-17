/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Dropout
public import NN.Runtime.Context
public import NN.Proofs.Tensor.Basic.LinearAlgebra

/-!
# Dropout analysis properties

TorchLean splits stochastic training-mode dropout into two pieces:

- a mask/seed producer, treated as non-differentiated data in autograd proofs, and
- a deterministic tensor map once the mask or inference probability is fixed.

This file records small spec-level identities for the deterministic inference map. The fixed-mask
training-mode derivative infrastructure lives with the autograd tape-node proofs.

Reference: Srivastava et al., 2014, “Dropout: A Simple Way to Prevent Neural Networks from
Overfitting”.
-/

@[expose] public section

namespace Proofs

open _root_.Spec
open _root_.Spec.Tensor

noncomputable section

/--
Deterministic dropout inference scaling is the identity when `p = 0`.

Inference dropout multiplies activations by the keep/dropout scaling factor from the spec. At zero
dropout probability that factor is `1`, so the whole tensor is unchanged.
-/
theorem dropout_inference_spec_p0_eq_id {s : Shape} (x : Tensor ℝ s) :
    Spec.dropoutInferenceSpec (α := ℝ) (s := s) (p := (0 : ℝ)) x = x := by
  -- Reduce to pointwise scaling by `1`, then use the shared tensor-map identity.
  simp [Spec.dropoutInferenceSpec, Spec.Tensor.scaleSpec]
  induction x with
  | scalar v => rfl
  | dim f ih =>
      simp [Spec.Tensor.mapSpec]
      funext i
      exact ih i

end

end Proofs
