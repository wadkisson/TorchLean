/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorOps

/-!
# Dropout (deterministic spec)

Dropout is traditionally randomized: each element is kept with probability `keep = 1 - p`.
In this repository we often want a *deterministic* spec that still documents the intended meaning,
so downstream models can choose explicit inference-time or mask-driven dropout semantics.

We therefore expose two deterministic variants:

- `dropoutInferenceSpec p x = x`, matching evaluation mode for inverted dropout.

- `dropoutMaskedSpec p mask x`
  A fully deterministic "training-style" dropout that takes the mask explicitly. We use safe
  scaling by `max(keep, ε)` so it is always defined even if `p ≈ 1`.

How this differs from PyTorch:

- `torch.nn.Dropout(p)` uses **inverted dropout** during training: `y = mask * x / (1 - p)`,
  and becomes identity during evaluation (`y = x`).
- The spec layer here avoids randomness. If you want something close to PyTorch *training*
  semantics,
  use `dropoutMaskedSpec` and pass the mask explicitly. For evaluation semantics, use
  `dropoutInferenceSpec`.

Gradients:

- We treat `p` and `mask` as non-differentiable inputs. The backward specs only return the gradient
  with respect to `x`.
-/

@[expose] public section


namespace Spec

open Tensor
open Numbers

variable {α : Type} [Context α]

/-- Evaluation-mode dropout. The probability is retained in the signature because it belongs to
the layer configuration, but evaluation itself is the identity. -/
def dropoutInferenceSpec {s : Shape} (p : α) (x : Tensor α s) : Tensor α s :=
  let _ := p
  x

/-- Backward/VJP for evaluation-mode dropout: the cotangent is unchanged. -/
def dropoutInferenceBackwardSpec {s : Shape} (p : α) (grad_output : Tensor α s) : Tensor α s :=
  let _ := p
  grad_output

/-- Deterministic training-style dropout with an explicit mask.

If `mask[i] = true`, keep element `x[i]`, otherwise drop it to `0`.
We use inverted-dropout scaling `x / keepSafe` with `keepSafe = max(1 - p, ε)`. -/
def dropoutMaskedSpec {s : Shape} (p : α) (mask : Tensor Bool s) (x : Tensor α s) : Tensor α s :=
  let keep : α := (1 : α) - p
  let keepSafe : α := Max.max keep Numbers.epsilon
  map2Spec (fun xi mi => if mi then xi / keepSafe else 0) x mask

/-- Backward/VJP for `dropoutMaskedSpec` with respect to `x`.

This mirrors the forward: gradients are masked and (in the kept positions) scaled by `1/keepSafe`.
  -/
def dropoutMaskedBackwardSpec {s : Shape} (p : α) (mask : Tensor Bool s) (grad_output : Tensor α
  s) : Tensor α s :=
  let keep : α := (1 : α) - p
  let keepSafe : α := Max.max keep Numbers.epsilon
  map2Spec (fun gi mi => if mi then gi / keepSafe else 0) grad_output mask

end Spec
