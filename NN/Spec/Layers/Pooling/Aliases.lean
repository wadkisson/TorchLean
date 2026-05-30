/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Pooling.ND

@[expose] public section


namespace Spec
open Tensor
open Spec (Image MultiChannelImage getValueAtPosition extractWindow)

variable {α : Type} [Context α]

/-!
# Pooling Aliases

Short names for the dimension-polymorphic pooling specs.
-/

/-!
### Friendly aliases
-/

/-- Alias for `max_pool_spec`. -/
abbrev maxPool := @maxPoolSpec

/-- Alias for `avg_pool_spec`. -/
abbrev avgPool := @avgPoolSpec

/-- Alias for `smooth_max_pool_spec`. -/
abbrev smoothMaxPool := @smoothMaxPoolSpec

/-- Alias for `max_pool_backward_spec`. -/
abbrev maxPoolBackward := @maxPoolBackwardSpec

/-- Alias for `avg_pool_backward_spec`. -/
abbrev avgPoolBackward := @avgPoolBackwardSpec

/-- Alias for `smooth_max_pool_backward_spec`. -/
abbrev smoothMaxPoolBackward := @smoothMaxPoolBackwardSpec
end Spec
