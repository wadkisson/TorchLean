/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor.API

/-!
# Tensor entrypoint

Curated umbrella import for TorchLean's core tensor/shape API.

Use this import for tensor literals, shape aliases, dynamic tensors, and small executable tensor
helpers. `NN.Tensor.API` remains the implementation leaf; downstream users should prefer either
`NN.Library` for the full public API or `NN.Entrypoint.Tensor` for the tensor layer alone.
-/

@[expose] public section
