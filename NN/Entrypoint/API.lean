/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public
public import NN.API.Adapters
public import NN.API.Models.Generative
public import NN.API.SelfSupervised

/-!
# API entrypoint

This entrypoint re-exports the primary user-facing API surface, `NN.API.Public`, beside the other
curated `NN.Entrypoint.*` imports.

Ordinary model and training files should use `import NN`. Import this module when you want
the same public API surface under the `NN.Entrypoint.*` tree, without the broader library umbrella.
-/

@[expose] public section
