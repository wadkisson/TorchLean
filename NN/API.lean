/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade
public import NN.API.Adapters
public import NN.API.Models.Generative
public import NN.API.SelfSupervised

/-!
# TorchLean API

The focused import for applications that want TorchLean's concise `TorchLean.*` interface without
loading the proof, verification, floating-point, and widget subsystems.

The main namespaces are `TorchLean.nn` for neural networks, `TorchLean.classical` for classical
and statistical models, `TorchLean.Trainer` for training, `TorchLean.optim` for optimizers, and
`TorchLean.Data` for datasets. Use `import NN` when the same file also needs specifications,
proofs, verification, or backend internals.
-/

@[expose] public section
