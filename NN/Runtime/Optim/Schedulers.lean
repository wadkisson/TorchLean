/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Optim.Schedulers.Native
public import NN.Runtime.Optim.Schedulers.PyTorch

/-!
# Learning-Rate Schedulers

This umbrella exports TorchLean's native total schedules and the variants that reproduce PyTorch
step-count and phase conventions. Shared scheduler arithmetic lives in
`NN.Runtime.Optim.Schedulers.Core`.
-/
