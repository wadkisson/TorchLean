/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Training.Core.Base
public import NN.API.Public.Training.Core.Manual

/-!
# Manual Training API

Umbrella import for direct runners, callback loops, losses, metrics, schedules, and training
artifacts. Ordinary application code should use `TorchLean.Trainer`; optimizer configuration lives
in the single public namespace `TorchLean.optim`.
-/

@[expose] public section
