/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Training.Core

/-!
# Public Training Entrypoint

Import-only entrypoint for manual public training APIs. Ordinary code should use
`import NN` and train through `Trainer.new` plus `trainer.train`; advanced manual-loop
operations live in `NN.API.Public.Training.Core`.
-/
