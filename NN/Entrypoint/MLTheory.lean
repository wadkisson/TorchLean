/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.API

/-!
# ML theory entrypoint

Curated umbrella import for TorchLean's ML-theory subsystem (robustness, verification-oriented
theory modules, CROWN/Lyapunov infrastructure, etc.).

`NN.MLTheory.API` remains the canonical curated import set. This entrypoint gives downstream code a
stable "one import" path consistent with the other subsystems.
-/

@[expose] public section
