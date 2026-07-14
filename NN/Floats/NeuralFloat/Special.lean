/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Special.FTZ

/-!
# Special Execution Policies

Execution policies that alter the generic format semantics live here.  Flush-to-zero is kept out
of the format core because it removes subnormal results and therefore changes both representability
and error behavior.
-/

@[expose] public section
