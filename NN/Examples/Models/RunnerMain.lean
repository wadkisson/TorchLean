/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Models.Runner

/-!
# TorchLean Example Runner Entry Point

The reusable command dispatcher lives in `NN.Examples.Models.Runner`. Keeping the global executable
entry point here lets profiling tools and other Lean programs import that dispatcher without also
importing a competing declaration named `main`.
-/

@[expose] public section

def main (args : List String) : IO UInt32 :=
  NN.Examples.Models.Runner.main args
