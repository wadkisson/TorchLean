/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Compiled.IRExec.Helpers
public import NN.Runtime.Autograd.Compiled.IRExec.Lowering
public import NN.Runtime.Autograd.Compiled.IRExec.API

/-!
# IR Graph Lowering

Umbrella import for lowering helpers, the exhaustive node compiler, and the public graph entrypoint.
-/

@[expose] public section
