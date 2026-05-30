/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Ops.Conv.BackwardDot.Common
public import NN.Proofs.Autograd.Tape.Ops.Conv.BackwardDot.Kernel
public import NN.Proofs.Autograd.Tape.Ops.Conv.BackwardDot.Input
public import NN.Proofs.Autograd.Tape.Ops.Conv.BackwardDot.Main

/-!
Convolution backward-dot proof entry point.

The split files isolate the input, weight, and main algebraic components needed to justify
convolution-gradient computations in the tape semantics.
-/
