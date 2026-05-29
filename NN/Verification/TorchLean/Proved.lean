/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Syntax
public import NN.Verification.TorchLean.Proved.Compile
public import NN.Verification.TorchLean.Proved.Correctness
public import NN.Verification.TorchLean.Proved.Public

/-!
# Verified Forward Fragment

A first-order TorchLean forward language, its compiler into the verifier IR, and the checked theorem
that compiled IR evaluation agrees with the source evaluator.
-/
