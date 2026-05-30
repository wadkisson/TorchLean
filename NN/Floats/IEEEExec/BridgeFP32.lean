/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.BridgeFP32.Compare

/-!
Bridge lemmas between executable IEEE32 arithmetic and the FP32-facing API.

The submodules expose operation-level, rounding, and totality facts for verification code that
uses binary32 semantics rather than Lean's host `Float` behavior.
-/
