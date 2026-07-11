/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Cert
public import NN.Verification.Geometry3D
public import NN.Verification.ODE
public import NN.Verification.PINN
public import NN.Verification.Robustness
public import NN.Verification.Splines
public import NN.Verification.TorchLean.Compile
public import NN.Verification.TorchLean.CompileExec
public import NN.Verification.TorchLean.Verified
public import NN.Verification.Util.Json
public import NN.Verification.VNNComp
public import NN.MLTheory.CROWN.Proofs.Overview

/-!
# Verification entrypoint

Import this file for TorchLean’s verification infrastructure: JSON utilities, certificate formats,
ODE/PINN-style checkers, proof-backed certificate soundness, and the proof-backed TorchLean-to-IR
forward compiler bridge.

We keep one verification entrypoint on purpose. The compiler bridge is imported here
through the naming-friendly `NN.Verification.TorchLean.Verified` API, so users do not have to choose
between separate “proved” and “verified” umbrellas.

Runnable CLIs stay out of this umbrella. If you want a command-line tool, import
the registry explicitly (for example `NN.Verification.CLI`).

The underlying CROWN/LiRPA soundness development enters here too. Executable examples can parse
JSON artifacts and recompute bounds inside Lean; theorem-level credit comes from those imported
soundness modules, where locally valid certificates are connected to Lean graph semantics.
-/

@[expose] public section
