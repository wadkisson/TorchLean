/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.Arb
public import NN.Floats.FP32
public import NN.Floats.Float32
public import NN.Floats.IEEEExec
public import NN.Floats.Interval
public import NN.Floats.NeuralFloat

/-!
# Floats entrypoint

Import this file when you want the floating-point semantics in one place:

- proof-oriented real-valued models (`FP32`, `NeuralFloat`),
- the executable bit-level model (`IEEEExec`),
- interval/enclosure utilities (`Interval`),
- the external Arb oracle integration (`Arb`),
- and the shared error-bound vocabulary used across the library.

The entrypoint imports the focused `NN.Floats.*` subsystems directly so downstream users have one
stable path.
-/

@[expose] public section
