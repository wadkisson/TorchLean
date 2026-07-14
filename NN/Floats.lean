/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.Arb
public import NN.Floats.Calc
public import NN.Floats.FP32
public import NN.Floats.Float32
public import NN.Floats.IEEEExec
public import NN.Floats.Interval
public import NN.Floats.NeuralFloat

/-!
# Floating-Point Semantics

Import this file when you want the floating-point semantics in one place:

- proof-oriented real-valued models (`FP32`, `NeuralFloat`),
- effective rounded-arithmetic components (`Calc`),
- the executable bit-level model (`IEEEExec`),
- interval/enclosure utilities (`Interval`),
- the external Arb oracle integration (`Arb`),
- and the shared error-bound vocabulary used across the library.

The focused `NN.Floats.*` subsystems are collected here so downstream users have one stable import.
-/

@[expose] public section
