/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.Calc
public import NN.Floats.FP32
public import NN.Floats.Float32
public import NN.Floats.IEEEExec
public import NN.Floats.Interval
public import NN.Floats.NeuralFloat
public import NN.Floats.Quantization

/-!
# Floating-Point Semantics

Import this file when you want the floating-point semantics in one place:

- proof-oriented real-valued models (`FP32`, `NeuralFloat`),
- effective rounded-arithmetic components (`Calc`),
- the executable bit-level model (`IEEEExec`),
- interval/enclosure utilities (`Interval`),
- and the shared error-bound vocabulary used across the library.

The focused, Lean-native `NN.Floats.*` subsystems are collected here so downstream users have one
stable import without pulling in tensors, models, runtimes, CUDA, or external processes. The
optional Arb oracle is available separately through `NN.Floats.Arb`.
-/

@[expose] public section
