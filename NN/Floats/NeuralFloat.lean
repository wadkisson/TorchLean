/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Analysis
public import NN.Floats.NeuralFloat.Core
public import NN.Floats.NeuralFloat.Error
public import NN.Floats.NeuralFloat.Format
public import NN.Floats.NeuralFloat.Metadata
public import NN.Floats.NeuralFloat.Rounding
public import NN.Floats.NeuralFloat.Scalar
public import NN.Floats.NeuralFloat.Special

/-!
# `NN.Floats.NeuralFloat`

This is the “rounding-on-ℝ” side of TorchLean’s float models. We use it when we want to talk about
formats and rounding generically, without committing to a concrete IEEE bit encoding:

- generic format/rounding infrastructure,
- the rounded scalar type `NF`,
- ULP-style error bounds and small calculation libraries.

Most files that are *not* bit-level IEEE execution live under `NN/Floats/NeuralFloat/`.
-/

@[expose] public section
