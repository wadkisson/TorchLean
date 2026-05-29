/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape.ShapeChange
public import NN.Spec.Core.TensorReductionShape.Broadcasting
public import NN.Spec.Core.TensorReductionShape.Reductions
public import NN.Spec.Core.TensorReductionShape.LinearAlgebra
public import NN.Spec.Core.TensorReductionShape.ConcatSlice

/-!
# Tensor Reductions and Shape Helpers

Shape-changing tensor specs, broadcasting, reductions, linear-algebra helpers, concatenation, and
slicing. The original import path remains the public entrypoint; the implementation lives in focused
submodules by operation family.
-/
