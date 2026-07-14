/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Ops.Convolution
public import NN.Runtime.Autograd.Torch.Core.Ops.Elementwise
public import NN.Runtime.Autograd.Torch.Core.Ops.Indexing
public import NN.Runtime.Autograd.Torch.Core.Ops.Layers
public import NN.Runtime.Autograd.Torch.Core.Ops.LinearAlgebra
public import NN.Runtime.Autograd.Torch.Core.Ops.Pooling
public import NN.Runtime.Autograd.Torch.Core.Ops.ShapeReduction

/-!
# Eager Tensor Operations

Typed eager operations record CPU or CUDA tape nodes through a shared dispatch layer. The focused
modules group operations by behavior; this umbrella preserves the complete eager operation API.
-/
