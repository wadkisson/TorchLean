/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Ops.Attention

/-!
CUDA operation dispatch surface.

The submodules separate tensor views, reductions, neural-network kernels, linear algebra, FFT, and
other CUDA-backed eager operations while keeping the public import path stable.
-/
