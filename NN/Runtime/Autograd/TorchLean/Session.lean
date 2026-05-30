/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.Session.Eager
public import NN.Runtime.Autograd.TorchLean.Session.Types
public import NN.Runtime.Autograd.TorchLean.Session.Ops
public import NN.Runtime.Autograd.TorchLean.Session.ShapeIndex
public import NN.Runtime.Autograd.TorchLean.Session.Neural
public import NN.Runtime.Autograd.TorchLean.Session.Autograd

/-!
Session API for runtime graph execution.

The session layer owns shape-indexed runtime values, autograd handles, operation dispatch, and the
state needed to keep host and CUDA mirrors coherent.
-/
