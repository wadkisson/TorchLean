/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Quickstart.TensorBasics
public import NN.Examples.Quickstart.Common
public import NN.Examples.Quickstart.StarterWorkflow
public import NN.Examples.Quickstart.Widgets
public import NN.Examples.Quickstart.AutogradBasics
public import NN.Examples.Quickstart.SimpleMlpTrain
public import NN.Examples.Quickstart.MinibatchMlpTrain
public import NN.Examples.Quickstart.SimpleCnnTrain
public import NN.Examples.Quickstart.Proofs

/-!
# Quickstart

Curated first-tour examples for TorchLean.

This umbrella is narrower than `NN.Examples.Zoo`. It teaches the primitives a new user
needs before opening the model zoo:

- typed tensors and runtime scalar choices,
- a minimal public-only training setup,
- editor widgets for inspecting tensors, floats, graphs, and logs,
- autograd helpers for gradients, Jacobians, Hessian-vector products, and detach,
- compact end-to-end MLP/CNN training loops, and
- the proof/compile-time side of TorchLean's shape-indexed API.

Larger ResNet, data-loader, PyTorch interop, RL, and verification examples remain in their
specialized folders. The point of `Quickstart` is a clean on-ramp, not another model zoo.
-/

@[expose] public section
