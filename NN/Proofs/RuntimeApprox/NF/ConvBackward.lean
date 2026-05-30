/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.ConvBackward.Common
public import NN.Proofs.RuntimeApprox.NF.ConvBackward.BiasKernel
public import NN.Proofs.RuntimeApprox.NF.ConvBackward.Input
public import NN.Proofs.RuntimeApprox.NF.ConvBackward.RevNode

/-!
NeuralFloat convolution-backward approximation proofs.

This entry point gathers the bias, input, kernel, and reverse-node bounds used to relate
finite-precision convolution gradients to their real-valued specifications.
-/
