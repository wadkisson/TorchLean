/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Nodes.Losses.MSE
public import NN.Proofs.Autograd.Tape.Nodes.Losses.CrossEntropy
public import NN.Proofs.Autograd.Tape.Nodes.Losses.NLL
public import NN.Proofs.Autograd.Tape.Nodes.Losses.BCEWithLogits
public import NN.Proofs.Autograd.Tape.Nodes.Losses.KLDivergence

/-!
# Loss Tape Nodes

Differentiable loss nodes used by training and verification examples: MSE, cross entropy with
one-hot targets, negative log likelihood, BCE-with-logits, and KL divergence.
-/
