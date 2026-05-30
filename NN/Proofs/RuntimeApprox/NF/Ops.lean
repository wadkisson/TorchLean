/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.Ops.Plumbing
public import NN.Proofs.RuntimeApprox.NF.Ops.Scalar
public import NN.Proofs.RuntimeApprox.NF.Ops.Sum
public import NN.Proofs.RuntimeApprox.NF.Ops.Elementwise
public import NN.Proofs.RuntimeApprox.NF.Ops.Nodes

/-!
# NF Primitive Ops

Forward-error lemmas for the rounded `NF` backend.

This module collects the public NF operation facts used by runtime-approximation proofs: scalar
rounding bounds, tensor-level arithmetic and activation bounds, rounded reductions, and the
`FwdNode` constructors that package those facts for graph composition.
-/
