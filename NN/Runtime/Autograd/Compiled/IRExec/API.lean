/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Compiled.IRExec.Lowering

/-!
# Compiled IR Execution API

Public entrypoint for validating and lowering a complete shared IR graph.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Compiled

open Spec
open Tensor
open Proofs.Autograd.Algebra
open NN.IR

/--
Compile an op-tagged IR graph into an executable SSA graph (`GraphData`) for forward evaluation.

Requirements:
- Node id 0 must be `.input`.
- The graph must satisfy `Graph.checkWellFormed`.
- The external payload must contain entries for every `.const`/`.linear`/`.conv2d` node id.

This returns an `ExecGraphData` whose `eval` computes all node values in topo order.

This is the main API consumed by runtime callers that want executable evaluation while remaining
aligned with the shared `NN.IR.Graph` semantics.
-/
def execGraphOfIR
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) : Except String (ExecGraphData α) := do
  g.checkWellFormed
  let n0 ← g.getNode 0
  match n0.kind with
  | .input =>
      let inShape := n0.outShape
      let stFinal ← IRExec.buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := 1) (st := (⟨[], .nil⟩ : IRExec.State α inShape))
      let ⟨ss, gd⟩ := stFinal
      pure { inShape := inShape, ss := ss, g := gd }
  | _ =>
      throw s!"IRExec: node 0 is not `.input` (got {n0.kind.tag})"
