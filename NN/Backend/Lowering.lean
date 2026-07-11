/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Audit
public import NN.Backend.IR

/-!
# Backend Lowering Plans

Graph-aware lowering data for backend execution.

`GraphExecutionPlan` chooses a capsule for each runtime-relevant IR node. A `GraphLoweringPlan`
groups those choices into backend calls. The conservative lowering in this file coalesces adjacent
nodes only when they use the same operation and the exact same capsule contract. More aggressive
fusion can be added later as new capsules with their own contracts.
-/

@[expose] public section

namespace NN
namespace Backend

/-- Why adjacent planned nodes share one audit/scheduling group. This does not assert kernel fusion. -/
inductive GroupingKind where
  | singleton
  | sameCapsuleRun
  deriving Repr

/-- One lowered backend call, with source IR nodes kept for audit and debugging. -/
structure LoweredKernelGroup where
  nodeIds : List Nat
  kinds : List NN.IR.OpKind
  op : BackendOp
  capsule : KernelCapsule
  groupingKind : GroupingKind := .singleton
  deriving Repr

/-- Graph lowering plan: backend groups in execution order. -/
structure GraphLoweringPlan where
  groups : List LoweredKernelGroup
  deriving Repr

namespace LoweredKernelGroup

/-- Erase lowering metadata to the older single-kernel execution-plan row. -/
def toPlannedKernel (g : LoweredKernelGroup) : PlannedKernel :=
  { op := g.op, capsule := g.capsule }

/-- Whether this lowered group crosses a trusted-external boundary. -/
def hasTrustedExternal (g : LoweredKernelGroup) : Bool :=
  g.capsule.isTrustedExternal

/-- Whether a planned node can be appended to this lowered backend group. -/
def canAppend (g : LoweredKernelGroup) (k : IR.PlannedNodeKernel) : Bool :=
  g.op == k.op && g.capsule.sameIdentity k.capsule

/-- Append a node to an existing backend group, preserving source-node provenance. -/
def appendNode (g : LoweredKernelGroup) (k : IR.PlannedNodeKernel) : LoweredKernelGroup :=
  { g with
    nodeIds := g.nodeIds ++ [k.nodeId]
    kinds := g.kinds ++ [k.kind]
    groupingKind := .sameCapsuleRun }

end LoweredKernelGroup

namespace GraphLoweringPlan

/-- Source IR node ids covered by the lowering plan, in execution order. -/
def nodeIds (p : GraphLoweringPlan) : List Nat :=
  p.groups.foldr (fun g acc => g.nodeIds ++ acc) []

/-- Selected capsule names, in lowering order. -/
def capsuleNames (p : GraphLoweringPlan) : List String :=
  p.groups.map fun g => g.capsule.name

/-- Erase lowering metadata to the audit/execution-plan view. -/
def toExecutionPlan (p : GraphLoweringPlan) : ExecutionPlan :=
  { kernels := p.groups.map LoweredKernelGroup.toPlannedKernel }

/-- Audit the selected backend boundaries of the lowering plan. -/
def audit (p : GraphLoweringPlan) : ExecutionAudit :=
  p.toExecutionPlan.audit

/-- Whether any lowered group crosses a trusted-external boundary. -/
def hasTrustedExternal (p : GraphLoweringPlan) : Bool :=
  p.groups.any LoweredKernelGroup.hasTrustedExternal

end GraphLoweringPlan

namespace IR

namespace PlannedNodeKernel

/-- Lower one graph-planned node to a singleton backend group. -/
def toSingletonLoweredGroup (k : PlannedNodeKernel) : LoweredKernelGroup :=
  { nodeIds := [k.nodeId]
    kinds := [k.kind]
    op := k.op
    capsule := k.capsule
    groupingKind := .singleton }

end PlannedNodeKernel

namespace GraphExecutionPlan

/-- Default lowering: one backend group per runtime-relevant IR node. -/
def toSingletonLoweringPlan (p : GraphExecutionPlan) : GraphLoweringPlan :=
  { groups := p.kernels.map PlannedNodeKernel.toSingletonLoweredGroup }

/-- Fold state for conservative same-boundary lowering. -/
structure LoweringState where
  groupsRev : List LoweredKernelGroup
  deriving Repr

namespace LoweringState

/-- Add one planned node to the lowering state, coalescing with the previous group when safe. -/
def pushKernel (s : LoweringState) (k : PlannedNodeKernel) : LoweringState :=
  match s.groupsRev with
  | [] =>
      { groupsRev := [k.toSingletonLoweredGroup] }
  | g :: rest =>
      if g.canAppend k then
        { groupsRev := g.appendNode k :: rest }
      else
        { groupsRev := k.toSingletonLoweredGroup :: s.groupsRev }

end LoweringState

/--
Conservative grouping: adjacent nodes with the same backend boundary share one audit/scheduling
group.

This does not claim that one kernel invocation implements the group. A future fused capsule must
declare its multi-node pattern and execution contract explicitly.
-/
def toCoalescedLoweringPlan (p : GraphExecutionPlan) : GraphLoweringPlan :=
  let s := p.kernels.foldl LoweringState.pushKernel { groupsRev := [] }
  { groups := s.groupsRev.reverse }

end GraphExecutionPlan

end IR

end Backend
end NN
