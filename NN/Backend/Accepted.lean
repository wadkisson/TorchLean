/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Gate

/-!
# Accepted Backend Plans

One entry point for the backend-contract pipeline.

This module ties together target availability, graph planning, conservative lowering, recheck, and
acceptance gates. It is intentionally still data-level: it produces a lowered plan that has passed a
policy gate, or it returns the gate failures. Execution backends can consume the accepted lowered
plan without manually repeating the trust-boundary checks.
-/

@[expose] public section

namespace NN
namespace Backend

/-- A graph backend plan after planning, lowering, and acceptance-gate checking. -/
structure AcceptedGraphPlan where
  graphPlan : IR.GraphExecutionPlan
  loweringPlan : GraphLoweringPlan
  policy : AcceptancePolicy
  gateProof : loweringPlan.gate policy = .accepted

instance : Repr AcceptedGraphPlan where
  reprPrec p _ := Std.Format.text s!"AcceptedGraphPlan({repr p.loweringPlan})"

namespace AcceptedGraphPlan

/-- Source IR node ids covered by the accepted lowering. -/
def nodeIds (p : AcceptedGraphPlan) : List Nat :=
  p.loweringPlan.nodeIds

/-- Selected capsule names in accepted lowering order. -/
def capsuleNames (p : AcceptedGraphPlan) : List String :=
  p.loweringPlan.capsuleNames

/-- Audit for the accepted lowering. -/
def audit (p : AcceptedGraphPlan) : ExecutionAudit :=
  p.loweringPlan.audit

/-- Recheck reports for the accepted lowering. -/
def obligationReports (p : AcceptedGraphPlan) : List ObligationReport :=
  p.audit.obligationReports

end AcceptedGraphPlan

/-- Result of planning/lowering/gating a graph. -/
inductive AcceptedPlanResult where
  | accepted (plan : AcceptedGraphPlan)
  | rejected (loweringPlan : GraphLoweringPlan) (failures : List GateFailure)
  deriving Repr

namespace AcceptedPlanResult

/-- Whether the pipeline returned an accepted plan. -/
def isAccepted : AcceptedPlanResult → Bool
  | .accepted _ => true
  | .rejected .. => false

/-- Gate failures when the pipeline rejected the plan. -/
def failures : AcceptedPlanResult → List GateFailure
  | .accepted _ => []
  | .rejected _ failures => failures

end AcceptedPlanResult

/-- Gate a graph lowering and expose an accepted plan only when every obligation passes policy. -/
def acceptGraphPlan (graphPlan : IR.GraphExecutionPlan) (loweringPlan : GraphLoweringPlan)
    (policy : AcceptancePolicy) : AcceptedPlanResult :=
  match h : loweringPlan.gate policy with
  | .accepted => .accepted { graphPlan, loweringPlan, policy, gateProof := h }
  | .rejected failures => .rejected loweringPlan failures

end Backend
end NN
