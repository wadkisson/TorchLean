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

/-- Selected capsule names in accepted lowering order. -/
def capsuleNames (p : AcceptedGraphPlan) : List String :=
  p.loweringPlan.capsuleNames

end AcceptedGraphPlan

/-- Result of planning/lowering/gating a graph. -/
inductive AcceptedPlanResult where
  | accepted (plan : AcceptedGraphPlan)
  | rejected (loweringPlan : GraphLoweringPlan) (failures : List GateFailure)
  deriving Repr

/-- Gate a graph lowering and expose an accepted plan only when every obligation passes policy. -/
def acceptGraphPlan (graphPlan : IR.GraphExecutionPlan) (loweringPlan : GraphLoweringPlan)
    (policy : AcceptancePolicy) : AcceptedPlanResult :=
  match h : loweringPlan.gate policy with
  | .accepted => .accepted { graphPlan, loweringPlan, policy, gateProof := h }
  | .rejected failures => .rejected loweringPlan failures

end Backend
end NN
