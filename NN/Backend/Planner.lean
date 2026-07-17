/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Availability

/-!
# Backend Planner

The planner is the bridge between a semantic graph and backend capsules.

At this stage it is deliberately small: given an execution config, an operation tag, and a capsule
registry, choose an admissible capsule or explain why none is available. Graph-aware layers can
recover those operation tags from `NN.IR.OpKind`, lower adjacent nodes, and eventually produce
executable command buffers without changing the contract story.
-/

@[expose] public section

namespace NN
namespace Backend

/-- A backend choice for one graph operation or fused operation. -/
structure PlannedKernel where
  op : BackendOp
  capsule : KernelCapsule
  deriving Repr

/-- A simple execution plan: one selected capsule per requested operation. -/
structure ExecutionPlan where
  kernels : List PlannedKernel
  deriving Repr

namespace ExecutionPlan

/-- Names of the selected backend capsules, useful for audits and logs. -/
def capsuleNames (p : ExecutionPlan) : List String :=
  p.kernels.map fun k => k.capsule.name

end ExecutionPlan

/-- Put preferred-provider capsules first while preserving the relative order otherwise. -/
def orderByPreference (cfg : ExecutionConfig) (capsules : List KernelCapsule) :
    List KernelCapsule :=
  match cfg.backend with
  | .prefer p =>
      capsules.filter (fun c => c.provider == p) ++
      capsules.filter (fun c => !(c.provider == p))
  | _ => capsules

/-- Choose a backend capsule for one operation. -/
def planOp (cfg : ExecutionConfig) (registry : List KernelCapsule)
    (op : BackendOp) : Except String PlannedKernel := do
  match chooseCapsuleFor? cfg op (orderByPreference cfg registry) with
  | some capsule => pure { op, capsule }
  | none =>
      throw s!"no admissible backend capsule for op {op.name} on device {cfg.device.cliName}"

/-- Choose backend capsules for a sequence of operations. -/
def planOps (cfg : ExecutionConfig) (registry : List KernelCapsule)
    (ops : List BackendOp) : Except String ExecutionPlan := do
  let kernels ← ops.mapM (planOp cfg registry)
  pure { kernels }

/-- Choose backend capsules after filtering the registry by machine/build availability. -/
def planOpsAvailable (cfg : ExecutionConfig) (availability : Availability)
    (registry : List KernelCapsule) (ops : List BackendOp) : Except String ExecutionPlan :=
  planOps cfg (availability.filterCapsules registry) ops

end Backend
end NN
