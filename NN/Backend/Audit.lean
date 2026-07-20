/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Planner

/-!
# Backend Plan Audits

Inspection data for contract-carrying backend plans.

The planner chooses capsules. The audit layer records what that choice means for trust boundaries:
which provider was selected, which device it targets, which spec it claims to refine, and whether
the plan crosses a trusted-external boundary.
-/

@[expose] public section

namespace NN
namespace Backend

namespace KernelCapsule

/-- Whether this capsule crosses a trusted external boundary. -/
def isTrustedExternal (c : KernelCapsule) : Bool :=
  c.trustLevel == .trustedExternal

end KernelCapsule

/-! ## Portable contract evidence

Backend capsules contain proof terms when a contract is discharged by a theorem or a verified
checker. Those terms belong in the process that checks a plan; they are not suitable fields for a
JSON artifact or a cache key. The snapshot types below retain the identity and provenance of the
evidence while deliberately erasing its proof term. Replanning reconstructs the full capsule and
therefore remains the authoritative correctness check.
-/

/-- Data-only identity of contract evidence, suitable for exported audit artifacts. -/
inductive ContractEvidenceSnapshot where
  | theorem (theoremName : String)
  | checker (checkerName : String)
  | runtimeGuard (name : String)
  | testSuite (name : String)
  | fuzzOracle (name : String)
  | trustedBoundary (reason : String)
  | notProvided
  deriving DecidableEq, Repr

namespace ContractEvidence

/-- Erase proof terms while preserving the evidence constructor and its stable name. -/
def snapshot : ContractEvidence -> ContractEvidenceSnapshot
  | .theorem theoremName .. => .theorem theoremName
  | .checker checkerName .. => .checker checkerName
  | .runtimeGuard name => .runtimeGuard name
  | .testSuite name => .testSuite name
  | .fuzzOracle name => .fuzzOracle name
  | .trustedBoundary reason => .trustedBoundary reason
  | .notProvided => .notProvided

end ContractEvidence

/-- Proof-free part of a contract descriptor retained in audit artifacts. -/
structure ContractDescriptorSnapshot where
  claim : ContractClaim
  summary : String
  evidence : ContractEvidenceSnapshot
  provenance : List ContractProvenance
  deriving DecidableEq, Repr

namespace ContractDescriptor

/-- Remove the proof term from a descriptor without weakening its in-process counterpart. -/
def snapshot (descriptor : ContractDescriptor) : ContractDescriptorSnapshot :=
  { claim := descriptor.claim
    summary := descriptor.summary
    evidence := descriptor.evidence.snapshot
    provenance := descriptor.provenance }

end ContractDescriptor

/-- Audit row for one selected backend kernel. -/
structure KernelAudit where
  op : BackendOp
  capsuleName : String
  provider : Provider
  device : Device
  trustLevel : TrustLevel
  vjpMode : VJPMode
  shapeContract : ContractDescriptor
  layoutContract : ContractDescriptor
  valueContract : ContractDescriptor
  vjpContract : ContractDescriptor
  numericalPolicy : NumericalPolicy
  deriving Repr

/-- Data-only audit row retained by numerical certificates and backend reports. -/
structure KernelAuditSnapshot where
  op : BackendOp
  capsuleName : String
  provider : Provider
  device : Device
  trustLevel : TrustLevel
  vjpMode : VJPMode
  shapeContract : ContractDescriptorSnapshot
  layoutContract : ContractDescriptorSnapshot
  valueContract : ContractDescriptorSnapshot
  vjpContract : ContractDescriptorSnapshot
  numericalPolicy : NumericalPolicy
  deriving DecidableEq, Repr

namespace KernelAudit

/-- Build an audit row from a selected planner kernel. -/
def ofPlannedKernel (k : PlannedKernel) : KernelAudit :=
  { op := k.op
    capsuleName := k.capsule.name
    provider := k.capsule.provider
    device := k.capsule.device
    trustLevel := k.capsule.trustLevel
    vjpMode := k.capsule.vjpMode
    shapeContract := k.capsule.shapeContract
    layoutContract := k.capsule.layoutContract
    valueContract := k.capsule.valueContract
    vjpContract := k.capsule.vjpContract
    numericalPolicy := k.capsule.numericalPolicy }

/-- Whether this selected kernel crosses a trusted external boundary. -/
def isTrustedExternal (a : KernelAudit) : Bool :=
  a.trustLevel == .trustedExternal

/-- Erase proof terms from one selected kernel audit row. -/
def snapshot (audit : KernelAudit) : KernelAuditSnapshot :=
  { op := audit.op
    capsuleName := audit.capsuleName
    provider := audit.provider
    device := audit.device
    trustLevel := audit.trustLevel
    vjpMode := audit.vjpMode
    shapeContract := audit.shapeContract.snapshot
    layoutContract := audit.layoutContract.snapshot
    valueContract := audit.valueContract.snapshot
    vjpContract := audit.vjpContract.snapshot
    numericalPolicy := audit.numericalPolicy }

end KernelAudit

/-- Audit view of an execution plan. -/
structure ExecutionAudit where
  kernels : List KernelAudit
  deriving Repr

/-- Portable audit of a complete execution plan. The proof-carrying plan is reconstructed before
this snapshot is accepted. -/
structure ExecutionAuditSnapshot where
  kernels : List KernelAuditSnapshot
  deriving DecidableEq, Repr

namespace ExecutionAuditSnapshot

/-- Capsule names in plan order. -/
def capsuleNames (audit : ExecutionAuditSnapshot) : List String :=
  audit.kernels.map (·.capsuleName)

/-- Operations whose exported audit rows name a trusted-external boundary. -/
def trustedExternalOps (audit : ExecutionAuditSnapshot) : List String :=
  (audit.kernels.filter (fun kernel => kernel.trustLevel == .trustedExternal)).map (·.op.name)

/-- Whether an exported audit crosses any trusted-external boundary. -/
def hasTrustedExternal (audit : ExecutionAuditSnapshot) : Bool :=
  audit.kernels.any (fun kernel => kernel.trustLevel == .trustedExternal)

end ExecutionAuditSnapshot

namespace ExecutionAudit

/-- Erase proof terms from a complete audit. -/
def snapshot (audit : ExecutionAudit) : ExecutionAuditSnapshot :=
  { kernels := audit.kernels.map KernelAudit.snapshot }

/-- Trust levels selected by the plan, in plan order. -/
def trustLevels (a : ExecutionAudit) : List TrustLevel :=
  a.kernels.map (·.trustLevel)

/-- Capsule names selected by the plan, in plan order. -/
def capsuleNames (a : ExecutionAudit) : List String :=
  a.kernels.map (·.capsuleName)

/-- Operation names whose selected capsule is trusted external. -/
def trustedExternalOps (a : ExecutionAudit) : List String :=
  (a.kernels.filter KernelAudit.isTrustedExternal).map (·.op.name)

/-- Whether the plan crosses any trusted external boundary. -/
def hasTrustedExternal (a : ExecutionAudit) : Bool :=
  a.kernels.any KernelAudit.isTrustedExternal

end ExecutionAudit

namespace ExecutionPlan

/-- Audit a selected execution plan. -/
def audit (p : ExecutionPlan) : ExecutionAudit :=
  { kernels := p.kernels.map KernelAudit.ofPlannedKernel }

/-- Whether a selected execution plan crosses any trusted external boundary. -/
def hasTrustedExternal (p : ExecutionPlan) : Bool :=
  p.audit.hasTrustedExternal

/-- Operation names whose selected capsules are trusted external. -/
def trustedExternalOps (p : ExecutionPlan) : List String :=
  p.audit.trustedExternalOps

end ExecutionPlan

end Backend
end NN
