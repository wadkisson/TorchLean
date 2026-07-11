/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Types

/-!
# Backend Capsules

A backend capsule is TorchLean's unit of delegation to fast code.

The capsule does not prove the foreign implementation. It records the contract TorchLean expects
from that implementation: which op it implements, which spec it refines, what layout and shape
conventions are assumed, how the value/VJP claims are justified, and what trust level the planner
must account for.
-/

@[expose] public section

namespace NN
namespace Backend

/-- A source file inside the TorchLean checkout that supports a backend contract. -/
structure SourceRef where
  path : String
  deriving DecidableEq, BEq, Repr

/--
Reference to a native/FFI symbol used by a backend capsule.

The linter checks that `path` exists, that `symbol` occurs in that source file, and that
`buildTarget?`, when present, names a Lake target in `lakefile.lean`.
-/
structure NativeSymbolRef where
  path : String
  symbol : String
  buildTarget? : Option String := none
  deriving DecidableEq, BEq, Repr

/-- Whether a selected capsule is wired to an executable runtime path. -/
inductive RuntimeSupport where
  /-- The capsule can be selected by an eager runtime path. -/
  | eager
  /-- The capsule has a direct bridge/check, but is not selected by the eager runtime yet. -/
  | testOnly
  /-- The capsule is useful for planning/auditing only. -/
  | plannerOnly
  /-- The capsule is intentionally registered before runtime wiring exists. -/
  | notWired
  deriving DecidableEq, Repr

/-- Source-level provenance for a contract descriptor. Provenance is not correctness evidence. -/
inductive ContractProvenance where
  | sourceFile (ref : SourceRef)
  | nativeSymbol (ref : NativeSymbolRef)
  | note (text : String)
  deriving DecidableEq, BEq, Repr

/-- Concrete tensor-layout convention named by a backend contract. -/
inductive TensorLayout where
  /-- TorchLean's ordinary typed tensor representation. -/
  | canonicalTensor
  /-- Contiguous flat storage with the last axis varying fastest. -/
  | flatRowMajor
  /-- A contiguous CUDA tensor view owned by LibTorch. -/
  | libTorchCudaView
  deriving DecidableEq, Repr

/-- A structured backend obligation, independent of how evidence for it is obtained. -/
inductive ContractClaim where
  /-- Inputs and outputs satisfy the shape rule associated with an operation. -/
  | shapeSafety (op : BackendOp)
  /-- Runtime buffers use the declared layout for an operation. -/
  | layoutCompatibility (op : BackendOp) (layout : TensorLayout)
  /-- Forward execution refines the named mathematical specification. -/
  | valueRefinement (op : BackendOp) (specName : String)
  /-- Local backward execution refines the named VJP specification. -/
  | vjpRefinement (op : BackendOp) (specName : String) (mode : VJPMode)
  deriving DecidableEq, Repr

/-- How a capsule justifies one part of its contract.

`runtimeGuard` and `testSuite` record useful engineering assurance but do not discharge a theorem.
Only `theorem` and `checker` contain proof terms.
-/
inductive ContractEvidence where
  | theorem (theoremName : String) (statement : Prop) (proof : statement)
  | checker (checkerName : String) (statement : Prop) (accepted : Bool)
      (sound : accepted = true -> statement) (acceptanceProof : accepted = true)
  | runtimeGuard (name : String)
  | testSuite (name : String)
  | fuzzOracle (name : String)
  | trustedBoundary (reason : String)
  | notProvided

instance : Repr ContractEvidence where
  reprPrec evidence _ := Std.Format.text <| match evidence with
    | .theorem name .. => s!"theorem({name})"
    | .checker name .. => s!"checker({name})"
    | .runtimeGuard name => s!"runtimeGuard({name})"
    | .testSuite name => s!"testSuite({name})"
    | .fuzzOracle name => s!"fuzzOracle({name})"
    | .trustedBoundary reason => s!"trustedBoundary({reason})"
    | .notProvided => "notProvided"

/-- A structured contract claim together with its evidence and human-readable explanation. -/
structure ContractDescriptor where
  claim : ContractClaim
  summary : String
  evidence : ContractEvidence
  provenance : List ContractProvenance := []

instance : Repr ContractDescriptor where
  reprPrec d _ := Std.Format.text s!"ContractDescriptor({repr d.claim}, {repr d.evidence})"

namespace ContractDescriptor

/-- A contract claim enforced by a named runtime guard. -/
def guarded (claim : ContractClaim) (summary guard : String)
    (provenance : List ContractProvenance := []) : ContractDescriptor :=
  { claim, summary, evidence := .runtimeGuard guard, provenance }

/-- A contract claim covered by a named regression suite. -/
def tested (claim : ContractClaim) (summary suite : String)
    (provenance : List ContractProvenance := []) : ContractDescriptor :=
  { claim, summary, evidence := .testSuite suite, provenance }

/-- A contract claim delegated to an explicitly named trusted boundary. -/
def trusted (claim : ContractClaim) (summary reason : String)
    (provenance : List ContractProvenance := []) : ContractDescriptor :=
  { claim, summary, evidence := .trustedBoundary reason, provenance }

end ContractDescriptor

/-- The four contract fields carried by every kernel capsule. -/
inductive ContractObligationKind where
  | shape
  | layout
  | value
  | vjp
  deriving DecidableEq, Repr

namespace ContractClaim

/-- Whether a claim has the expected kind and operation for a capsule contract field. -/
def matchesObligation (op : BackendOp) (vjpMode : VJPMode) :
    ContractObligationKind → ContractClaim → Bool
  | .shape, .shapeSafety claimOp => claimOp == op
  | .layout, .layoutCompatibility claimOp _ => claimOp == op
  | .value, .valueRefinement claimOp _ => claimOp == op
  | .vjp, .vjpRefinement claimOp _ claimMode => claimOp == op && claimMode == vjpMode
  | _, _ => false

end ContractClaim

/-- A contract-carrying fast kernel or reference implementation. -/
structure KernelCapsule where
  name : String
  op : BackendOp
  provider : Provider
  device : Device
  specName : String
  trustLevel : TrustLevel
  supportsForward : Bool := true
  vjpMode : VJPMode := .none
  runtimeSupport : RuntimeSupport := .eager
  shapeContract : ContractDescriptor
  layoutContract : ContractDescriptor
  valueContract : ContractDescriptor
  vjpContract : ContractDescriptor
  notes : String := ""
  deriving Repr

namespace KernelCapsule

/-- Whether each descriptor states the obligation advertised by its field.

Evidence is useful only when it proves or checks the right claim. This guard prevents, for example,
a value-refinement theorem from being placed in the shape field and then accepted as shape evidence.
-/
def contractsAligned (c : KernelCapsule) : Bool :=
  c.shapeContract.claim.matchesObligation c.op c.vjpMode .shape &&
  c.layoutContract.claim.matchesObligation c.op c.vjpMode .layout &&
  c.valueContract.claim.matchesObligation c.op c.vjpMode .value &&
  c.vjpContract.claim.matchesObligation c.op c.vjpMode .vjp

/-- Stable identity used when adjacent graph nodes select the same registered capsule. -/
def sameIdentity (a b : KernelCapsule) : Bool :=
  a.name == b.name && a.op == b.op && a.provider == b.provider && a.device == b.device

/-- Whether the trust policy admits this capsule. -/
def allowedBy (cfg : ExecutionConfig) (c : KernelCapsule) : Bool :=
  cfg.trustPolicy.accepts c.trustLevel

/-- Whether the backend preference admits this capsule's provider. -/
def matchesPreference (cfg : ExecutionConfig) (c : KernelCapsule) : Bool :=
  match cfg.backend with
  | .auto => true
  | .prefer _ => true
  | .only p => p = c.provider

/-- Whether this capsule is available on the selected device. -/
def matchesDevice (cfg : ExecutionConfig) (c : KernelCapsule) : Bool :=
  cfg.device = c.device

/--
Whether the capsule's gradient boundary is compatible with the requested execution config.

`none` is inference mode, so any forward-capable capsule is suitable even when it also advertises a
VJP. `externalAutograd` is intentionally opt-in. A native backend VJP is still compatible with the
normal TorchLean tape mode: the TorchLean tape owns the node and calls the backend for the local VJP.
-/
def matchesVJP (cfg : ExecutionConfig) (c : KernelCapsule) : Bool :=
  if cfg.vjpMode == .none || !c.op.requiresVJP then
    true
  else
    match cfg.vjpMode with
    | .none => true
    | .torchLeanTape => c.vjpMode == .torchLeanTape || c.vjpMode == .backendVJP
    | .backendVJP => c.vjpMode == .backendVJP
    | .externalAutograd => c.vjpMode == .externalAutograd

/-- Planner-side admissibility predicate for a single capsule. -/
def admissible (cfg : ExecutionConfig) (c : KernelCapsule) : Bool :=
  c.supportsForward && c.contractsAligned && c.allowedBy cfg && c.matchesPreference cfg &&
    c.matchesDevice cfg && c.matchesVJP cfg

end KernelCapsule

/-- Pick the first admissible capsule for a typed operation. -/
def chooseCapsuleFor? (cfg : ExecutionConfig) (op : BackendOp)
    (capsules : List KernelCapsule) : Option KernelCapsule :=
  capsules.find? fun c => c.op == op && c.admissible cfg

end Backend
end NN
