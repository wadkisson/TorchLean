/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Profile

/-!
# Backend Reports

Small human-readable reports for contract-carrying backend plans.

The planner data is intentionally precise; these helpers are the user-facing explanation layer. They
are useful in examples, command-line choosers, docs, and debugging output.
-/

@[expose] public section

namespace NN
namespace Backend

namespace Provider

/-- Short stable spelling for a backend provider. -/
def label : Provider → String
  | .reference => "reference"
  | .torchLean => "torchlean"
  | .nativeCuda => "native-cuda"
  | .libTorch => "libtorch"
  | .aten => "aten"
  | .mps => "mps"
  | .webGpu => "webgpu"
  | .cuBLAS => "cublas"
  | .cuDNN => "cudnn"
  | .cuFFT => "cufft"
  | .xla => "xla"
  | .neuron => "neuron"
  | .customChip => "custom-chip"
  | .external => "external"

end Provider

namespace TrustLevel

/-- Short stable spelling for a capsule trust level. -/
def label : TrustLevel → String
  | .verified => "verified"
  | .checked => "checked"
  | .fuzzed => "fuzzed"
  | .trustedExternal => "trusted-external"

end TrustLevel

namespace TrustPolicy

/-- Short stable spelling for a profile trust policy. -/
def label : TrustPolicy → String
  | .verifiedOnly => "verified-only"
  | .checked => "checked"
  | .fuzzedOk => "fuzzed-ok"
  | .allowTrustedExternal => "trusted-external-ok"

end TrustPolicy

namespace VJPMode

/-- Short stable spelling for how a capsule handles backward/VJP. -/
def label : VJPMode → String
  | .none => "none"
  | .torchLeanTape => "torchlean-tape"
  | .backendVJP => "backend-vjp"
  | .externalAutograd => "external-autograd"

end VJPMode

namespace RuntimeSupport

/-- Short stable spelling for how a capsule is wired to runtime execution. -/
def label : RuntimeSupport → String
  | .eager => "eager"
  | .testOnly => "test-only"
  | .plannerOnly => "planner-only"
  | .notWired => "not-wired"

end RuntimeSupport

namespace TensorLayout

/-- Short stable spelling for a tensor-layout contract. -/
def label : TensorLayout → String
  | .canonicalTensor => "canonical-tensor"
  | .flatRowMajor => "flat-row-major"
  | .libTorchCudaView => "libtorch-cuda-view"

end TensorLayout

namespace ContractClaim

/-- Human-readable statement of a structured backend obligation. -/
def label : ContractClaim → String
  | .shapeSafety op => s!"shape safety for {op.name}"
  | .layoutCompatibility op layout =>
      s!"{layout.label} layout compatibility for {op.name}"
  | .valueRefinement op specName => s!"{op.name} forward refines {specName}"
  | .vjpRefinement op specName mode =>
      s!"{op.name} {mode.label} VJP refines {specName}"

end ContractClaim

namespace ContractEvidence

/-- Concise description of the evidence attached to a contract claim. -/
def label : ContractEvidence → String
  | .theorem name .. => s!"proved by {name}"
  | .checker name .. => s!"accepted by proved checker {name}"
  | .runtimeGuard name => s!"guarded at runtime by {name}"
  | .testSuite name => s!"covered by test suite {name}"
  | .fuzzOracle name => s!"compared by fuzz oracle {name}"
  | .trustedBoundary reason => s!"trusted boundary: {reason}"
  | .notProvided => "no evidence recorded"

end ContractEvidence

namespace ContractDescriptor

/-- One report line for a named contract field. -/
def reportLine (field : String) (d : ContractDescriptor) : String :=
  s!"    {field}: {d.claim.label}; {d.evidence.label}"

end ContractDescriptor

namespace KernelAudit

/-- One-line summary for a selected backend capsule. -/
def reportLine (a : KernelAudit) : String :=
  s!"  {a.op.name}: {a.capsuleName} " ++
  s!"provider={a.provider.label} trust={a.trustLevel.label} vjp={a.vjpMode.label} " ++
  s!"runtime={a.runtimeSupport.label}"

/-- Full contract report for a selected backend capsule. -/
def detailedReportLines (a : KernelAudit) : List String :=
  [ a.reportLine
  , a.shapeContract.reportLine "shape"
  , a.layoutContract.reportLine "layout"
  , a.valueContract.reportLine "value"
  , a.vjpContract.reportLine "vjp" ]

end KernelAudit

namespace ExecutionAudit

/-- Human-readable lines for all selected capsules. -/
def reportLines (a : ExecutionAudit) : List String :=
  a.kernels.map KernelAudit.reportLine

/-- Human-readable contract details for all selected capsules. -/
def detailedReportLines (a : ExecutionAudit) : List String :=
  a.kernels.flatMap KernelAudit.detailedReportLines

end ExecutionAudit

namespace ExecutionPlan

/-- Human-readable lines for a selected backend plan. -/
def reportLines (p : ExecutionPlan) : List String :=
  p.audit.reportLines

/-- Human-readable contract details for a selected backend plan. -/
def detailedReportLines (p : ExecutionPlan) : List String :=
  p.audit.detailedReportLines

end ExecutionPlan

namespace BackendProfile

/-- Representative operation set for an explicit profile preview. This is not an execution trace. -/
def representativeOps : List BackendOp :=
  [ .matmul
  , .bmm
  , .linear
  , .mseLoss
  , .add
  , .relu
  , .softmax
  , .reduceSum
  , .reshape
  , .broadcastTo
  , .gatherRowsNat
  , .layerNorm
  , .batchNormChannelFirst
  , .conv
  , .conv2d
  , .convTranspose2d
  , .maxPool
  , .maxPool2d
  , .smoothMaxPool2d
  , .avgPool
  , .avgPool2d
  , .scaledDotProductAttention
  ]

/-- One-line profile description for logs and interactive choosers. -/
def summary (p : BackendProfile) : String :=
  s!"profile={p.name} device={p.config.device.cliName} trust={p.config.trustPolicy.label} " ++
  s!"vjp={p.config.vjpMode.label}"

/-- Plan a list of backend ops and format the selected capsules. -/
def planReport (p : BackendProfile) (ops : List BackendOp) : Except String String := do
  let plan ← p.planOps ops
  let boundary :=
    if plan.hasTrustedExternal then
      "trusted external boundary: " ++ String.intercalate ", " plan.trustedExternalOps
    else
      "trusted external boundary: none"
  pure <| String.intercalate "\n" <|
    [p.summary, boundary] ++ plan.detailedReportLines

end BackendProfile

end Backend
end NN
