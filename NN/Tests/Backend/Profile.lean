/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Report
public import NN.Backend.IR
public import NN.Runtime.Autograd.Torch.Core.Ops

/-!
# Backend Profile Tests

Regression checks for contract-carrying backend profiles.

These are policy checks, not numerical kernel tests: they make sure backend planning does not
silently cross a trusted boundary or fall back to an unavailable platform provider.
-/

@[expose] public section

namespace NN.Tests.Backend.Profile

open NN.Backend

def expect (tag : String) (ok : Bool) : IO Unit := do
  unless ok do
    throw <| IO.userError s!"backend profile check failed: {tag}"

def expectCapsules (tag : String) (got expected : List String) : IO Unit := do
  expect tag (got == expected)

def expectOp (tag : String) (kind : NN.IR.OpKind) (expected : Option BackendOp) : IO Unit := do
  expect tag (NN.Backend.IR.op? kind == expected)

def expectContains (tag needle haystack : String) : IO Unit := do
  expect tag (haystack.contains needle)

def tinyReluGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0
        parents := []
        kind := .input
        outShape := Spec.Shape.scalar },
      { id := 1
        parents := [0]
        kind := .relu
        outShape := Spec.Shape.scalar },
      { id := 2
        parents := [1]
        kind := .relu
        outShape := Spec.Shape.scalar }
    ] }

def externalReluCapsule : KernelCapsule :=
  { Reference.relu with
    name := "external.relu"
    provider := .external
    device := .external
    trustLevel := .checked
    notes := "Test-only external capsule used to check target gating." }

def disabledForwardReluCapsule : KernelCapsule :=
  { Reference.relu with
    name := "reference.relu_disabled"
    supportsForward := false
    notes := "Test-only capsule used to check planner forward support gating." }

def fuzzedReluCapsule : KernelCapsule :=
  { Reference.relu with
    name := "reference.relu_fuzzed"
    valueContract :=
      { claim := .valueRefinement .relu "test-only ReLU value contract"
        summary := "Test-only fuzz-backed value contract."
        evidence := .fuzzOracle "profile-test-fuzz-oracle" }
    notes := "Test-only capsule used to check strict policy rejection of fuzz-backed evidence." }

def provedDescriptor (claim : ContractClaim) : ContractDescriptor :=
  { claim
    summary := "Test proposition."
    evidence := .theorem "True.intro" True True.intro }

def provedReluCapsule : KernelCapsule :=
  { Reference.relu with
    name := "proved.relu"
    trustLevel := .verified
    shapeContract := provedDescriptor (.shapeSafety .relu)
    layoutContract := provedDescriptor (.layoutCompatibility .relu .canonicalTensor)
    valueContract := provedDescriptor (.valueRefinement .relu "test proposition")
    vjpContract := provedDescriptor (.vjpRefinement .relu "test proposition" .torchLeanTape) }

def malformedProvedReluCapsule : KernelCapsule :=
  { provedReluCapsule with
    name := "proved.relu_malformed"
    shapeContract := provedDescriptor (.valueRefinement .relu "wrong obligation kind") }

def forwardOnlyReluCapsule : KernelCapsule :=
  { Reference.relu with
    name := "reference.relu_forward_only"
    vjpMode := .none
    vjpContract := provedDescriptor (.vjpRefinement .relu "no VJP" .none) }

def planOrThrow (tag : String) (profile : BackendProfile) (ops : List BackendOp) :
    IO ExecutionPlan := do
  match profile.planOps ops with
  | .ok plan => pure plan
  | .error msg => throw <| IO.userError s!"{tag}: planning failed: {msg}"

def expectPlanningFails (tag : String) (profile : BackendProfile) (ops : List BackendOp) :
    IO Unit := do
  match profile.planOps ops with
  | .ok plan =>
      throw <| IO.userError
        s!"{tag}: expected planning to fail, got capsules {plan.capsuleNames}"
  | .error _ => pure ()

def acceptedGraphOrThrow (tag : String) (profile : BackendProfile) (g : NN.IR.Graph) :
    IO AcceptedGraphPlan := do
  match profile.acceptGraph g with
  | .ok (.accepted plan) => pure plan
  | .ok (.rejected _ failures) =>
      throw <| IO.userError s!"{tag}: expected accepted graph, got gate failures {repr failures}"
  | .error msg =>
      throw <| IO.userError s!"{tag}: graph planning failed: {msg}"

def expectGraphRejected (tag : String) (profile : BackendProfile) (g : NN.IR.Graph) :
    IO Unit := do
  match profile.acceptGraph g with
  | .ok (.accepted plan) =>
      throw <| IO.userError s!"{tag}: expected rejected graph, got capsules {plan.capsuleNames}"
  | .ok (.rejected _ _) => pure ()
  | .error msg =>
      throw <| IO.userError s!"{tag}: graph planning failed before gate: {msg}"

def expectNativeCudaGuardAccepts (_tag : String)
    (opts : Runtime.Autograd.Torch.Options) (op : BackendOp) : IO Unit := do
  let base ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float)
  let s := { base with opts := opts }
  Runtime.Autograd.Torch.Internal.EagerSession.requireNativeCudaCapsule s op

def expectNativeCudaGuardRejects (tag : String)
    (opts : Runtime.Autograd.Torch.Options) (op : BackendOp) : IO Unit := do
  let base ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float)
  let s := { base with opts := opts }
  let rejected ← try
    Runtime.Autograd.Torch.Internal.EagerSession.requireNativeCudaCapsule s op
    pure false
  catch _ =>
    pure true
  expect tag rejected

def expectRuntimeDeviceRejected (tag : String) (device : NN.Backend.Device) :
    IO Unit := do
  try
    let _ ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float)
      { device := device }
    throw <| IO.userError s!"{tag}: expected runtime device rejection"
  catch e =>
    let msg := toString e
    expectContains tag s!"device `{device.cliName}` is a named TorchLean target" msg

/-- User-facing CUDA sessions must agree with the implementation linked behind the CUDA symbols. -/
def expectCudaSessionMatchesRuntime : IO Unit := do
  let opts : Runtime.Autograd.Torch.Options := { device := .cuda }
  match Runtime.Autograd.Cuda.Buffer.runtimeStatus with
  | .nativeAvailable =>
      let _ ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float) opts
      pure ()
  | .cpuStub =>
      try
        let _ ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float) opts
        throw <| IO.userError "CPU-stub build unexpectedly admitted a user CUDA session"
      catch e =>
        expectContains "CPU-stub CUDA session rejection" "CPU parity stubs" e.toString
  | .nativeUnavailable =>
      try
        let _ ← Runtime.Autograd.Torch.Internal.EagerSession.new (α := Float) opts
        throw <| IO.userError "CUDA build without a visible device unexpectedly admitted a session"
      catch e =>
        expectContains "unavailable native CUDA session rejection" "no usable CUDA device" e.toString

def run : IO Unit := do
  expect "default registry contract fields are aligned"
    (Registry.defaultCapsules.all KernelCapsule.contractsAligned)
  expect "LibTorch registry contract fields are aligned"
    (Registry.withLibTorchCapsules.all KernelCapsule.contractsAligned)
  let inferenceOpts : Runtime.Autograd.Torch.Options := { trackGradients := false }
  expect "no-grad runtime planning requests no VJP"
    (inferenceOpts.backendProfile.config.vjpMode == .none)
  let trainingOpts : Runtime.Autograd.Torch.Options := { trackGradients := true }
  expect "training runtime planning requests the TorchLean tape"
    (trainingOpts.backendProfile.config.vjpMode == .torchLeanTape)
  expectOp "IR add maps to exact add capsule" .add (some .add)
  expectOp "IR linear maps to exact linear capsule" .linear (some .linear)
  expectOp "IR conv2d maps to exact conv2d capsule"
    (.conv2d 1 1 3 3 1 0) (some .conv2d)
  expectOp "IR maxPool2d maps to exact max_pool2d capsule"
    (.maxPool2d 2 2 2) (some .maxPool2d)
  expectOp "IR rand uniform maps to exact forward-only capsule"
    (.randUniform 0) (some .randUniform)
  expectOp "IR permute maps to exact permute capsule"
    (.permute [1, 0]) (some .permute)
  expectOp "IR input has no backend capsule" .input none

  expect "cpu availability rejects external device capsule"
    (!(Availability.cpu.admitsCapsule externalReluCapsule))
  expect "external target admits external capsule when provider is available"
    (Target.external.declaredAvailability.admitsCapsule externalReluCapsule)
  match planOpsAvailable
      { device := .external, backend := .only .external }
      Target.external.declaredAvailability
      [externalReluCapsule]
      [.relu] with
  | .ok plan =>
      expectCapsules "external target can plan explicit external capsule" plan.capsuleNames
        ["external.relu"]
  | .error msg =>
      throw <| IO.userError s!"external capsule planning failed: {msg}"
  match planOps { device := .cpu } [disabledForwardReluCapsule] [.relu] with
  | .ok plan =>
      throw <| IO.userError
        s!"disabled forward capsule unexpectedly planned as {plan.capsuleNames}"
  | .error _ => pure ()
  match planOps
      { device := .cpu, vjpMode := .none }
      [Reference.relu] [.relu] with
  | .ok inferencePlan =>
      expectCapsules "inference accepts a forward capsule that also supports VJP"
        inferencePlan.capsuleNames ["reference.relu"]
  | .error msg =>
      throw <| IO.userError s!"inference VJP compatibility test: planning failed: {msg}"
  match planOps { device := .cpu, vjpMode := .torchLeanTape }
      [forwardOnlyReluCapsule] [.relu] with
  | .ok forwardOnly =>
      throw <| IO.userError
        s!"forward-only differentiable capsule unexpectedly planned as {forwardOnly.capsuleNames}"
  | .error _ => pure ()
  match planOps { device := .cpu, trustPolicy := .verifiedOnly }
      [malformedProvedReluCapsule] [.relu] with
  | .ok malformed =>
      throw <| IO.userError
        s!"malformed capsule unexpectedly planned as {malformed.capsuleNames}"
  | .error _ => pure ()
  match planOps { device := .cpu } [fuzzedReluCapsule] [.relu] with
  | .ok fuzzedRelu =>
      expect "strict gate rejects fuzz-only evidence"
        (!fuzzedRelu.acceptedBy AcceptancePolicy.strict)
      expect "runtime gate accepts fuzz-backed checked evidence"
        (fuzzedRelu.acceptedBy AcceptancePolicy.allowTrustedRuntime)
  | .error msg =>
      throw <| IO.userError s!"fuzzed relu policy test: planning failed: {msg}"

  match planOps
      { device := .cpu, trustPolicy := .verifiedOnly }
      [provedReluCapsule] [.relu] with
  | .ok provedRelu =>
      expect "strict gate accepts proof-bearing evidence"
        (provedRelu.acceptedBy AcceptancePolicy.strict)
  | .error msg =>
      throw <| IO.userError s!"proved relu policy test: planning failed: {msg}"

  let acceptedCpuGraph ← acceptedGraphOrThrow "checked cpu graph acceptance"
    BackendProfile.checkedCpu tinyReluGraph
  expectCapsules "checked cpu graph coalesces same relu capsule"
    acceptedCpuGraph.capsuleNames ["reference.relu"]
  expectCapsules "checked cpu graph keeps source node ids"
    (acceptedCpuGraph.nodeIds.map (fun n => toString n)) ["1", "2"]
  expect "checked cpu graph has no missing evidence" acceptedCpuGraph.audit.hasNoMissingEvidence
  expect "checked cpu graph has no trusted external" (!acceptedCpuGraph.audit.hasTrustedExternal)

  let singletonCpuProfile : BackendProfile :=
    { BackendProfile.checkedCpu with
      name := "checked_cpu_singleton_test"
      loweringMode := .singleton }
  let singletonCpuGraph ← acceptedGraphOrThrow "checked cpu singleton graph acceptance"
    singletonCpuProfile tinyReluGraph
  expectCapsules "singleton graph keeps repeated relu capsules"
    singletonCpuGraph.capsuleNames ["reference.relu", "reference.relu"]

  let strictLibTorchProfile : BackendProfile :=
    { BackendProfile.libTorchForwardCuda with
      name := "libtorch_forward_strict_gate_test"
      acceptancePolicy := .strict }
  let softmaxGraph : NN.IR.Graph :=
    { nodes := #[
        { id := 0
          parents := []
          kind := .input
          outShape := Spec.Shape.scalar },
        { id := 1
          parents := [0]
          kind := .softmax 0
          outShape := Spec.Shape.scalar }
      ] }
  -- `softmax` has no LibTorch-only capsule in this registry, so LibTorch-only planning should fail
  -- before the acceptance gate rather than silently falling back.
  match strictLibTorchProfile.acceptGraph softmaxGraph with
  | .ok (.accepted plan) =>
      throw <| IO.userError
        s!"strict libtorch graph unexpectedly accepted capsules {plan.capsuleNames}"
  | .ok (.rejected _ _) =>
      throw <| IO.userError "strict libtorch graph should fail planning for missing capsule"
  | .error _ => pure ()

  let exactOps :=
    [ BackendOp.matmul, .bmm, .linear, .mseLoss, .add, .sub, .mul, .scale, .abs, .sqrt
    , .clamp, .max, .min, .relu, .gelu, .sigmoid, .tanh
    , .softmax, .softplus, .exp, .log, .inv, .safeLog, .logSoftmax, .sum, .reduceSum
    , .reduceMean, .randUniform, .bernoulliMask, .flatten, .reshape, .permute
    , .transpose2d, .swapAdjacentAtDepth
    , .transpose3dFirstToLast, .transpose3dLastToFirst, .transpose3dLastTwo
    , .broadcastTo, .concatVectors, .concatLeadingAxis, .sliceLeadingAxisRange
    , .gatherScalar, .gatherScalarNat, .gatherRow, .gatherVecNat, .gatherRowsNat
    , .scatterAddVec, .scatterAddRow, .layerNorm, .batchNorm, .batchNormChannelFirst
    , .conv
    , .conv2d
    , .convTranspose, .convTranspose2d, .maxPool2d, .maxPool2dPad, .smoothMaxPool
    , .smoothMaxPool2d, .maxPool, .avgPool, .avgPool2d, .avgPool2dPad ]

  let exactReferenceCapsules := exactOps.map fun op => s!"reference.{op.name}"
  let exactNativeCudaCapsules := exactOps.map fun op => s!"native_cuda.{op.name}"
  let cpuOnlyOps := [BackendOp.sin, .cos]

  let cpu ← planOrThrow "checked cpu" BackendProfile.checkedCpu
    [ .matmul, .bmm, .relu, .softmax, .layerNorm, .batchNormChannelFirst
    , .conv, .convTranspose2d, .maxPool2d, .smoothMaxPool2d, .avgPool2d, .mseLoss
    , .scaledDotProductAttention ]
  expectCapsules "checked cpu capsule order" cpu.capsuleNames
    [ "reference.matmul"
    , "reference.bmm"
    , "reference.relu"
    , "reference.softmax"
    , "reference.layer_norm"
    , "reference.batchnorm_channel_first"
    , "reference.conv"
    , "reference.conv_transpose2d"
    , "reference.max_pool2d"
    , "reference.smooth_max_pool2d"
    , "reference.avg_pool2d"
    , "reference.mse_loss"
    , "reference.attention"
    ]
  expect "checked cpu has no trusted external" (!cpu.hasTrustedExternal)

  let cpuExact ← planOrThrow "checked cpu exact ops" BackendProfile.checkedCpu exactOps
  expectCapsules "checked cpu exact capsules" cpuExact.capsuleNames exactReferenceCapsules
  let cpuOnly ← planOrThrow "checked cpu cpu-only ops" BackendProfile.checkedCpu cpuOnlyOps
  expectCapsules "checked cpu cpu-only capsules" cpuOnly.capsuleNames
    ["reference.sin", "reference.cos"]
  match BackendProfile.checkedCpu.planReport BackendProfile.representativeOps with
  | .ok report =>
      expectContains "checked cpu report names exact add" "add: reference.add" report
      expectContains "checked cpu report names exact reshape" "reshape: reference.reshape" report
      expectContains "checked cpu report names exact bmm" "bmm: reference.bmm" report
      expectContains "checked cpu report names exact batchnorm"
        "batchnorm_channel_first: reference.batchnorm_channel_first" report
      expectContains "checked cpu report names exact smooth max pool"
        "smooth_max_pool2d: reference.smooth_max_pool2d" report
      expectContains "checked cpu report names exact attention"
        "scaled_dot_product_attention: reference.attention" report
  | .error msg =>
      throw <| IO.userError s!"checked cpu report failed: {msg}"

  let cuda ← planOrThrow "checked cuda" BackendProfile.checkedCuda
    [ .matmul, .bmm, .relu, .softmax, .layerNorm, .batchNormChannelFirst
    , .conv, .convTranspose2d, .maxPool2d, .smoothMaxPool2d, .avgPool2d, .mseLoss
    , .scaledDotProductAttention ]
  expectCapsules "checked cuda capsule order" cuda.capsuleNames
    [ "native_cuda.matmul"
    , "native_cuda.bmm"
    , "native_cuda.relu"
    , "native_cuda.softmax"
    , "native_cuda.layer_norm"
    , "native_cuda.batchnorm_channel_first"
    , "native_cuda.conv"
    , "native_cuda.conv_transpose2d"
    , "native_cuda.max_pool2d"
    , "native_cuda.smooth_max_pool2d"
    , "native_cuda.avg_pool2d"
    , "native_cuda.mse_loss"
    , "native_cuda.flash_attention"
    ]
  expect "checked cuda has no trusted external" (!cuda.hasTrustedExternal)

  let cudaExact ← planOrThrow "checked cuda exact ops" BackendProfile.checkedCuda exactOps
  expectCapsules "checked cuda exact capsules" cudaExact.capsuleNames exactNativeCudaCapsules
  expectPlanningFails "checked cuda has no sin capsule yet" BackendProfile.checkedCuda [.sin]
  match BackendProfile.checkedCuda.planReport BackendProfile.representativeOps with
  | .ok report =>
      expectContains "checked cuda report names exact add" "add: native_cuda.add" report
      expectContains "checked cuda report names exact max pool" "max_pool: native_cuda.max_pool" report
      expectContains "checked cuda report names exact bmm" "bmm: native_cuda.bmm" report
      expectContains "checked cuda report names exact batchnorm"
        "batchnorm_channel_first: native_cuda.batchnorm_channel_first" report
      expectContains "checked cuda report names exact smooth max pool"
        "smooth_max_pool2d: native_cuda.smooth_max_pool2d" report
      expectContains "checked cuda report names exact attention"
        "scaled_dot_product_attention: native_cuda.flash_attention" report
  | .error msg =>
      throw <| IO.userError s!"checked cuda report failed: {msg}"

  let libtorchForward ← planOrThrow "libtorch forward cuda" BackendProfile.libTorchForwardCuda
    [.scaledDotProductAttention]
  expectCapsules "libtorch forward capsule order" libtorchForward.capsuleNames
    ["libtorch.sdpa_forward"]
  expect "libtorch forward records external boundary" libtorchForward.hasTrustedExternal
  expect "libtorch forward is eager runtime support"
    (libtorchForward.audit.kernels.map (·.runtimeSupport) == [.eager])
  expectCapsules "libtorch forward external op" libtorchForward.trustedExternalOps
    ["scaled_dot_product_attention"]
  expect "strict gate rejects trusted LibTorch forward"
    (!libtorchForward.acceptedBy AcceptancePolicy.strict)
  match BackendProfile.libTorchForwardCuda.planReport [.scaledDotProductAttention] with
  | .ok report =>
      expectContains "libtorch forward report names TorchLean tape"
        "vjp=torchlean-tape" report
      expectContains "libtorch forward report names capsule"
        "scaled_dot_product_attention: libtorch.sdpa_forward" report
      expectContains "libtorch forward report names runtime support"
        "runtime=eager" report
  | .error msg =>
      throw <| IO.userError s!"libtorch forward report failed: {msg}"

  let libtorchAutograd ← planOrThrow "libtorch autograd cuda" BackendProfile.libTorchAutogradCuda
    [.scaledDotProductAttention]
  expectCapsules "libtorch autograd capsule order" libtorchAutograd.capsuleNames
    ["libtorch.sdpa_autograd"]
  expect "libtorch autograd records external boundary" libtorchAutograd.hasTrustedExternal
  expect "libtorch autograd is eager runtime support"
    (libtorchAutograd.audit.kernels.map (·.runtimeSupport) == [.eager])
  expectCapsules "libtorch autograd external op" libtorchAutograd.trustedExternalOps
    ["scaled_dot_product_attention"]
  expect "strict gate rejects trusted LibTorch autograd"
    (!libtorchAutograd.acceptedBy AcceptancePolicy.strict)
  match BackendProfile.libTorchAutogradCuda.planReport [.scaledDotProductAttention] with
  | .ok report =>
      expectContains "libtorch autograd report names external autograd"
        "vjp=external-autograd" report
      expectContains "libtorch autograd report names capsule"
        "scaled_dot_product_attention: libtorch.sdpa_autograd" report
      expectContains "libtorch autograd report names runtime support"
        "runtime=eager" report
  | .error msg =>
      throw <| IO.userError s!"libtorch autograd report failed: {msg}"

  expectPlanningFails "future metal has no matmul capsule yet" BackendProfile.futureMetal [.matmul]
  expectPlanningFails "future rocm has no matmul capsule yet" BackendProfile.futureRocm [.matmul]
  expectPlanningFails "future wasm has no matmul capsule yet" BackendProfile.futureWasm [.matmul]
  expectPlanningFails "future tpu has no matmul capsule yet" BackendProfile.futureTpu [.matmul]
  expectPlanningFails "future trainium has no matmul capsule yet"
    BackendProfile.futureTrainium [.matmul]
  expectPlanningFails "future custom chip has no matmul capsule yet"
    BackendProfile.futureCustomChip [.matmul]
  expectPlanningFails "future external has no matmul capsule yet"
    BackendProfile.futureExternal [.matmul]

  for (tag, device) in
      [ ("runtime rejects named-but-unimplemented metal", NN.Backend.Device.metal)
      , ("runtime rejects named-but-unimplemented rocm", NN.Backend.Device.rocm)
      , ("runtime rejects named-but-unimplemented wasm", NN.Backend.Device.wasm)
      , ("runtime rejects named-but-unimplemented tpu", NN.Backend.Device.tpu)
      , ("runtime rejects named-but-unimplemented trainium", NN.Backend.Device.trainium)
      , ("runtime rejects named-but-unimplemented custom chip", NN.Backend.Device.custom)
      , ("runtime rejects named-but-unimplemented external", NN.Backend.Device.external)
      ] do
    expectRuntimeDeviceRejected tag device

  expectCudaSessionMatchesRuntime

  let checkedCudaOpts : Runtime.Autograd.Torch.Options :=
    { device := .cuda
      backendProfile? := some BackendProfile.checkedCuda }

  let mismatchedCudaOpts : Runtime.Autograd.Torch.Options :=
    { device := .cuda
      backendProfile? := some BackendProfile.checkedCpu }
  for op in
      [ BackendOp.matmul
      , .bmm
      , .batchNormChannelFirst
      , .maxPool
      , .avgPool
      , .smoothMaxPool
      , .maxPool2d
      , .maxPool2dPad
      , .smoothMaxPool2d
      , .avgPool2d
      , .avgPool2dPad
      ] do
    expectNativeCudaGuardAccepts s!"checked cuda runtime guard accepts `{op.name}`"
      checkedCudaOpts op
    expectNativeCudaGuardRejects s!"reference profile on cuda runtime rejects `{op.name}`"
      mismatchedCudaOpts op

  IO.println "  backend profiles: ok"

end NN.Tests.Backend.Profile
