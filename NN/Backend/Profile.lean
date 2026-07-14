/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Accepted

/-!
# Backend Profiles

Named backend profiles bundle the choices that should move together:

- target/build availability,
- execution configuration,
- capsule registry,
- graph lowering mode,
- and acceptance policy.

This gives downstream APIs one object to pass around instead of separately threading flags such as
"CUDA", "LibTorch enabled", "trusted external allowed", and "coalesce adjacent graph nodes".
-/

@[expose] public section

namespace NN
namespace Backend

/-- Which capsule registry a profile uses. -/
inductive RegistryMode where
  | default
  | withLibTorch
  deriving DecidableEq, Repr

namespace RegistryMode

/-- Capsule registry selected by a registry mode. -/
def capsules : RegistryMode → List KernelCapsule
  | .default => Registry.defaultCapsules
  | .withLibTorch => Registry.withLibTorchCapsules

end RegistryMode

/-- How graph-node plans are lowered to backend execution groups. -/
inductive LoweringMode where
  | singleton
  | coalesced
  deriving DecidableEq, Repr

/-- One named backend execution profile. -/
structure BackendProfile where
  name : String
  config : ExecutionConfig
  target : Target
  registryMode : RegistryMode := .default
  loweringMode : LoweringMode := .coalesced
  acceptancePolicy : AcceptancePolicy := .strict
  deriving Repr

namespace BackendProfile

/-- Planner capabilities declared by the profile target; runtime execution probes separately. -/
def availability (p : BackendProfile) : Availability :=
  p.target.declaredAvailability

/-- Capsule registry selected by the profile. -/
def registry (p : BackendProfile) : List KernelCapsule :=
  p.registryMode.capsules

/-- Plan operations using the profile registry, target, and execution config. -/
def planOps (p : BackendProfile) (ops : List BackendOp) : Except String ExecutionPlan :=
  NN.Backend.planOpsAvailable p.config p.availability p.registry ops

/-- Gate a planned operation sequence under the profile's acceptance policy. -/
def gateOps (p : BackendProfile) (ops : List BackendOp) : Except String GateResult := do
  let plan ← p.planOps ops
  pure <| plan.gate p.acceptancePolicy

/-- Plan runtime-relevant IR nodes using the profile registry, target, and execution config. -/
def planGraphNodes (p : BackendProfile) (g : NN.IR.Graph) :
    Except String NN.Backend.IR.GraphExecutionPlan :=
  NN.Backend.IR.checkedPlanGraphNodesWithRegistry p.config p.availability p.registry g

/-- Lower a graph-node execution plan according to the profile's lowering mode. -/
def lowerGraphPlan (p : BackendProfile) (plan : NN.Backend.IR.GraphExecutionPlan) :
    GraphLoweringPlan :=
  match p.loweringMode with
  | .singleton => plan.toSingletonLoweringPlan
  | .coalesced => plan.toCoalescedLoweringPlan

/-- Plan, lower, and gate a graph under the profile. -/
def acceptGraph (p : BackendProfile) (g : NN.IR.Graph) :
    Except String AcceptedPlanResult := do
  let graphPlan ← p.planGraphNodes g
  let loweringPlan := p.lowerGraphPlan graphPlan
  pure <| acceptGraphPlan graphPlan loweringPlan p.acceptancePolicy

/-- Maintained portable CPU/reference profile with runtime guards and regression evidence. -/
def checkedCpu : BackendProfile :=
  { name := "checked_cpu"
    config :=
      { device := .cpu
        backend := .auto
        trustPolicy := .checked
        vjpMode := .torchLeanTape }
    target := Target.portableCpu
    registryMode := .default
    loweringMode := .coalesced
    acceptancePolicy := .checkedRuntime }

/-- Checked CPU/reference profile for a named operating system and architecture. -/
def checkedCpuTarget (os : OperatingSystem) (arch : Architecture := .unknown) : BackendProfile :=
  { checkedCpu with
    name := s!"checked_{reprStr os}_{reprStr arch}_cpu"
    target := Target.portableCpu os arch }

/-- Checked native CUDA profile. External trusted providers are not admitted. -/
def checkedCuda : BackendProfile :=
  { name := "checked_cuda"
    config :=
      { device := .cuda
        backend := .auto
        trustPolicy := .checked
        vjpMode := .torchLeanTape }
    target := Target.linuxCuda false
    registryMode := .default
    loweringMode := .coalesced
    acceptancePolicy := .checkedRuntime }

/--
LibTorch forward scaling profile.

LibTorch is allowed to provide selected forward values, but TorchLean still records the graph/tape
boundary and does not hand local backward ownership to LibTorch autograd.
-/
def libTorchForwardCuda : BackendProfile :=
  { name := "libtorch_forward_cuda"
    config :=
      { device := .cuda
        backend := .only .libTorch
        trustPolicy := .allowTrustedExternal
        vjpMode := .torchLeanTape }
    target := Target.linuxCuda true
    registryMode := .withLibTorch
    loweringMode := .coalesced
    acceptancePolicy := .allowTrustedRuntime }

/-- Explicit LibTorch autograd scaling profile. Trusted external backward boundaries are visible. -/
def libTorchAutogradCuda : BackendProfile :=
  { name := "libtorch_autograd_cuda"
    config :=
      { device := .cuda
        backend := .only .libTorch
        trustPolicy := .allowTrustedExternal
        vjpMode := .externalAutograd }
    target := Target.linuxCuda true
    registryMode := .withLibTorch
    loweringMode := .coalesced
    acceptancePolicy := .allowTrustedRuntime }

/-- Checked macOS CPU/reference profile. -/
def macOSCpu : BackendProfile :=
  checkedCpuTarget .macOS .aarch64

/-- Checked Windows CPU/reference profile. Used for CI bring-up before native Windows kernels exist. -/
def windowsCpu : BackendProfile :=
  checkedCpuTarget .windows .x86_64

/-- Construct a named target whose runtime capsules are not implemented yet. -/
def future
    (name : String)
    (device : Device)
    (target : Target)
    (backend : BackendPreference := .auto)
    (trustPolicy : TrustPolicy := .checked)
    (vjpMode : VJPMode := .torchLeanTape)
    (acceptancePolicy : AcceptancePolicy := .strict) : BackendProfile :=
  { name
    config := { device, backend, trustPolicy, vjpMode }
    target
    registryMode := .default
    loweringMode := .coalesced
    acceptancePolicy }

/--
Future Metal/MPS profile.

The target and device are named now, but the default registry has no Metal capsules yet. Planning a
non-reference op under this profile therefore fails with a missing-capsule error instead of falling
back silently to CUDA or CPU.
-/
def futureMetal : BackendProfile :=
  future "future_metal" .metal Target.macOSMetal

/-- Future ROCm profile. It is a named planning target until HIP/ROCm capsules are added. -/
def futureRocm : BackendProfile :=
  future "future_rocm" .rocm Target.linuxRocm

/-- Future WebGPU/WASM profile. It is a named planning target until WebGPU capsules are added. -/
def futureWasm : BackendProfile :=
  future "future_wasm" .wasm Target.wasm

/-- Future TPU/XLA profile. It is a named planning target until TPU capsules are added. -/
def futureTpu : BackendProfile :=
  future "future_tpu" .tpu Target.linuxTpu

/-- Future AWS Trainium/Neuron profile. It is a named planning target until Neuron capsules exist. -/
def futureTrainium : BackendProfile :=
  future "future_trainium" .trainium Target.linuxTrainium

/-- Future first-party/lab custom accelerator profile. -/
def futureCustomChip : BackendProfile :=
  future "future_custom_chip" .custom Target.customChip

/-- Future caller-supplied external accelerator profile. -/
def futureExternal : BackendProfile :=
  future "future_external" .external Target.external
    (.only .external) .allowTrustedExternal .externalAutograd .allowTrustedRuntime

end BackendProfile

end Backend
end NN
