/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Compiled.Core
public import NN.Runtime.Autograd.Compiled.GraphM
public import NN.Runtime.Autograd.Engine.Cuda.Tape
public import NN.Runtime.Autograd.Engine.Cuda.Kernels
public import NN.Runtime.Autograd.Engine.Cuda.ConvPool
public import NN.Runtime.Autograd.Engine.Cuda.Ops
public import NN.Runtime.Autograd.Engine.Cuda.Shape
public import NN.Backend.Profile
import Batteries.Data.Vector.Lemmas
import Init.Data.Vector.Lemmas
import Mathlib.Algebra.Order.Algebra

/-!
# Torch Runtime Types

Public handles and options for the Torch-style front-end. This file contains no eager
operation implementations; it defines the objects the other runtime modules share.
-/


@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec
open Tensor
open Proofs.Autograd.Algebra

/--
`TList` is the dependently-typed heterogeneous tensor list used by the proved IR.

We re-export it here under the `Torch` front-end namespace because many user-facing helpers
(trainer APIs, parameter packs, etc.) are naturally expressed as `TList`s.
-/
abbrev TList (α : Type) (ss : List Shape) := Proofs.Autograd.Algebra.TList α ss

/--
Execution backend for the Torch-style front-end.

- `.eager`: build and execute a runtime tape directly (imperative, PyTorch-like).
- `.compiled`: record typed IR and run a compiled tape (proof-friendly path, see
  `Torch.LinkedSession` / `TorchLean.Session`).

This is not a CUDA Graph selector. CUDA is controlled by `Options.device` on the
eager backend; CUDA Graph capture/replay will require a distinct persistent-buffer backend.
-/
inductive Backend where
  | eager
  | compiled
deriving Repr, DecidableEq

/--
Options controlling the behavior of the Torch-style front-end.

PyTorch comparison: these are approximately session/global settings, such as the default
`requires_grad` value and requested execution device.
-/
structure Options where
  /-- Execution backend selection. -/
  backend : Backend := .eager
  /-- Default `requires_grad` value for newly created parameters/inputs when a caller omits it. -/
  requiresGradByDefault : Bool := true
  /--
  Global deterministic seed for runtime randomness.

  TorchLean keeps the semantic core pure and seed-threaded (JAX-style), so this value is the
  runtime seed that user code can thread into:
  - model initialization (per-layer init keys),
  - dataset shuffles / sampling,
  - and session-level RNG state (dropout, etc.).

  PyTorch analogue: `torch.manual_seed(seed)`.
  -/
  seed : Nat := 0
  /--
  Track gradients for newly recorded leaves.

  Inference helpers set this to `false`: forward values are still materialized so they can be read
  back, but parameter/input leaves are recorded with `requires_grad = false`. The tape may still
  exist as a runtime value store; this flag controls whether newly recorded leaves participate in
  backward, not whether forward execution is allowed to allocate intermediate values.
  -/
  trackGradients : Bool := true
  /-- Explicit eager execution device. The CLI resolves `auto` before constructing this record. -/
  device : NN.Backend.Device := .cpu
  /--
  Optional backend-contract profile selected by higher-level helpers.

  If omitted, `Options.backendProfile` derives a conservative profile from `device`.
  -/
  backendProfile? : Option NN.Backend.BackendProfile := none
  /-- Print each accepted backend capsule the first time an eager session executes it. -/
  showBackend : Bool := false
deriving Repr

/- Convenience API for PyTorch-style device selection. -/
namespace Options

/-- Return a copy of the options that selects the requested execution device. -/
def toDevice (opts : Options) (device : NN.Backend.Device) : Options :=
  { opts with device, backendProfile? := none }

/-- Whether the current eager runtime implements an explicit backend device. -/
def runtimeImplementsDevice : NN.Backend.Device → Bool
  | .cpu | .cuda => true
  | .rocm | .metal | .wasm | .tpu | .trainium | .custom | .external => false

/-- Explain why an explicit backend target cannot execute in the current eager runtime. -/
def unsupportedDeviceMessage (device : NN.Backend.Device) : String :=
  s!"device `{device.cliName}` is a named TorchLean target, but this runtime build only implements cpu and cuda execution"

/-- Whether the effective runtime device is CUDA. -/
def usesCuda (opts : Options) : Bool :=
  opts.device == .cuda

/-- Reject devices that are named for planning but not implemented by the eager runtime yet. -/
def validateDevice (opts : Options) : Except String Unit :=
  if runtimeImplementsDevice opts.device then
    pure ()
  else
    throw <| unsupportedDeviceMessage opts.device

/-- Validate both the named device and the linked native runtime before executing user code. -/
def validateForExecution (opts : Options) : IO Unit := do
  match opts.validateDevice with
  | .ok () => pure ()
  | .error msg => throw <| IO.userError msg
  if opts.usesCuda then
    Runtime.Autograd.Cuda.Buffer.requireNativeRuntime

/-- CLI/log spelling for the effective runtime device. -/
def deviceName (opts : Options) : String :=
  opts.device.cliName

/-- Conservative backend-contract profile implied by the selected runtime device. -/
def defaultBackendProfile (opts : Options) : NN.Backend.BackendProfile :=
  match opts.device with
  | .cpu => NN.Backend.BackendProfile.checkedCpu
  | .cuda => NN.Backend.BackendProfile.checkedCuda
  | .rocm => NN.Backend.BackendProfile.futureRocm
  | .metal => NN.Backend.BackendProfile.futureMetal
  | .wasm => NN.Backend.BackendProfile.futureWasm
  | .tpu => NN.Backend.BackendProfile.futureTpu
  | .trainium => NN.Backend.BackendProfile.futureTrainium
  | .custom => NN.Backend.BackendProfile.futureCustomChip
  | .external => NN.Backend.BackendProfile.futureExternal

/-- Backend-contract profile attached to this runtime options record. -/
def backendProfile (opts : Options) : NN.Backend.BackendProfile :=
  let profile := match opts.backendProfile? with
  | some p => p
  | none => opts.defaultBackendProfile
  if opts.trackGradients then
    profile
  else
    { profile with config := { profile.config with vjpMode := .none } }

/-- Select a backend capsule for one operation under this options record. -/
def planBackendOp (opts : Options) (op : NN.Backend.BackendOp) :
    Except String NN.Backend.AcceptedKernel :=
  match opts.backendProfile.planOps [op] with
  | .ok { kernels := k :: _ } =>
      match k.accept opts.backendProfile.acceptancePolicy with
      | .error failures =>
          .error s!"backend profile {opts.backendProfile.name} rejected `{op.name}`: {repr failures}"
      | .ok accepted =>
          if k.capsule.runtimeSupport == .eager then
            .ok accepted
          else
            .error <|
              s!"backend capsule `{k.capsule.name}` for `{op.name}` is " ++
                s!"{repr k.capsule.runtimeSupport}, not eager runtime support"
  | .ok { kernels := [] } =>
      .error s!"backend profile {opts.backendProfile.name} returned no capsule for {op.name}"
  | .error msg => .error msg

end Options

/--
Opaque handle to a tensor value in the current session/tape.

Like a PyTorch tensor handle, the value is identified by a node or leaf id in the autograd tape.
The phantom shape index `s` makes shape mismatches explicit at compile time.
-/
structure TensorRef (α : Type) (s : Shape) where
  /-- Node/leaf identifier in the owning session tape. -/
  id : Nat
deriving Repr

/--
Handle to a `Nat` stored in the session's non-differentiable environment.

This is used to model index-like inputs (class labels, gather indices, etc.) which should not
receive gradients.
-/
structure NatRef where
  /-- Index into the session's non-differentiable `Nat` environment. -/
  id : Nat
deriving Repr

/--
Handle to a contiguous block of `k` `Nat`s in the session's non-differentiable environment.
-/
structure NatVecRef (k : Nat) where
  /-- Start offset of the contiguous `k`-element block in the session's `Nat` environment. -/
  start : Nat
deriving Repr

/--
Trainable parameter: a mutable tensor value plus metadata.

PyTorch comparison: analogous to `torch.nn.Parameter`, except the parameter becomes part of the
autograd graph only when you `use` it in a particular session/tape.
-/
structure Param (α : Type) (s : Shape) where
  /-- Optional user-facing name for logging/debugging. -/
  name : Option String := none
  /-- Value at the current point. -/
  value : IO.Ref (Tensor α s)
  /--
  Optional CUDA-resident mirror of `value`.

  The eager CUDA trainer uses this as a persistent-parameter cache: repeated forward
  passes can reuse the device buffer instead of uploading the host tensor every step.  The host
  `value` remains the public source for CPU runs and exact/symbolic scalar instantiations.  When a
  caller explicitly reads parameters after CUDA training, the device mirror is copied back here.
  -/
  cudaValue : IO.Ref (Option Runtime.Autograd.Cuda.AnyBuffer)
  /--
  Whether `value` is known to match `cudaValue`.

  CUDA optimizer steps mark this `false` after updating only the device mirror.  Public parameter
  readback synchronizes and flips it back to `true`.
  -/
  hostCurrent : IO.Ref Bool
  /-- Whether this parameter receives accumulated gradients and optimizer updates. -/
  requiresGrad : Bool := true

/--
Type-erased parameter wrapper.

This exists so session code can store heterogeneous parameter shapes in a single `HashMap` keyed
by leaf id (used for SGD updates).
-/
structure AnyParam (α : Type) where
  /-- Runtime shape of the erased parameter. -/
  s : Shape
  /-- Whether the underlying parameter receives optimizer updates. -/
  requiresGrad : Bool
  /-- Read the current parameter value with its runtime shape. -/
  get : IO (Runtime.AnyTensor α)
  /-- Overwrite the current parameter value, checking shape at the call site. -/
  set : Runtime.AnyTensor α → IO Unit
  /-- Store a CUDA buffer mirror without forcing an immediate host download. -/
  setCuda : Runtime.Autograd.Cuda.AnyBuffer → IO Unit

namespace AnyParam

/--
Make the result of a native CUDA cleanup call observable in `IO`.

Some CUDA cleanup functions are exposed as pure opaque calls because they are also useful in pure
buffer-building expressions. In executable cleanup paths, we still want the native call to be
sequenced with the surrounding eager-session updates. Branching into `IO` on the returned flag gives
Lean a real dependency on the result without printing anything or changing behavior.
-/
def observeCudaCleanupFlag (released : UInt32) : IO Unit :=
  if released == 0 then
    IO.sleep 0
  else
    pure ()

/-- Release a cached CUDA mirror, if one exists.

CUDA buffers are external objects whose native finalizer tolerates repeated cleanup attempts, but
parameter updates know exactly when an old device mirror is no longer the current value. Releasing
that mirror here keeps eager CUDA sessions explicit about ownership.
-/
def releaseCachedCudaValue {α : Type} {s : Shape} (p : Param α s) : IO Unit := do
  match ← p.cudaValue.get with
  | none => pure ()
  | some any =>
      let released := Runtime.Autograd.Cuda.Buffer.release any.buf
      observeCudaCleanupFlag released

/--
Package a typed `Param α s` as an `AnyParam α`, checking shape on `set`.

This is the bridge that allows generic optimizers/update routines to operate over heterogeneous
parameter packs.
-/
def ofParam {α : Type} {s : Shape} (p : Param α s) : AnyParam α :=
  { s := s
    requiresGrad := p.requiresGrad
    get := do
      let v ← p.value.get
      pure (Runtime.Autograd.AnyTensor.mk v)
    set := fun v => do
      if h : v.s = s then
        releaseCachedCudaValue p
        p.value.set (Tensor.castShape v.t h)
        p.cudaValue.set none
        p.hostCurrent.set true
      else
        throw <| IO.userError
          s!"torch: param update shape mismatch (expected {Shape.pretty s}, got {Shape.pretty v.s})"
    setCuda := fun v => do
      if _h : v.s = s then
        releaseCachedCudaValue p
        p.cudaValue.set (some { s := s, buf := v.buf })
        p.hostCurrent.set false
      else
        throw <| IO.userError
          s!"torch: CUDA param update shape mismatch (expected {Shape.pretty s}, got {Shape.pretty v.s})"
          }

end AnyParam

/-- Convenience: throw `IO.userError` on a `.error` result. -/
abbrev okOrThrow {α : Type} : Runtime.Autograd.Result α → IO α :=
  Runtime.Autograd.okOrThrow
end Torch
end Autograd
end Runtime
