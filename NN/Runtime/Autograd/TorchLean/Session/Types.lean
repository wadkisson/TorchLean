/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.Session.Eager

/-!
Session value types.

The definitions here describe runtime tensor handles, parameter references, and shape-indexed
containers used by the TorchLean session API.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor

/--
Implementation choice for a `Session`.

This is an internal sum type used to dispatch each operation to either:
- the eager tape-backed runtime (`.eager`), or
- the proof-linked compiled runtime (`.compiled`).
-/
inductive SessionImpl (α : Type) where
  | eager (s : EagerSession α)
  | compiled (s : _root_.Runtime.Autograd.Torch.Internal.SessionIR α)

/--
Unified imperative session: choose `.eager` vs `.compiled` at construction via `opts.backend`.

This is the recommended "one interface" for:
- training/debugging (eager),
- verification-friendly execution (compiled/proof-linked),
without users having to learn two different Session APIs.
-/
structure Session (α : Type) where
  /-- opts. -/
  opts : _root_.Runtime.Autograd.Torch.Options
  /-- impl. -/
  impl : SessionImpl α

namespace Session

/--
Create a new unified session.

The backend is selected by `opts.backend`:
- `.eager` builds a tape-backed runtime session, and
- `.compiled` builds a proof-linked compiled session.
-/
def new {α : Type} (opts : _root_.Runtime.Autograd.Torch.Options := {}) : IO (Session α) := do
  match opts.backend with
  | .eager =>
      let s ← EagerSession.new (α := α) (opts := opts)
      pure { opts := opts, impl := .eager s }
  | .compiled =>
      let s ← _root_.Runtime.Autograd.Torch.Internal.SessionIR.new (α := α) (opts := opts)
      pure { opts := opts, impl := .compiled s }

/-- Reset the autograd tape / graph-building state. -/
def resetTape {α : Type} (s : Session α) : IO Unit := do
  match s.impl with
  | .eager sess => EagerSession.resetTape (α := α) sess
  | .compiled sess => _root_.Runtime.Autograd.Torch.Internal.SessionIR.resetTape (α := α) sess

/--
Create a learnable parameter (a leaf tensor) owned by the session.

PyTorch analogue: `torch.nn.Parameter` (conceptually), created inside a module/init and later used
in forward passes.
-/
def param {α : Type} (s : Session α) {sh : Shape}
  (init : Tensor α sh) (name : Option String := none) (requiresGrad : Option Bool := none) :
  IO (_root_.Runtime.Autograd.Torch.Param α sh) := do
  match s.impl with
  | .eager sess =>
      EagerSession.param (α := α) sess (sh := sh)
        init (name := name) (requiresGrad := requiresGrad)
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.param (α := α) sess (sh := sh)
        init (name := name) (requiresGrad := requiresGrad)

/--
Use a parameter inside the current tape/graph.

This returns a `TensorRef` that can be passed to ops to build a forward graph.
-/
def use {α : Type} (s : Session α) {sh : Shape} [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  (p : _root_.Runtime.Autograd.Torch.Param α sh) : IO (_root_.Runtime.Autograd.Torch.TensorRef α sh)
    := do
  match s.impl with
  | .eager sess => EagerSession.use (α := α) sess (sh := sh) p
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.use (α := α) sess (sh := sh) p

/--
Add an input tensor to the current tape/graph.

Inputs are leaf tensors that may or may not require gradients (controlled by `requiresGrad`).
-/
def input {α : Type} (s : Session α) {sh : Shape} [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  (v : Tensor α sh) (name : Option String := none) (requiresGrad : Bool := false) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.impl with
  | .eager sess =>
      EagerSession.input (α := α) sess (sh := sh) v (name := name) (requiresGrad := requiresGrad)
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.input (α := α) sess (sh := sh)
        v (name := name) (requiresGrad := requiresGrad)

/--
Add a non-differentiable `Nat` input to the session.

This is used for labels/indices (e.g. classification targets, gather indices) without forcing a
numeric embedding into `α`.
-/
def inputNat {α : Type} (s : Session α) (v : Nat) : IO (_root_.Runtime.Autograd.Torch.NatRef) := do
  match s.impl with
  | .eager sess => EagerSession.inputNat (α := α) sess v
  | .compiled sess => _root_.Runtime.Autograd.Torch.Internal.SessionIR.inputNat (α := α) sess v

/-- Read back a `NatRef` value. -/
def getNat {α : Type} (s : Session α) (r : _root_.Runtime.Autograd.Torch.NatRef) : IO Nat := do
  match s.impl with
  | .eager sess => EagerSession.getNat (α := α) sess r
  | .compiled sess => _root_.Runtime.Autograd.Torch.Internal.SessionIR.getNat (α := α) sess r

/-- Mutate a `NatRef` value. -/
def setNat {α : Type} (s : Session α) (r : _root_.Runtime.Autograd.Torch.NatRef) (v : Nat) : IO Unit
  := do
  match s.impl with
  | .eager sess => EagerSession.setNat (α := α) sess r v
  | .compiled sess => _root_.Runtime.Autograd.Torch.Internal.SessionIR.setNat (α := α) sess r v

/--
Add a non-differentiable vector-of-`Nat` input leaf.

This is convenient for batched indices (e.g. gather a batch of rows) without embedding indices into
  `α`.
-/
def inputNatVec {α : Type} {k : Nat} (s : Session α) (v : Tensor Nat (.dim k .scalar)) :
    IO (_root_.Runtime.Autograd.Torch.NatVecRef k) := do
  match s.impl with
  | .eager sess => EagerSession.inputNatVec (α := α) (k := k) sess v
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.inputNatVec (α := α) (k := k) sess v

/-- Read back a `NatVecRef` value. -/
def getNatVec {α : Type} {k : Nat} (s : Session α) (r : _root_.Runtime.Autograd.Torch.NatVecRef k) :
    IO (Tensor Nat (.dim k .scalar)) := do
  match s.impl with
  | .eager sess => EagerSession.getNatVec (α := α) (k := k) sess r
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.getNatVec (α := α) (k := k) sess r

/-- Mutate a `NatVecRef` value. -/
def setNatVec {α : Type} {k : Nat} (s : Session α) (r : _root_.Runtime.Autograd.Torch.NatVecRef k)
    (v : Tensor Nat (.dim k .scalar)) : IO Unit := do
  match s.impl with
  | .eager sess => EagerSession.setNatVec (α := α) (k := k) sess r v
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.setNatVec (α := α) (k := k) sess r v

/-! ### Deterministic RNG state (Session-level) -/

/--
Deterministic RNG state stored inside a session.

We model RNG state explicitly using two non-differentiable leaves:
- `seed`: the current seed value
- `counter`: a monotone counter used to derive fresh keys

PyTorch analogy: explicit `torch.manual_seed` + per-op counter, but represented as explicit state.
-/
structure RngState where
  /-- Random seed. -/
  seed : _root_.Runtime.Autograd.Torch.NatRef
  /-- counter. -/
  counter : _root_.Runtime.Autograd.Torch.NatRef

/--
Initialize an `RngState` from a concrete seed.

This allocates two `NatRef`s in the session (`seed` and `counter`) and initializes `counter` to 0.
-/
def initRng {α : Type} (s : Session α) (seed : Nat) : IO RngState := do
  let seedRef ← inputNat (α := α) s seed
  let counterRef ← inputNat (α := α) s 0
  pure { seed := seedRef, counter := counterRef }

/-- Draw a fresh seed from `IO` (best-effort entropy). -/
def freshSeedIO : IO Nat := do
  -- We use `IO.rand` for practicality/ergonomics; this is *not* part of the semantic core.
  -- The semantic model remains seed-threaded deterministic RNG: this just chooses an initial seed.
  IO.rand 0 (Nat.pow 2 63 - 1)

/--
Initialize a deterministic RNG state by sampling an initial seed from `IO`.

This is the recommended "PyTorch-like ergonomics, JAX-like semantics" bridge:
- you get a convenient source of entropy at the boundary,
- but the *core* semantics remains deterministic and replayable given the chosen seed.
-/
def initRngFromIO {α : Type} (s : Session α) : IO RngState := do
  initRng (α := α) s (← freshSeedIO)

/-!
Practical note: in the proof-linked `.compiled` backend, the current session implementation
requires that all leaves (tensor inputs/parameters and `NatRef`s) are created before any op nodes.
So for maximum portability, initialize and split RNG states *up-front* before building a graph.
-/

/--
Split an RNG stream into a fresh child stream (deterministic).

This is useful for isolating submodules (e.g. separate dropout sites) without sharing RNG state.
-/
def splitRng {α : Type} (s : Session α) (rng : RngState) : IO RngState := do
  let seedNat ← getNat (α := α) s rng.seed
  let ctrNat ← getNat (α := α) s rng.counter
  -- Derive two fresh seeds deterministically.
  let childSeed := Random.nextSeed seedNat ctrNat
  let parentSeed := Random.nextSeed childSeed (ctrNat + 1)
  setNat (α := α) s rng.seed parentSeed
  setNat (α := α) s rng.counter (ctrNat + 2)
  initRng (α := α) s childSeed

/--
Insert a constant tensor into the current graph.

PyTorch analogy: using a tensor constant/literal in `forward`.
-/
def const {α : Type} (s : Session α) {sh : Shape} [Zero α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  (v : Tensor α sh) (name : Option String := none) : IO (_root_.Runtime.Autograd.Torch.TensorRef α
    sh) := do
  match s.impl with
  | .eager sess => EagerSession.const (α := α) sess (sh := sh) v (name := name)
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.const (α := α) sess (sh := sh) v
        (name := name)

/-- Read the concrete value for a tensor ref (for logging/debugging). -/
def getValue {α : Type} (s : Session α) {sh : Shape} [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO (Tensor α sh) := do
  match s.impl with
  | .eager sess => EagerSession.getValue (α := α) sess (sh := sh) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.getValue (α := α) sess (sh := sh) x

/--
Detach a tensor ref from the graph (stop gradient flow through it).

PyTorch analogy: `x.detach()`.
-/
def detach {α : Type} (s : Session α) [Context α] {sh : Shape} [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
    IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.impl with
  | .eager sess => EagerSession.detach (α := α) sess (sh := sh) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.detach (α := α) sess (sh := sh) x


end Session

end TorchLean
end Autograd
end Runtime
