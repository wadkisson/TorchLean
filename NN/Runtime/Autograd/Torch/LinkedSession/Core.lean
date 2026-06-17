/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core
import Mathlib.Algebra.Order.Algebra

/-!
# LinkedSession

Proof-linked imperative Session (eager-style API, proved IR under the hood).

Background:
- `Runtime.Autograd.TorchLean.Session` provides a unified imperative API for training/debugging
  (eager) and verification-friendly execution (compiled).
- `Proofs.Autograd.Algebra.GraphData` is the proved/typed SSA(DAG) IR used by the
  proof-compiled pipeline (`Proofs.Autograd.Algebra.Graph.compileAuxData`), and
  `NN/Proofs/Autograd/Runtime/Link.lean` proves that running the runtime reverse-mode loop on the
  compiled tape matches `GraphData.backpropAllCtx`.

This file provides a *session-style* API that records a `GraphData` (well-typed IR) as you call
ops imperatively, and then runs the standard runtime tape loop on the compiled tape.

Key guarantee (pure theorem, no `IO` reasoning needed):
- If the session snapshot is `(g, x)`, then `Tape.backwardDenseFrom (compileAuxData g x)` equals
  `GraphData.backpropAllCtx g x` (via `backwardDenseFrom_compileAuxData_eq_backpropAllCtx`).

Practical note:
- This session enforces a simple invariant: **all leaf tensors are created before any op node**.
  This matches the standard training pattern (reset → add leaves → forward → backward).
- `const` is available as a graph node, so you can still introduce literal constants mid-graph.
- This is the fully proof-linked variant used by `TorchLean.Session` when `opts.backend :=
  .compiled`.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Torch

open Spec
open Tensor

namespace Internal

/--
Convenience: turn a `Result α` into `IO α` by throwing `IO.userError` on `.error`.

This mirrors the common pattern in the eager runtime front-end (`Torch.Core`).
-/
abbrev okOrThrow {α : Type} : Runtime.Autograd.Result α → IO α :=
  Runtime.Autograd.okOrThrow

/-- Non-differentiable external environment for the proved graph: a small array of `Nat` inputs. -/
abbrev NatEnv : Type := Array Nat

/-- Internal proof-linked session state (a well-typed `GraphData` plus its leaf values). -/
structure SessionIRState (α : Type) where
  /-- Leaf shapes (inputs/parameters), in creation order. -/
  Γ : List Shape
  /-- Leaf values, aligned with `Γ`. -/
  x : _root_.Proofs.Autograd.Algebra.TList α Γ
  /-- Non-differentiable external inputs (e.g. class labels/indices). -/
  nat : NatEnv
  /-- Internal node shapes, in creation order. -/
  ss : List Shape
  /-- SSA/DAG graph nodes (one per entry in `ss`). -/
  g : _root_.Proofs.Autograd.Algebra.GraphData α NatEnv Γ ss

namespace SessionIRState

/-- Empty session state: no leaves, no nodes, empty nat-environment. -/
def empty {α : Type} : SessionIRState α :=
  { Γ := []
    x := .nil
    nat := #[]
    ss := []
    g := .nil }

end SessionIRState

/--
`SessionIR` is an imperative session that records a `GraphData` (proved IR) as it runs.

It is "eager-style" (you call ops imperatively), but it is proof-linked: the recorded graph can be
compiled and then the runtime tape backward loop is provably equal to `GraphData.backpropAllCtx`.
-/
structure SessionIR (α : Type) where
  /-- Session options shared with the eager front-end. -/
  opts : Options
  /-- Mutable proof-linked graph snapshot. -/
  st : IO.Ref (SessionIRState α)
  /-- Map from graph leaf ids to mutable parameter objects. -/
  paramsByLeaf : IO.Ref (Std.HashMap Nat (AnyParam α))

namespace SessionIR

/--
Create a new proof-linked session.

This allocates `IO.Ref`s for the session snapshot (`SessionIRState`) and the leaf-id→parameter map.
Call `resetTape` to start a new "graph recording" phase.
-/
def new {α : Type} (opts : Options := {}) : IO (SessionIR α) := do
  let st ← IO.mkRef (SessionIRState.empty (α := α))
  let paramsByLeaf ← IO.mkRef (Std.HashMap.emptyWithCapacity)
  pure { opts := opts, st := st, paramsByLeaf := paramsByLeaf }

/--
Reset the session to an empty snapshot.

Important invariant: this session requires that **all leaves are created before any op node**.
`resetTape` is the intended boundary between training steps/forwards.
-/
def resetTape {α : Type} (s : SessionIR α) : IO Unit := do
  s.st.set (SessionIRState.empty (α := α))
  s.paramsByLeaf.set (Std.HashMap.emptyWithCapacity)

/--
Create a mutable parameter object (not yet part of the recorded graph).

To use the parameter in the recorded graph, call `use`, which reads its current value and records
it as a *leaf* in `Γ`.
PyTorch comparison: analogous to creating a `torch.nn.Parameter` and then using it in a forward.
-/
def param {α : Type} (s : SessionIR α) {sh : Shape}
  (init : Tensor α sh) (name : Option String := none) (requiresGrad : Option Bool := none) :
  IO (Param α sh) := do
  let r ← IO.mkRef init
  let cudaValue ← IO.mkRef (none : Option Runtime.Autograd.Cuda.AnyBuffer)
  let hostCurrent ← IO.mkRef true
  pure { name := name
         value := r
         cudaValue := cudaValue
         hostCurrent := hostCurrent
         requiresGrad := requiresGrad.getD s.opts.requiresGradByDefault }

/--
Enforce the session invariant: leaves must be created before any op node.

This matches the usual training pattern: `resetTape → add leaves → forward ops → backward`.
-/
def ensureNoNodes {α : Type} (st : SessionIRState α) : IO Unit := do
  match st.ss with
  | [] => pure ()
  | _ :: _ =>
      throw <| IO.userError
        ("torch(SessionIR): cannot add a new leaf after graph nodes have been " ++
          "created (resetTape first)")

/--
Record a new differentiable leaf tensor in the session context `Γ`.

This is the primitive used by `use` (parameters) and `input` (external inputs).
-/
def addLeaf {α : Type} (s : SessionIR α) {sh : Shape} (v : Tensor α sh) :
    IO (TensorRef α sh) := do
  let st0 ← s.st.get
  ensureNoNodes st0
  let id := st0.Γ.length
  let Γ' := st0.Γ ++ [sh]
  let x' : _root_.Proofs.Autograd.Algebra.TList α Γ' :=
    _root_.Proofs.Autograd.Algebra.TList.snoc (α := α) (ss := st0.Γ) (τ := sh) st0.x v
  -- No nodes yet, so the graph stays `nil`.
  let st1 : SessionIRState α :=
    { Γ := Γ'
      x := x'
      nat := st0.nat
      ss := []
      g := .nil }
  s.st.set st1
  pure { id := id }

/--
Use a `Param` in the recorded graph by reading its current value and recording it as a leaf.

The returned `TensorRef` is the graph handle you pass to subsequent ops. The session also remembers
which leaf-id corresponds to which parameter, so `sgdStepAll` can update parameters after backward.
PyTorch comparison: like referencing a `torch.nn.Parameter` in the forward; the parameter's value
is treated as a leaf for autograd.
-/
def use {α : Type} (s : SessionIR α) {sh : Shape} [DecidableEq Shape]
  (p : Param α sh) : IO (TensorRef α sh) := do
  let v ← p.value.get
  let leaf ← addLeaf (α := α) s (sh := sh) v
  s.paramsByLeaf.modify (fun m => m.insert leaf.id (AnyParam.ofParam p))
  pure leaf

/--
Record an external differentiable input tensor as a leaf.

`name` and `requiresGrad` are accepted for API parity with the eager session, but this proof-linked
session always records the input in `Γ` (a leaf) and uses typing/invariants to determine what
gradients are meaningful.
-/
def input {α : Type} (s : SessionIR α) {sh : Shape} [DecidableEq Shape]
  (v : Tensor α sh) (name : Option String := none) (requiresGrad : Bool := false) :
  IO (TensorRef α sh) := do
  -- `name`/`requiresGrad` are accepted for API parity with the eager Session.
  -- This proof-linked session always records the value as a leaf in `Γ`.
  let _ := name
  let _ := requiresGrad
  addLeaf (α := α) s (sh := sh) v

/--
Record a non-differentiable `Nat` input in the external environment.

This is used for "index-like" inputs (labels, gather indices, etc.) that should not receive
gradients.
PyTorch comparison: like passing an integer tensor / index to an op; indices are not differentiable.
-/
def inputNat {α : Type} (s : SessionIR α) (v : Nat) : IO NatRef := do
  let st0 ← s.st.get
  ensureNoNodes st0
  let id := st0.nat.size
  s.st.set { st0 with nat := st0.nat.push v }
  pure { id := id }

/-- Read a previously recorded `NatRef`. -/
def getNat {α : Type} (s : SessionIR α) (r : NatRef) : IO Nat := do
  let st0 ← s.st.get
  if h : r.id < st0.nat.size then
    pure <| st0.nat[r.id]'h
  else
    throw <| IO.userError "torch(SessionIR): invalid nat id"

/-- Overwrite a previously recorded `NatRef`. -/
def setNat {α : Type} (s : SessionIR α) (r : NatRef) (v : Nat) : IO Unit := do
  let st0 ← s.st.get
  if h : r.id < st0.nat.size then
    let i : Fin st0.nat.size := ⟨r.id, h⟩
    s.st.set { st0 with nat := st0.nat.set i v }
  else
    throw <| IO.userError "torch(SessionIR): invalid nat id"

/--
Convert a small `Tensor Nat (.dim k .scalar)` into an `Array Nat`.

This is used to stage `NatVecRef` inputs into the session nat-environment.
-/
def natVecToArray {k : Nat} (v : Tensor Nat (.dim k .scalar)) : Array Nat :=
  Array.ofFn (fun i : Fin k =>
    match getAtSpec v i with
    | .scalar n => n)

/--
Record a non-differentiable vector of `Nat` inputs.

Returns a `NatVecRef k` which points into the nat-environment. This is useful for "runtime gather"
style ops where indices are supplied externally (and are not differentiable).
-/
def inputNatVec {α : Type} {k : Nat} (s : SessionIR α) (v : Tensor Nat (.dim k .scalar)) : IO
  (NatVecRef k) := do
  let st0 ← s.st.get
  ensureNoNodes st0
  let start := st0.nat.size
  let xsNew := (natVecToArray (k := k) v).foldl (fun acc x => acc.push x) st0.nat
  s.st.set { st0 with nat := xsNew }
  pure { start := start }

/-- Read back the `k`-vector stored at a `NatVecRef k`. -/
def getNatVec {α : Type} {k : Nat} (s : SessionIR α) (r : NatVecRef k) : IO (Tensor Nat (.dim k
  .scalar)) := do
  let st0 ← s.st.get
  if h : r.start + k ≤ st0.nat.size then
    pure <|
      Tensor.dim (fun i =>
        have hi : r.start + i.val < r.start + k := Nat.add_lt_add_left i.is_lt r.start
        have hi' : r.start + i.val < st0.nat.size := lt_of_lt_of_le hi h
        Tensor.scalar (st0.nat[r.start + i.val]'hi'))
  else
    throw <| IO.userError "torch(SessionIR): invalid nat vec ref (out of bounds)"

/-- Overwrite the nat-environment segment referenced by `NatVecRef k`. -/
def setNatVec {α : Type} {k : Nat} (s : SessionIR α) (r : NatVecRef k) (v : Tensor Nat (.dim k
  .scalar)) : IO Unit := do
  let st0 ← s.st.get
  if h : r.start + k ≤ st0.nat.size then
    let xs' :=
      (List.finRange k).foldl (fun acc (i : Fin k) =>
        have hi : r.start + i.val < st0.nat.size := by
          have hlt : r.start + i.val < r.start + k := Nat.add_lt_add_left i.is_lt r.start
          exact lt_of_lt_of_le hlt h
        let vi : Nat :=
          match getAtSpec v i with
          | .scalar n => n
        acc.set! (r.start + i.val) vi
      ) st0.nat
    s.st.set { st0 with nat := xs' }
  else
    throw <| IO.userError "torch(SessionIR): invalid nat vec ref (out of bounds)"

/--
Build a typed index into the current context `Γ ++ ss` from a raw numeric id and expected shape.

This is the main "dynamic check" used by `getValue` (and by a few index-driven nodes): it ensures
that the `Nat` id points to an existing tensor in the session context and that the shape matches.
-/
def mkIdxOrThrow {_α : Type} {Γ ss : List Shape} (id : Nat) (s : Shape) :
    Runtime.Autograd.Result (_root_.Proofs.Autograd.Algebra.Idx (Γ ++ ss) s) := by
    if h : id < (Γ ++ ss).length then
      let fin : Fin (Γ ++ ss).length := ⟨id, h⟩
      let got : Shape := (Γ ++ ss).get fin
      if hg : got = s then
        exact .ok ⟨fin, hg⟩
      else
        exact .error <|
          s!"torch(SessionIR): shape mismatch at id={id}: expected {Shape.pretty s}, got "
            ++ s!"{Shape.pretty got}"
  else
    exact .error s!"torch(SessionIR): invalid id={id} for ctxLen={(Γ ++ ss).length}"

/--
Evaluate the recorded graph and return the value of a `TensorRef`.

This is a pure graph evaluation (`GraphData.eval`) using the recorded leaf values and
nat-environment. It does **not** run the runtime tape or mutate session state.
-/
def getValue {α : Type} (s : SessionIR α) {sh : Shape} [DecidableEq Shape]
  (x : TensorRef α sh) : IO (Tensor α sh) := do
  let st0 ← s.st.get
  -- Evaluate the recorded graph at the recorded leaf values.
  let ctx : _root_.Proofs.Autograd.Algebra.TList α (st0.Γ ++ st0.ss) :=
    _root_.Proofs.Autograd.Algebra.GraphData.eval (α := α) (Δ := NatEnv) (Γ := st0.Γ) (ss := st0.ss)
      st0.g st0.x st0.nat
  let idx ← okOrThrow (mkIdxOrThrow (_α := α) (Γ := st0.Γ) (ss := st0.ss) x.id sh)
  pure (_root_.Proofs.Autograd.Algebra.getIdx (α := α) (xs := ctx) idx)
end SessionIR

end Internal

end Torch
end Autograd
end Runtime
