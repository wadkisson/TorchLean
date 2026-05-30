/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Context
public import NN.Spec.Layers.Attention
public import NN.Spec.Layers.Conv
public import NN.Spec.Layers.Linear
public import NN.Spec.Layers.Normalization
public import NN.Spec.Layers.Pooling

/-!
# Engine Core

A small dynamic (DAG) autograd engine.

This is intended to be the "runtime" counterpart to `Spec.OpSpec`: instead of manually writing
backward passes end-to-end, you build a tape during a forward pass and then call `backward`.

Design goals:
- Works for arbitrary `Tensor α s` shapes using `Runtime.AnyTensor` packing.
- Supports DAGs (shared subexpressions): gradients are accumulated by summation.
- Keeps the API compact and shape-safe enough for practical use.

Scope boundaries:
- A fully verified analytic calculus proof (`HasFDerivAt` etc.) for all ops. The engine is
  correct *given* the local backward rules used to build nodes, and we add small regression
  checks in `NN/Tests/Runtime`.
- For a PyTorch-style imperative API over this tape, see `NN.Runtime.Autograd.Torch.Core`.

References (PyTorch / background reading):
- PyTorch docs: `torch.autograd` and "Autograd mechanics":
  https://pytorch.org/docs/stable/autograd.html
  https://pytorch.org/docs/stable/notes/autograd.html
- "micrograd" (small autograd engine, useful for intuition):
  https://github.com/karpathy/micrograd
-/

@[expose] public section


namespace Runtime
namespace Autograd

open Spec
open Tensor

/-!
## Core Types

The eager autograd engine is built out of a few small pieces:

* `Result`: a pure error monad used throughout the tape API.
* `AnyTensor`: a shape-erased tensor used to store heterogeneous node values on a single tape.
* `Node`: one recorded computation step, with a local backward (VJP) rule.
* `Tape`: a grow-only array of nodes; reverse-mode traversals walk it in reverse order.
-/

/--
Runtime error monad for the eager autograd engine.

We use plain `Except String` (instead of `IO` exceptions) so the tape constructors remain pure and
easy to test. Front-ends that prefer exceptions can use `okOrThrow`.
-/
abbrev Result (α : Type) := Except String α

/--
Convert an `Autograd.Result` into an `IO` action by throwing `IO.userError` on failure.

This is mainly used by the imperative Torch/TorchLean front-ends to keep their code readable.
-/
def okOrThrow {α : Type} : Result α → IO α
  | .ok a => pure a
  | .error e => throw <| IO.userError e

namespace AnyTensor

/--
Pack a typed tensor as a runtime `AnyTensor`.

This is the primary bridge between the dependently-typed `Tensor α s` world and the dynamic tape,
which stores heterogeneous shapes in a single array.
-/
def mk {α : Type} {s : Shape} (t : Tensor α s) : Runtime.AnyTensor α :=
  { s := s, t := t }

/--
Cast an `AnyTensor` to a specific shape, given an equality proof.

This is used after dynamic shape checks (e.g. `Tape.requireValue`).
-/
def cast {α : Type} {s₂ : Shape} (t : Runtime.AnyTensor α) (h : t.s = s₂) : Tensor α s₂ :=
  Tensor.castShape t.t h

/--
Accumulate two `AnyTensor` values by elementwise addition, with a dynamic shape check.

This is the heart of DAG support: if two different paths contribute gradients to the same parent,
we sum the contributions.
-/
def add {α : Type} [Add α] [DecidableEq Shape]
  (a b : Runtime.AnyTensor α) : Result (Runtime.AnyTensor α) := by
  if h : a.s = b.s then
    let b' : Tensor α a.s := Tensor.castShape b.t h.symm
    exact .ok { s := a.s, t := addSpec a.t b' }
  else
    exact .error "autograd: gradient shape mismatch during accumulation"

end AnyTensor

/--
A tape node representing a single tensor value in the recorded computation graph.

Fields:
- `value`: the forward value (shape-erased).
- `parents`: ids of parent nodes in the tape.
- `backward`: a local VJP rule. Given an upstream cotangent for `value`, it returns a list of
  `(parentId, parentCotangent)` contributions (one per parent, usually).

PyTorch comparison: analogous to an autograd `Function` instance + saved tensors, but here we store
the backward closure directly.
-/
structure Node (α : Type) where
  /-- Optional node name used for debugging and pretty-printing. -/
  name : Option String := none
  /-- Forward value computed at this node (shape-erased). -/
  value : Runtime.AnyTensor α
  /--
  Whether reverse-mode propagation should visit this node.

  If `false`, reverse-mode traversal skips this node and does not accumulate gradients into it.
  -/
  requires_grad : Bool := true
  /-- Parent node ids (dependencies) in the tape. -/
  parents : List Nat := []
  /--
  Local VJP rule for this node.

  Given an upstream cotangent for `value`, return a list of `(parentId, parentCotangent)`
  contributions. If multiple children contribute to the same parent, the engine will sum
  contributions via `AnyTensor.add`.
  -/
  backward : Runtime.AnyTensor α → Result (List (Nat × Runtime.AnyTensor α))

/--
Autograd tape: a grow-only array of nodes.

Node ids are array indices (`Nat`). All ops append exactly one node and return its id.
This makes it easy to implement reverse-mode by traversing ids in reverse order.
-/
structure Tape (α : Type) where
  /--
  Tape nodes in evaluation order.

  Node ids are array indices (`Nat`). Each tape op appends exactly one node and returns its id.
  -/
  nodes : Array (Node α) := #[]

/-!
## Tape Construction

The `Tape` namespace provides *pure* constructors for building a recorded computation graph.
Each op appends exactly one node and returns the updated tape plus the new node id.

If you prefer an implicit tape-threading style, see `NN.Runtime.Autograd.Engine.TapeM`.
-/

namespace Tape

variable {α : Type}

/-- Empty tape (no nodes). -/
def empty : Tape α := {}

/-- Number of nodes stored in the tape. -/
def size (t : Tape α) : Nat := t.nodes.size

/-- Read a node by id (returns `none` if out of bounds). -/
def getNode? (t : Tape α) (id : Nat) : Option (Node α) :=
  t.nodes[id]?

/-- Read just the stored forward value for a node id. -/
def getValue? (t : Tape α) (id : Nat) : Option (Runtime.AnyTensor α) :=
  (t.getNode? id).map (·.value)

/--
Append a node and return its id.

Invariant: the returned id is `t.size`, the pre-append size of the tape.
-/
def addNode (t : Tape α) (node : Node α) : Tape α × Nat :=
  let id := t.nodes.size
  ({ nodes := t.nodes.push node }, id)

/-- `addNode` returns the current tape size as the fresh node id. -/
@[simp] theorem addNode_id (t : Tape α) (node : Node α) :
    (t.addNode node).2 = t.size := by
  simp [addNode, size]

/-- Appending a node increases the tape size by one. -/
@[simp] theorem size_addNode (t : Tape α) (node : Node α) :
    (t.addNode node).1.size = t.size + 1 := by
  simp [addNode, size]

/--
Add a leaf node (no parents).

PyTorch comparison: a tensor that enters the graph as a leaf (e.g. input or parameter value).
-/
def leaf {α : Type} {s : Shape}
  (t : Tape α) (value : Tensor α s) (name : Option String := none) (requires_grad : Bool := true) :
  Tape α × Nat :=
  t.addNode {
    name := name,
    value := AnyTensor.mk value,
    requires_grad := requires_grad,
    parents := [],
    backward := fun _ => .ok []
  }

/--
Read a typed tensor value from a tape node id.

This is the main "dynamic check" boundary in the eager runtime:
- fails if the id is invalid, or
- fails if the stored runtime shape doesn't match the expected dependent shape `s`.
-/
def requireValue {α : Type} [DecidableEq Shape] {s : Shape}
  (t : Tape α) (id : Nat) : Result (Tensor α s) := by
  match t.getValue? id with
  | none => exact .error "autograd: invalid node id"
  | some any =>
    if h : any.s = s then
      exact .ok (Tensor.castShape any.t h)
    else
      exact .error "autograd: shape mismatch"

/--
Read a typed upstream gradient tensor from a runtime `AnyTensor`.

This is the backward analogue of `Tape.requireValue`: it checks that the upstream gradient has the
expected shape `τ` and then performs the dependent cast.
-/
def requireGrad {α : Type} [DecidableEq Shape] {τ : Shape}
    (dLdyAny : Runtime.AnyTensor α) : Result (Tensor α τ) := by
  if h : dLdyAny.s = τ then
    exact .ok (Tensor.castShape dLdyAny.t h)
  else
    exact .error "autograd: upstream gradient shape mismatch"

/--
Generic constructor for unary ops.

You provide:
- `forward : Tensor α σ → Tensor α τ`
- `backward : Tensor α σ → Tensor α τ → Tensor α σ` (a VJP rule; note it may depend on the input)

The returned node stores the forward value and a backward closure that checks the upstream
gradient's shape and returns the parent contribution.
-/
def unary {α : Type} [DecidableEq Shape] {σ τ : Shape}
  (t : Tape α) (opName : String) (xId : Nat)
  (forward : Tensor α σ → Tensor α τ)
  (backward : Tensor α σ → Tensor α τ → Tensor α σ) :
  Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=σ) xId
  let y := forward x
  let node : Node α :=
    { name := some opName
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := τ) dLdyAny
        let dLdx : Tensor α σ := backward x dLdy
        pure [(xId, AnyTensor.mk dLdx)]
    }
  pure (t.addNode node)
