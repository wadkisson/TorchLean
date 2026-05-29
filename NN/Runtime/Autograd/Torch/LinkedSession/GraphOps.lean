/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.LinkedSession.Core

/-!
# Proof-Linked Session: Basic Graph Operations
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec
open Tensor

namespace Internal

namespace SessionIR

/-! ## Graph-node ops (implemented by reusing `Compiled.GraphM`) -/

/--
Run a `Compiled.GraphM` computation against the current `(ss, g)` pair.

`Compiled.GraphM` is the builder monad used by the proof-friendly compiled pipeline; reusing it
here ensures this eager-style API records *the same* typed IR that the compiler expects.
-/
def runGraphM {α : Type} {Γ : List Shape} {β : Type}
    (m : Runtime.Autograd.Compiled.GraphM.MWith α NatEnv Γ β)
    (ss : List Shape) (g : _root_.Proofs.Autograd.Algebra.GraphData α NatEnv Γ ss) :
    Runtime.Autograd.Result (β × (Σ ss' : List Shape, _root_.Proofs.Autograd.Algebra.GraphData α
      NatEnv Γ ss')) :=
  StateT.run m ⟨ss, g⟩

/--
Atomically apply a graph-building update to the session snapshot.

This is the central adapter used by each op wrapper below: it reads `s.st`, runs a builder that
returns an updated `SessionIRState`, stores it back into `s.st`, and returns the op result.
-/
def commitGraphM {α : Type} (s : SessionIR α) {β : Type}
    (k :
      ∀ {Γ : List Shape} {ss : List Shape},
        (x : _root_.Proofs.Autograd.Algebra.TList α Γ) →
        (nat : NatEnv) →
        (g : _root_.Proofs.Autograd.Algebra.GraphData α NatEnv Γ ss) →
        Runtime.Autograd.Result (β × SessionIRState α)) :
    IO β := do
  let st0 ← s.st.get
  let r ← okOrThrow (k (Γ := st0.Γ) (ss := st0.ss) st0.x st0.nat st0.g)
  let (b, st1) := r
  s.st.set st1
  pure b

/--
Record a constant tensor.

Subtlety: if no op nodes have been created yet (`ss = []`), we record `const` as a leaf to match
the eager session's leaf-collection behavior. Once op nodes exist, we emit an explicit constant node
so users can introduce literal constants mid-graph.
PyTorch comparison: like `torch.tensor(...)` (a leaf) vs inserting a literal constant into the
graph; constants are treated as non-requires-grad.
-/
def const {α : Type} (s : SessionIR α) {sh : Shape} [Zero α] [DecidableEq Shape]
  (v : Tensor α sh) (name : Option String := none) : IO (TensorRef α sh) := do
  let _ := name
  let st0 ← s.st.get
  match st0.ss with
  | [] =>
      -- Still in the "leaf collection" phase: keep `const` as a leaf for parity with the eager
      -- Session.
      input (α := α) s (sh := sh) v (name := name) (requiresGrad := false)
  | _ :: _ =>
      -- Mid-graph: emit an explicit constant node.
      commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} xv nat g => do
        let (vout, st') ← runGraphM (α := α) (Γ := Γ)
          (Runtime.Autograd.Compiled.GraphM.const (α := α) (Γ := Γ) (s := sh) v)
          ss g
        let ⟨ss', g'⟩ := st'
        let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
        pure ({ id := vout.id }, st1))

/--
Record elementwise addition `a + b`.

PyTorch comparison: `torch.add(a, b)` / the `+` operator.
-/
def add {α : Type} (s : SessionIR α) [Add α] [Zero α] [DecidableEq Shape] {sh : Shape}
  (a b : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} x nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.add (α := α) (Γ := Γ) (s := sh) { id := a.id } { id := b.id
        })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := x, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise subtraction `a - b`.

PyTorch comparison: `torch.sub(a, b)` / the `-` operator.
-/
def sub {α : Type} (s : SessionIR α) [Sub α] [Add α] [Zero α] [DecidableEq Shape] {sh : Shape}
  (a b : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} x nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.sub (α := α) (Γ := Γ) (s := sh) { id := a.id } { id := b.id
        })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := x, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise multiplication `a * b`.

PyTorch comparison: `torch.mul(a, b)` / the `*` operator.
-/
def mul {α : Type} (s : SessionIR α) [Mul α] [Add α] [Zero α] [DecidableEq Shape] {sh : Shape}
  (a b : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} x nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.mul (α := α) (Γ := Γ) (s := sh) { id := a.id } { id := b.id
        })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := x, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record scaling by a scalar constant: `x * c`.

PyTorch comparison: like `x * c` (where `c` is a Python scalar).
-/
def scale {α : Type} (s : SessionIR α) [Mul α] [Add α] [Zero α] [DecidableEq Shape] {sh : Shape}
  (x : TensorRef α sh) (c : α) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.scale (α := α) (Γ := Γ) (s := sh) { id := x.id } c)
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise absolute value.

PyTorch comparison: `torch.abs(x)`.
-/
def abs {α : Type} (s : SessionIR α)
  [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.abs (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Stop-gradient boundary.

Forward semantics: identity.
Backward semantics: no gradient flows to the input.
PyTorch comparison: `x.detach()`.
-/
def detach {α : Type} (s : SessionIR α) [Context α] [DecidableEq Shape] {sh : Shape}
    (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.detach (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise square root.

PyTorch comparison: `torch.sqrt(x)`.
-/
def sqrt {α : Type} (s : SessionIR α)
  [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.sqrt (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise clamp to the interval `[minVal, maxVal]`.

PyTorch comparison: `torch.clamp(x, min=minVal, max=maxVal)`.
-/
def clamp {α : Type} (s : SessionIR α)
  [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {sh : Shape} (x : TensorRef α sh) (minVal maxVal : α) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.clamp (α := α) (Γ := Γ) (s := sh) { id := x.id } minVal
        maxVal)
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise maximum of `a` and `b`.

PyTorch comparison: `torch.maximum(a, b)`.
-/
def max {α : Type} (s : SessionIR α)
  [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {sh : Shape} (a b : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} x nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.max (α := α) (Γ := Γ) (s := sh) { id := a.id } { id := b.id
        })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := x, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise minimum of `a` and `b`.

PyTorch comparison: `torch.minimum(a, b)`.
-/
def min {α : Type} (s : SessionIR α)
  [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {sh : Shape} (a b : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} x nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.min (α := α) (Γ := Γ) (s := sh) { id := a.id } { id := b.id
        })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := x, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record 2D matrix multiplication.

PyTorch comparison: `torch.matmul(a, b)` for 2D tensors.
-/
def matmul {α : Type} (s : SessionIR α) [Context α] [DecidableEq Shape]
  {m n p : Nat}
  (a : TensorRef α (.dim m (.dim n .scalar)))
  (b : TensorRef α (.dim n (.dim p .scalar))) :
  IO (TensorRef α (.dim m (.dim p .scalar))) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim m (.dim p .scalar))) (fun {Γ} {ss} x nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.matmul (α := α) (Γ := Γ) (m := m) (n := n) (p := p) { id :=
        a.id } { id := b.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := x, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record batched matrix multiplication.

PyTorch comparison: `torch.bmm(a, b)` for 3D tensors of shape `(batch, m, n)` and `(batch, n, p)`.
-/
def bmm {α : Type} (s : SessionIR α) [Add α] [Mul α] [Zero α] [DecidableEq Shape]
  {batch m n p : Nat}
  (a : TensorRef α (.dim batch (.dim m (.dim n .scalar))))
  (b : TensorRef α (.dim batch (.dim n (.dim p .scalar)))) :
  IO (TensorRef α (.dim batch (.dim m (.dim p .scalar)))) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim batch (.dim m (.dim p .scalar)))) (fun {Γ} {ss} x
    nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.bmm (α := α) (Γ := Γ) (batch := batch) (m := m) (n := n) (p
        := p) { id := a.id } { id := b.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := x, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Concatenate two 1D vectors along dimension 0.

PyTorch comparison: `torch.cat([a, b], dim=0)` for 1D tensors.
-/
def concatVectors {α : Type} (s : SessionIR α) [Context α] [DecidableEq Shape]
  {n m : Nat}
  (a : TensorRef α (.dim n .scalar))
  (b : TensorRef α (.dim m .scalar)) :
  IO (TensorRef α (.dim (n + m) .scalar)) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim (n + m) .scalar)) (fun {Γ} {ss} x nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.concatVectors (α := α) (Γ := Γ) (n := n) (m := m) { id :=
        a.id } { id := b.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := x, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Concatenate two tensors along dimension 0.

PyTorch comparison: `torch.cat([a, b], dim=0)`.
-/
def concatDim0 {α : Type} (s : SessionIR α) [Context α] [DecidableEq Shape]
  {n m : Nat} {sh : Shape}
  (a : TensorRef α (.dim n sh))
  (b : TensorRef α (.dim m sh)) :
  IO (TensorRef α (.dim (n + m) sh)) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim (n + m) sh)) (fun {Γ} {ss} x nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.concatDim0 (α := α) (Γ := Γ) (n := n) (m := m) (s := sh)
        { id := a.id } { id := b.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := x, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Slice a tensor along dimension 0.

This returns `x[start : start+len]`. The proof argument `h` enforces bounds.
PyTorch comparison: `x[start:start+len]` for tensors with a leading dimension.
-/
def sliceRange0 {α : Type} (s : SessionIR α) [Zero α] [DecidableEq Shape]
  {n : Nat} {sh : Shape}
  (x : TensorRef α (.dim n sh)) (start len : Nat) (h : len + start ≤ n) :
  IO (TensorRef α (.dim len sh)) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim len sh)) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.Compiled.GraphM.sliceRange0 (α := α) (Γ := Γ) (n := n) (s := sh) { id :=
        x.id } start len h)
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : SessionIRState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))
end SessionIR

end Internal

end Torch
end Autograd
end Runtime

