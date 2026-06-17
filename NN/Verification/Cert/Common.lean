/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph
public import NN.Runtime.PyTorch.Import.Core
public import NN.Spec.Core.Utils
public import NN.Verification.Util.FloatApprox
public import NN.Verification.Util.Json
public import Lean.Data.Json

/-!
# Common Certificate Helpers

Shared JSON/parsing and approximate-comparison utilities for node-wise verification certificates.

The IBP, α-CROWN, and α/β-CROWN checkers all consume the same basic artifact shapes:
flat interval boxes, affine lower/upper bounds, and optional per-node vectors.  We keep those
format-level helpers here so the individual checkers can focus on their propagation rule:

- `IBPNodeCert` checks interval propagation;
- `CROWNNodeCert` checks affine CROWN propagation;
- `CROWNNodeCertAlphaBeta` checks affine CROWN propagation with β phase information.

The JSON artifact is always untrusted. These helpers only parse and compare data; acceptance still
requires each checker to recompute the corresponding bound inside Lean. The float tolerances here
exist because JSON stores decimal strings/numbers, not because the external producer is trusted.
-/

@[expose] public section

namespace NN.Verification.Cert.Common

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open NN.Verification.Util
open NN.Verification.Json
open Import.PyTorch
open _root_.Spec
open _root_.Spec.Tensor
open Lean Data Json

/--
Approximate equality for flat scalar tensors (length-`n` vectors), up to an absolute tolerance.

This is used when comparing Lean-recomputed bounds to decimal-serialized JSON certificate values.
-/
def approxEqTensor {n : Nat} (t u : Tensor Float (.dim n .scalar)) (tol : Float) : Bool :=
  match t, u with
  | .dim ft, .dim fu =>
      (List.finRange n).all (fun i =>
        match ft i, fu i with
        | .scalar a, .scalar b => approxEq a b (tol := tol))

/-- Approximate equality for flat matrices (shape `m × n`), up to an absolute tolerance. -/
def approxEqMatrix {m n : Nat}
    (A B : Tensor Float (.dim m (.dim n .scalar))) (tol : Float) : Bool :=
  match A, B with
  | .dim rA, .dim rB =>
      (List.finRange m).all (fun i =>
        match rA i, rB i with
        | .dim cA, .dim cB =>
            (List.finRange n).all (fun j =>
              match cA j, cB j with
              | .scalar a, .scalar b => approxEq a b (tol := tol)))

/-- Approximate equality for `FlatBox` bounds, componentwise on `lo` and `hi`. -/
def approxEqFlatBox (B1 B2 : FlatBox Float) (tol : Float) : Bool :=
  if h : B1.dim = B2.dim then
    match B1, B2 with
    | ⟨n, lo1, hi1⟩, ⟨_m, lo2, hi2⟩ =>
        by
          cases h
          exact approxEqTensor (n := n) lo1 lo2 tol && approxEqTensor (n := n) hi1 hi2 tol
  else false

/-- Approximate equality for affine vectors, componentwise on matrix `A` and offset `c`. -/
def approxEqAffineVec {n m : Nat} (a b : AffineVec Float n m) (tol : Float) : Bool :=
  approxEqMatrix (m := m) (n := n) a.A b.A tol && approxEqTensor (n := m) a.c b.c tol

/-- Approximate equality for flattened affine lower/upper bounds. -/
def approxEqFlatAffineBounds (B1 B2 : FlatAffineBounds Float) (tol : Float) : Bool :=
  if hin : B1.inDim = B2.inDim then
    if hout : B1.outDim = B2.outDim then
      match B1, B2 with
      | ⟨n1, m1, lo1, hi1⟩, ⟨_n2, _m2, lo2, hi2⟩ =>
          by
            cases hin
            cases hout
            exact approxEqAffineVec (n := n1) (m := m1) lo1 lo2 tol &&
              approxEqAffineVec (n := n1) (m := m1) hi1 hi2 tol
    else false
  else false

/-- Parse a flat interval box (two arrays of floats) from JSON. -/
def parseFlatBox? (dim : Nat) (j : Json) : IO (Option (FlatBox Float)) := do
  match j with
  | .null => pure none
  | _ =>
      let o ← expectObj j "ibp[i]"
      let loJ ← expectField o "lo" "ibp[i]"
      let hiJ ← expectField o "hi" "ibp[i]"
      let some loVec := parseFloatVec dim loJ
        | throw <| IO.userError s!"Invalid ibp[i].lo: expected float array length {dim}"
      let some hiVec := parseFloatVec dim hiJ
        | throw <| IO.userError s!"Invalid ibp[i].hi: expected float array length {dim}"
      let loT : Tensor Float (.dim dim .scalar) := Spec.vectorTensor loVec
      let hiT : Tensor Float (.dim dim .scalar) := Spec.vectorTensor hiVec
      pure (some { dim := dim, lo := loT, hi := hiT })

/--
Parse an optional α vector for α-CROWN ReLU relaxations.

The soundness theorem for the lower ReLU relaxation assumes every α component is in `[0, 1]`.
We enforce that contract at the JSON boundary, so a malformed external certificate cannot be
accepted by executable checking while relying on proof hypotheses that are false.
-/
def parseAlphaVec? (dim : Nat) (j : Json) (ctx : String := "alpha[i]") :
    IO (Option (FlatVec Float)) := do
  match j with
  | .null => pure none
  | _ =>
      let some v := parseFloatVec dim j
        | throw <| IO.userError s!"Invalid {ctx}: expected float array length {dim}"
      for k in List.finRange dim do
        let a := v k
        if a < 0.0 || a > 1.0 then
          throw <| IO.userError
            s!"Invalid {ctx}[{k.val}]: α-CROWN requires 0 ≤ alpha ≤ 1, got {a}"
      let t : Tensor Float (.dim dim .scalar) := Spec.vectorTensor v
      pure (some { n := dim, v := t })

/-- Parse flattened affine bounds (lower/upper) from JSON. -/
def parseAffineBounds? (inDim outDim : Nat) (j : Json) : IO (Option (FlatAffineBounds Float)) := do
  match j with
  | .null => pure none
  | _ =>
      let o ← expectObj j "crown[i]"
      let loAJ ← expectField o "loA" "crown[i]"
      let loCJ ← expectField o "loC" "crown[i]"
      let hiAJ ← expectField o "hiA" "crown[i]"
      let hiCJ ← expectField o "hiC" "crown[i]"
      let some loA := parseFloatMatrix outDim inDim loAJ
        | throw <| IO.userError s!"Invalid crown[i].loA: expected matrix {outDim}x{inDim}"
      let some hiA := parseFloatMatrix outDim inDim hiAJ
        | throw <| IO.userError s!"Invalid crown[i].hiA: expected matrix {outDim}x{inDim}"
      let some loC := parseFloatVec outDim loCJ
        | throw <| IO.userError s!"Invalid crown[i].loC: expected float array length {outDim}"
      let some hiC := parseFloatVec outDim hiCJ
        | throw <| IO.userError s!"Invalid crown[i].hiC: expected float array length {outDim}"
      let loAff : AffineVec Float inDim outDim :=
        { A := Spec.matrixTensor loA, c := Spec.vectorTensor loC }
      let hiAff : AffineVec Float inDim outDim :=
        { A := Spec.matrixTensor hiA, c := Spec.vectorTensor hiC }
      pure (some { inDim := inDim, outDim := outDim, loAff := loAff, hiAff := hiAff })

/--
Shared in-memory representation for node-wise CROWN-style certificates.

Plain α-CROWN uses these fields directly. α/β-CROWN extends the same core artifact with a β phase
array, so parsing the common fields here keeps the two checkers from drifting apart.
-/
structure CROWNNodeCoreCertificate where
  /-- Affine-propagation context, including the chosen input node and flattened input dimension. -/
  ctx : AffineCtx
  /-- Optional per-node interval bounds used by nonlinear CROWN steps. -/
  ibp : Array (Option (FlatBox Float))
  /-- Optional per-node affine lower/upper bounds. -/
  crown : Array (Option (FlatAffineBounds Float))
  /-- Optional per-node α values for ReLU lower relaxations. -/
  alpha : Array (Option (FlatVec Float))

/--
Parse the fields shared by α-CROWN and α/β-CROWN node certificates.

The producer may omit `"alpha"`; in that case we treat every node as having no custom α vector.
Whenever α is present, `parseAlphaVec?` enforces the `[0,1]` side condition required by the ReLU
relaxation proof.
-/
def parseCROWNNodeCoreCertificate (g : Graph) (topObj : Json) :
    IO CROWNNodeCoreCertificate := do
  let ctxObj ← expectFieldObj topObj "ctx" "top-level"
  let inputId ← expectFieldNat ctxObj "inputId" "ctx"
  let inputDim ← expectFieldNat ctxObj "inputDim" "ctx"
  let ctx : AffineCtx := { inputId := inputId, inputDim := inputDim }

  let ibpArr ← expectFieldArray topObj "ibp" "top-level"
  let crownArr ← expectFieldArray topObj "crown" "top-level"
  let alphaArr ←
    match ← optionalField? topObj "alpha" "top-level" with
    | none => pure (Array.replicate g.nodes.size Json.null)
    | some alphaJ => expectArray alphaJ "top-level.alpha"

  if hIbpSize : ibpArr.size = g.nodes.size then
    if hCrownSize : crownArr.size = g.nodes.size then
      if hAlphaSize : alphaArr.size = g.nodes.size then
        let mut ibp : Array (Option (FlatBox Float)) := Array.mkEmpty g.nodes.size
        let mut crown : Array (Option (FlatAffineBounds Float)) := Array.mkEmpty g.nodes.size
        let mut alpha : Array (Option (FlatVec Float)) := Array.mkEmpty g.nodes.size

        for i in List.finRange g.nodes.size do
          let node := g.nodes[i.val]'i.isLt
          let outDim := node.outShape.size
          let hIbp : i.val < ibpArr.size := by
            rw [hIbpSize]
            exact i.isLt
          let hCrown : i.val < crownArr.size := by
            rw [hCrownSize]
            exact i.isLt
          let hAlpha : i.val < alphaArr.size := by
            rw [hAlphaSize]
            exact i.isLt
          let ibpJson := ibpArr[i.val]'hIbp
          let ibpEntry ← parseFlatBox? outDim ibpJson
          ibp := ibp.push ibpEntry
          let crownJson := crownArr[i.val]'hCrown
          let crownEntry ← parseAffineBounds? ctx.inputDim outDim crownJson
          crown := crown.push crownEntry
          let alphaJson := alphaArr[i.val]'hAlpha
          let alphaEntry ← parseAlphaVec? outDim alphaJson
          alpha := alpha.push alphaEntry

        pure { ctx := ctx, ibp := ibp, crown := crown, alpha := alpha }
      else
        throw <| IO.userError s!"alpha length {alphaArr.size} ≠ g.nodes.size {g.nodes.size}"
    else
      throw <| IO.userError s!"crown length {crownArr.size} ≠ g.nodes.size {g.nodes.size}"
  else
    throw <| IO.userError s!"ibp length {ibpArr.size} ≠ g.nodes.size {g.nodes.size}"

/-- Check that an optional per-node certificate array contains all parents of node `id`. -/
def parentsOk {β : Type} (g : Graph) (cert : Array (Option β)) (id : Nat) : Bool :=
  match g.nodes[id]? with
  | none => false
  | some node =>
      node.parents.all (fun p =>
        if p < id then
          match cert[p]? with
          | some (some _) => true
          | _ => false
        else
          false)

/-- Safe lookup for optional flat boxes used by certificate-side shape checks. -/
def getFlatBox? (cert : Array (Option (FlatBox Float))) (id : Nat) : Option (FlatBox Float) :=
  match cert[id]? with
  | some box? => box?
  | none => none

/--
Check that binary elementwise parent boxes have the same flattened size as each other and as the
node output. This closes the hole where a malformed certificate could make the runtime helper use
the left box on a dimension mismatch.
-/
def binaryElementwiseBoxesMatchOutput
    (g : Graph) (cert : Array (Option (FlatBox Float))) (id : Nat) : Bool :=
  match g.nodes[id]? with
  | none => false
  | some node =>
      match node.parents with
      | p1 :: p2 :: _ =>
          match getFlatBox? cert p1, getFlatBox? cert p2 with
          | some B1, some B2 =>
              B1.dim == B2.dim && B1.dim == node.outShape.size
          | _, _ => false
      | _ => false

/-- Check whether a flat box is entirely inside the positive domain needed by true `log`. -/
def flatBoxStrictlyAbove (B : FlatBox Float) (eps : Float) : Bool :=
  match B with
  | ⟨n, lo, hi⟩ =>
      let flo := getDimScalarFn (α := Float) lo
      let fhi := getDimScalarFn (α := Float) hi
      (List.finRange n).all (fun i =>
        match flo i, fhi i with
        | .scalar l, .scalar u => l > eps && u > eps)

/--
Domain and shape preconditions that must hold before a node-wise certificate checker replays a
bound step. These are not proof shortcuts; they are the executable version of the side conditions
that the mathematical rules need.
-/
def ibpNodePreconditionsOk
    (g : Graph) (cert : Array (Option (FlatBox Float))) (id : Nat) : Bool :=
  match g.nodes[id]? with
  | none => false
  | some node =>
      match node.kind with
      | .add | .sub | .mul_elem | .maxElem | .minElem =>
          binaryElementwiseBoxesMatchOutput g cert id
      | .log =>
          match node.parents with
          | p1 :: _ =>
              match getFlatBox? cert p1 with
              | some B => flatBoxStrictlyAbove B Numbers.epsilon
              | none => false
          | _ => false
      | _ => true

/-- Pretty-printer for a flat box, used in certificate mismatch messages. -/
def prettyFlatBox (B : FlatBox Float) : String :=
  s!"dim={B.dim}, lo={Spec.pretty B.lo}, hi={Spec.pretty B.hi}"

/-- Pretty-printer for affine bounds, used in certificate mismatch messages. -/
def prettyAffineBounds (B : FlatAffineBounds Float) : String :=
  s!"inDim={B.inDim}, outDim={B.outDim}, loA={Spec.pretty B.loAff.A}, loC={Spec.pretty B.loAff.c}"

/--
Common node-level checker for CROWN-style affine certificates.

The only difference between α-CROWN and α/β-CROWN is how the candidate affine bound is recomputed.
Everything after that point is the same: parent availability, IBP side conditions, dimensions,
approximate JSON equality, and diagnostic messages.
-/
def checkCROWNLikeNode
    (label : String) (g : Graph)
    (certIbp : Array (Option (FlatBox Float)))
    (certCrown : Array (Option (FlatAffineBounds Float)))
    (ctx : AffineCtx)
    (id : Nat) (tol : Float)
    (computed? : Option (FlatAffineBounds Float)) : IO Bool := do
  let some node := g.nodes[id]?
    | IO.eprintln s!"[{label}] node {id}: out of bounds for graph with {g.nodes.size} nodes"
      pure false
  let needsParents :=
    match node.kind with
    | .input | .const _ => false
    | _ => true
  if needsParents && !(parentsOk g certCrown id) then
    IO.eprintln s!"[{label}] node {id}: parent affine bounds missing or not topo"
    return false
  if !(ibpNodePreconditionsOk g certIbp id) then
    IO.eprintln
      s!"[{label}] node {id}: certificate violates shape/domain preconditions for {repr node.kind}"
    return false

  let certCrown? :=
    match certCrown[id]? with
    | some entry => entry
    | none => none
  match certCrown?, computed? with
  | none, _ =>
      IO.eprintln s!"[{label}] node {id}: certificate missing (null)"
      pure false
  | _, none =>
      IO.eprintln s!"[{label}] node {id}: Lean propagation produced no affine bound"
      pure false
  | some certB, some leanB =>
      if certB.inDim ≠ ctx.inputDim then
        IO.eprintln s!"[{label}] node {id}: cert inDim {certB.inDim} ≠ ctx.inputDim {ctx.inputDim}"
        pure false
      else if certB.outDim ≠ node.outShape.size then
        IO.eprintln
          (s!"[{label}] node {id}: cert outDim {certB.outDim} ≠ " ++
            s!"outShape.size {node.outShape.size}")
        pure false
      else if leanB.outDim ≠ node.outShape.size then
        IO.eprintln
          (s!"[{label}] node {id}: Lean outDim {leanB.outDim} ≠ " ++
            s!"outShape.size {node.outShape.size}")
        pure false
      else if approxEqFlatAffineBounds certB leanB tol then
        pure true
      else
        IO.eprintln s!"[{label}] mismatch at node {id} ({repr node.kind})"
        IO.eprintln s!"  cert: {prettyAffineBounds certB}"
        IO.eprintln s!"  lean: {prettyAffineBounds leanB}"
        pure false

end NN.Verification.Cert.Common
