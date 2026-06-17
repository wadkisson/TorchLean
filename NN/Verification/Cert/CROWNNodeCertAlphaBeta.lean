/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Cert.AlphaBetaCROWN
public import NN.MLTheory.CROWN.Graph
public import NN.Spec.Core.Utils
public import NN.Verification.Cert.Common
public import Lean.Data.Json

/-!
# CROWNNodeCertAlphaBeta

Per-node α/β-CROWN certificate checking (graph dialect).

This extends `NN.Verification.CROWNNodeCert` with an optional β phase vector for ReLU nodes.

Certificate JSON format:

```json
{
  "ctx": { "inputId": 0, "inputDim": 2 },
  "ibp": [ null | { "lo": [...], "hi": [...] }, ... ],
  "crown": [
    null |
      { "loA": [[...], ...], "loC": [...],
        "hiA": [[...], ...], "hiC": [...] },
    ...
  ],
  "alpha": [ null | [...], ... ],
  "beta":  [ null | [-1,0,1,...], ... ]   // optional per-node ReLU phase vector
}
```

β encoding (per neuron):
- `-1` = forced inactive (`z ≤ 0`)
- `0`  = unconstrained / unstable
- `1`  = forced active (`0 ≤ z`)

As with the α-CROWN checker, the certificate is accepted only if Lean recomputation matches the
provided affine bounds up to a small float tolerance.
-/

@[expose] public section


namespace NN.Verification.CROWNNodeCertAlphaBeta

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN.Cert
open NN.Verification.Json
open NN.Verification.Cert.Common
open Import.PyTorch
open _root_.Spec
open _root_.Spec.Tensor
open Lean Data Json

/-!
Helpers for the alpha/beta-CROWN style node certificate checker.

These are the JSON-facing utilities for the checker: they parse imported bounds, compare
decimal-serialized floats with an explicit tolerance, and keep shape mismatches from reaching the
semantic checker.
-/

/-- Parse a JSON integer (used for beta vectors). -/
def parseInt? (j : Json) : Option Int :=
  match j with
  | .num n => n.toString.toInt?
  | .str s => s.toInt?
  | _ => none

/-- Parse a beta vector from JSON. -/
def parseBetaVec? (dim : Nat) (j : Json) : IO (Option (Array Int)) := do
  match j with
  | .null => pure none
  | .arr xs =>
      if hSize : xs.size = dim then
        let mut out : Array Int := Array.mkEmpty dim
        for k in List.finRange dim do
          let h : k.val < xs.size := by
            rw [hSize]
            exact k.isLt
          let some i := parseInt? (xs[k.val]'h)
            | throw <| IO.userError s!"Invalid beta[i][{k.val}]: expected int"
          if i = (-1) || i = 0 || i = 1 then
            out := out.push i
          else
            throw <| IO.userError s!"Invalid beta[i][{k.val}]: expected -1/0/1"
        pure (some out)
      else
        throw <| IO.userError s!"Invalid beta[i]: expected int array length {dim}"
  | _ => throw <| IO.userError "Invalid beta[i]: expected null or int array"

/-!
`AlphaBetaCROWNNodeCertificate` is the in-memory representation of an alpha/beta-CROWN node
certificate read from JSON.

This is part of the public surface of the checker because `readAlphaBetaCROWNNodeCertificate`
returns it, and because the blueprint points to it as the “shape of the artifact” being checked.
-/
structure AlphaBetaCROWNNodeCertificate where
  /-- Affine-propagation context, including the chosen input node and flattened input dimension. -/
  ctx : AffineCtx
  /-- Optional per-node interval bounds used by nonlinear CROWN steps. -/
  ibp : Array (Option (FlatBox Float))
  /-- Optional per-node affine lower/upper bounds. -/
  crown : Array (Option (FlatAffineBounds Float))
  /-- Optional per-node α values for ReLU lower relaxations. -/
  alpha : Array (Option (FlatVec Float))
  /-- Optional per-node β phase annotations for ReLU nodes. -/
  beta : Array (Option (Array Int))

/-- Read an alpha/beta-CROWN node certificate from JSON on disk. -/
def readAlphaBetaCROWNNodeCertificate (g : Graph) (path : String) : IO AlphaBetaCROWNNodeCertificate := do
  let topObj ← readJsonObjectFile path
  let core ← parseCROWNNodeCoreCertificate g topObj
  let betaArr ←
    match ← optionalField? topObj "beta" "top-level" with
    | none => pure (Array.replicate g.nodes.size Json.null)
    | some betaJ => expectArray betaJ "top-level.beta"

  if hSize : betaArr.size = g.nodes.size then
    let mut beta : Array (Option (Array Int)) := Array.mkEmpty g.nodes.size
    for i in List.finRange g.nodes.size do
      let node := g.nodes[i.val]'i.isLt
      let hBeta : i.val < betaArr.size := by
        rw [hSize]
        exact i.isLt
      let betaJson := betaArr[i.val]'hBeta
      let betaEntry ← parseBetaVec? node.outShape.size betaJson
      beta := beta.push betaEntry
    pure { ctx := core.ctx, ibp := core.ibp, crown := core.crown, alpha := core.alpha, beta := beta }
  else
    throw <| IO.userError s!"beta length {betaArr.size} ≠ g.nodes.size {g.nodes.size}"

/-- Check the local alpha/beta-CROWN enclosure condition for one node against a certificate entry.
  -/
def checkAlphaBetaCROWNNode (g : Graph) (ps : ParamStore Float)
    (certIbp : Array (Option (FlatBox Float)))
    (certAlpha : Array (Option (FlatVec Float)))
    (certBeta : Array (Option (Array Int)))
    (certCrown : Array (Option (FlatAffineBounds Float)))
    (ctx : AffineCtx)
    (id : Nat) (tol : Float) : IO Bool := do
  let computed? :=
    alphaBetaCrownStepNode? (α := Float) g.nodes ps certIbp certAlpha certBeta certCrown ctx id
  checkCROWNLikeNode "CROWNNodeCertAlphaBeta" g certIbp certCrown ctx id tol computed?

/--
Check a per-node α/β-CROWN certificate against Lean's propagation rules.

Returns `true` iff every node's certificate affine bounds agree (within `tol`) with what Lean
computes from the certificate parents + `ParamStore`.
-/
def checkAlphaBetaCROWNNodeCertificate (g : Graph) (ps : ParamStore Float) (path : String) (tol : Float := 1e-4) :
    IO Bool :=
  do
  let cert ← readAlphaBetaCROWNNodeCertificate g path
  let mut ok := true
  for id in [0:g.nodes.size] do
    let okNode ← checkAlphaBetaCROWNNode g ps cert.ibp cert.alpha cert.beta cert.crown cert.ctx id tol
    ok := ok && okNode
  if ok then
    IO.println "[CROWNNodeCertAlphaBeta] certificate verified: all nodes match Lean α/β-CROWN step."
  pure ok

end NN.Verification.CROWNNodeCertAlphaBeta
