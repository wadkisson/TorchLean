/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph
public import NN.Tensor.API
public import NN.Verification.Util.Json

/-!
# PINN Core

PINN helper library: reference graphs, seeding, and certificate parsing.

This module is shared by the PINN verification workflows. It provides:
- small CROWN graphs for a tanh MLP (1D and 2D inputs),
- deterministic parameters and input-box seeding helpers,
- a few interval/finite-difference residual helpers,
- JSON parsing for the certificate schema used by the surrounding examples.

Run the curated entrypoints instead of importing this file directly:
- `lake exe verify -- pinn-cert [NN/Examples/Verification/PINN/pinn_cert.json]`
- `lake exe verify -- pinn-dataset-check --dataset=PATH.json [--weights=WEIGHTS.json]`

References:
- PINNs (physics-informed neural nets): `https://arxiv.org/abs/1711.10561`
- CROWN (linear bound propagation): `https://arxiv.org/abs/1811.00866`
- IBP (interval bound propagation): `https://arxiv.org/abs/1810.12715`
-/

@[expose] public section


namespace NN.Verification.PINN

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open _root_.Spec
open _root_.Spec.Tensor
open Lean
open Json

/-- Require a JSON object field in an `Except` parser. -/
def expectFieldObjE (ctx key : String) (j : Json) :
    Except String (Std.TreeMap.Raw String Json compare) := do
  NN.API.Json.expectObjE s!"{ctx}.{key}" (← NN.API.Json.expectFieldE ctx key j)

/-- Require a JSON array field in an `Except` parser. -/
def expectFieldArrayE (ctx key : String) (j : Json) : Except String (Array Json) := do
  NN.API.Json.expectArrayE s!"{ctx}.{key}" (← NN.API.Json.expectFieldE ctx key j)

/-- Require a JSON array whose entries are all floats. -/
def expectFloatArrayE (ctx : String) (j : Json) : Except String (Array Float) := do
  let xs ← NN.API.Json.expectArrayE ctx j
  xs.mapIdxM (fun i x => NN.Verification.Json.expectFloatE s!"{ctx}[{i}]" x)

/-- Require a float-array field. -/
def expectFieldFloatArrayE (ctx key : String) (j : Json) :
    Except String (Array Float) := do
  expectFloatArrayE s!"{ctx}.{key}" (← NN.API.Json.expectFieldE ctx key j)

/-- Require an interval object `{ "lo": ..., "hi": ... }`. -/
def expectIntervalPairE (ctx : String) (j : Json) : Except String (Float × Float) := do
  let lo ← NN.Verification.Json.expectFieldFloatE ctx "lo" j
  let hi ← NN.Verification.Json.expectFieldFloatE ctx "hi" j
  pure (lo, hi)

/-- Require an array of interval pairs with an expected length. -/
def expectIntervalPairArrayE (ctx : String) (j : Json) (expected : Nat) :
    Except String (List (Float × Float)) := do
  let loA ← expectFieldFloatArrayE ctx "lo" j
  let hiA ← expectFieldFloatArrayE ctx "hi" j
  if h : loA.size = expected ∧ hiA.size = expected then
    pure <| (List.finRange expected).map (fun (i : Fin expected) =>
      have hlo : i.val < loA.size := by
        rw [h.1]
        exact i.isLt
      have hhi : i.val < hiA.size := by
        rw [h.2]
        exact i.isLt
      (loA[i.val]'hlo, hiA[i.val]'hhi))
  else
    throw s!"{ctx}: length mismatch (expected {expected})"

/-- Require one finite-difference `u_bounds` entry. -/
def expectUBoundsEntryE (ctx : String) (j : Json) :
    Except String (Float × (Float × Float) × (Float × Float) × (Float × Float)) := do
  let x ← NN.Verification.Json.expectFieldFloatE ctx "x" j
  let uMinus ← expectIntervalPairE s!"{ctx}.u_minus" (← NN.API.Json.expectFieldE ctx "u_minus" j)
  let u ← expectIntervalPairE s!"{ctx}.u" (← NN.API.Json.expectFieldE ctx "u" j)
  let uPlus ← expectIntervalPairE s!"{ctx}.u_plus" (← NN.API.Json.expectFieldE ctx "u_plus" j)
  pure (x, uMinus, u, uPlus)

/-- Configuration parsed from a PINN certificate JSON. -/
structure PinnCfg where
  /-- PDE identifier carried by the certificate. -/
  pde : String
  /-- Grid spacing used by the exported finite-difference residual. -/
  h   : Float
  /-- Input perturbation radius for interval checking. -/
  eps : Float
  /-- Number of sample points encoded in `pts`. -/
  nPts : Nat
  /--
  Sample points as a length-`nPts` 1D tensor.

  PyTorch analogue: this is the `torch.Tensor` you would keep in memory after loading a JSON/CSV
  list of sample coordinates.
  -/
  pts : Spec.Tensor Float (.dim nPts .scalar)

/-- Approximate equality for `Float` used by certificate consistency checks. -/
def approxEq (x y : Float) (tol : Float := 1e-5) : Bool :=
  let d := if x > y then x - y else y - x
  decide (d ≤ tol)

/-- Reference tanh-MLP graph for `u : R -> R` used by the compact PINN certificate workflow. -/
def buildGraph : Graph :=
  let n0 : Node := { id := 0, parents := [], kind := .input,  outShape := .dim 1 .scalar }
  let n1 : Node := { id := 1, parents := [0], kind := .linear, outShape := .dim 16 .scalar }
  let n2 : Node := { id := 2, parents := [1], kind := .tanh,   outShape := .dim 16 .scalar }
  let n3 : Node := { id := 3, parents := [2], kind := .linear, outShape := .dim 16 .scalar }
  let n4 : Node := { id := 4, parents := [3], kind := .tanh,   outShape := .dim 16 .scalar }
  let n5 : Node := { id := 5, parents := [4], kind := .linear, outShape := .dim 1 .scalar }
  { nodes := #[n0,n1,n2,n3,n4,n5] }

/-- Same reference architecture as `buildGraph`, but with a 2D input `u : R^2 -> R`. -/
def buildGraph2D : Graph :=
  -- Same architecture but with 2-D input
  let n0 : Node := { id := 0, parents := [], kind := .input,  outShape := .dim 2 .scalar }
  let n1 : Node := { id := 1, parents := [0], kind := .linear, outShape := .dim 16 .scalar }
  let n2 : Node := { id := 2, parents := [1], kind := .tanh,   outShape := .dim 16 .scalar }
  let n3 : Node := { id := 3, parents := [2], kind := .linear, outShape := .dim 16 .scalar }
  let n4 : Node := { id := 4, parents := [3], kind := .tanh,   outShape := .dim 16 .scalar }
  let n5 : Node := { id := 5, parents := [4], kind := .linear, outShape := .dim 1 .scalar }
  { nodes := #[n0,n1,n2,n3,n4,n5] }

/-- Deterministic weights matching the exporter convention (1D input). -/
def seedParamsFloat : ParamStore Float :=
  -- Same as exporter: W1: 16x1, b1:16; Wm:16x16,bm:16; W2:1x16,b2:1
  let W1 : Tensor Float (.dim 16 (.dim 1 .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun _ => Tensor.scalar (Float.ofNat (i.val + 1) * 0.1)))
  let b1 : Tensor Float (.dim 16 .scalar) :=
    Tensor.dim (fun i => Tensor.scalar (0.05 * (Float.ofNat i.val - 8.0)))
  let Wm : Tensor Float (.dim 16 (.dim 16 .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (if decide (i.val = j.val) then 1.0 else
      0.05)))
  let bm : Tensor Float (.dim 16 .scalar) := Tensor.dim (fun _ => Tensor.scalar 0.0)
  let W2 : Tensor Float (.dim 1 (.dim 16 .scalar)) :=
    Tensor.dim (fun _ => Tensor.dim (fun j => Tensor.scalar (0.1 + 0.01 * (Float.ofNat j.val))))
  let b2 : Tensor Float (.dim 1 .scalar) := Tensor.dim (fun _ => Tensor.scalar 0.0)
  let ps0 : ParamStore Float := {}
  let ps1 := { ps0 with linearWB := ps0.linearWB.insert 1 ({ m := 16, n := 1,  w := W1, b := b1 }) }
  let ps2 := { ps1 with linearWB := ps1.linearWB.insert 3 ({ m := 16, n := 16, w := Wm, b := bm }) }
  let ps3 := { ps2 with linearWB := ps2.linearWB.insert 5 ({ m := 1,  n := 16, w := W2, b := b2 }) }
  ps3

/-- Deterministic weights for the 2D variant `buildGraph2D`. -/
def seedParamsFloat2D : ParamStore Float :=
  -- First layer adapted to 2D input: 16x2
  let W1 : Tensor Float (.dim 16 (.dim 2 .scalar)) :=
    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        let base := Float.ofNat (i.val + 1) * 0.05
        let w := if decide (j.val = 0) then base * 2.0 else base
        Tensor.scalar w))
  let b1 : Tensor Float (.dim 16 .scalar) :=
    Tensor.dim (fun i => Tensor.scalar (0.05 * (Float.ofNat i.val - 8.0)))
  let Wm : Tensor Float (.dim 16 (.dim 16 .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (if decide (i.val = j.val) then 1.0 else
      0.05)))
  let bm : Tensor Float (.dim 16 .scalar) := Tensor.dim (fun _ => Tensor.scalar 0.0)
  let W2 : Tensor Float (.dim 1 (.dim 16 .scalar)) :=
    Tensor.dim (fun _ => Tensor.dim (fun j => Tensor.scalar (0.1 + 0.01 * (Float.ofNat j.val))))
  let b2 : Tensor Float (.dim 1 .scalar) := Tensor.dim (fun _ => Tensor.scalar 0.0)
  let ps0 : ParamStore Float := {}
  let ps1 := { ps0 with linearWB := ps0.linearWB.insert 1 ({ m := 16, n := 2,  w := W1, b := b1 }) }
  let ps2 := { ps1 with linearWB := ps1.linearWB.insert 3 ({ m := 16, n := 16, w := Wm, b := bm }) }
  let ps3 := { ps2 with linearWB := ps2.linearWB.insert 5 ({ m := 1,  n := 16, w := W2, b := b2 }) }
  ps3

/-- Seed a 1D input box `[x - eps, x + eps]` at node id 0. -/
def seedInputFloat (ps : ParamStore Float) (x : Float) (eps : Float) : ParamStore Float :=
  let x0 : Tensor Float (.dim 1 .scalar) := Tensor.dim (fun _ => Tensor.scalar x)
  ps.seedLInfBall 0 x0 eps

/-- Seed a 2D input box `[(x,y) - eps, (x,y) + eps]` at node id 0. -/
def seedInputFloat2D (ps : ParamStore Float) (x y : Float) (eps : Float) : ParamStore Float :=
  let x0 : Tensor Float (.dim 2 .scalar) :=
    Tensor.dim (fun i => Tensor.scalar (if decide (i.val = 0) then x else y))
  ps.seedLInfBall 0 x0 eps

/-
  Hessian/Laplacian helpers (2D)
  ------------------------------
  Helpers for computing d2u/dx2, d2u/dy2, and their sum (Laplacian) at the unique output node
  (id=5) from an already-seeded graph and parameter store.
 -/

/-- Compute (d2u/dx2, d2u/dy2) intervals at node id=5 when available.
    Returns a pair of Options for X and Y directions respectively. -/
def hessian2D (g : Graph) (ps : ParamStore Float)
  : (Option (Float × Float) × Option (Float × Float)) :=
  let ibp := NN.MLTheory.CROWN.Graph.runIBP (α:=Float) g ps
  -- infer input dimension from input box seed
  let inDim :=
    match ps.inputBoxes[0]? with
    | some B => B.dim
    | none => 1
  -- X direction
  let d2x : Option (Float × Float) :=
    let seedX := FlatBox.ofTensor (NN.Tensor.oneHotNat (α := Float) inDim 0)
    let d1x := NN.MLTheory.CROWN.Graph.runDerivDirectional (α:=Float) g ps ibp seedX
    let d2x := NN.MLTheory.CROWN.Graph.runDeriv2D (α:=Float) g ps ibp d1x
    match d2x[5]? with
    | some (some B) => some (Spec.Tensor.sumSpec B.lo, Spec.Tensor.sumSpec B.hi)
    | _ => none
  -- Y direction (only if inDim ≥ 2)
  let d2y : Option (Float × Float) :=
    if inDim ≥ 2 then
      let seedY := FlatBox.ofTensor (NN.Tensor.oneHotNat (α := Float) inDim 1)
      let d1y := NN.MLTheory.CROWN.Graph.runDerivDirectional (α:=Float) g ps ibp seedY
      let d2y := NN.MLTheory.CROWN.Graph.runDeriv2D (α:=Float) g ps ibp d1y
      match d2y[5]? with
      | some (some B) => some (Spec.Tensor.sumSpec B.lo, Spec.Tensor.sumSpec B.hi)
      | _ => none
    else none
  (d2x, d2y)

/-- Laplacian upper/lower interval: Δu = u_xx + u_yy. For 1D inputs, returns u_xx. -/
def laplacianBounds2D (g : Graph) (ps : ParamStore Float) : Option (Float × Float) :=
  let (d2x, d2y) := hessian2D g ps
  match d2x, d2y with
  | some (lx, ux), some (ly, uy) => some (lx + ly, ux + uy)
  | some (lx, ux), none => some (lx, ux)
  | _, _ => none

/-- Parse the JSON certificate consumed by the PINN verification CLI. -/
def parseCert (j : Json) : Except String (PinnCfg × (List (Float × Float)) × (List (Float × Float))
  × (List (Float × (Float×Float) × (Float×Float) × (Float×Float)))) := do
  let o ← NN.API.Json.expectObjE "PINN certificate" j
  let po ← expectFieldObjE "PINN certificate" "pinn" j
  let pdeStr ←
    match Std.TreeMap.Raw.get? po "pde" with
    | none => pure "u''(x) = 0"
    | some Json.null => pure "u''(x) = 0"
    | some pdeJ =>
        match pdeJ with
        | .str s => pure s
        | _ => throw "PINN certificate.pinn.pde: expected string"
  let h ← NN.Verification.Json.expectFieldFloatE "PINN certificate.pinn" "h" (.obj po)
  let eps ← NN.Verification.Json.expectFieldFloatE "PINN certificate.pinn" "eps" (.obj po)
  let ptsA ← expectFieldFloatArrayE "PINN certificate.pinn" "points" (.obj po)
  let nPts := ptsA.size
  let pts : Spec.Tensor Float (.dim nPts .scalar) :=
    Spec.Tensor.dim (fun i => Spec.Tensor.scalar ptsA[i])
  let rb ← NN.API.Json.expectFieldE "PINN certificate" "residual_bounds" j
  let resPairs ← expectIntervalPairArrayE "PINN certificate.residual_bounds" rb nPts
  -- optional derivative-based residuals
  let resPairsDeriv ←
    match Std.TreeMap.Raw.get? o "residual_bounds_deriv" with
    | some Json.null => pure <| List.replicate nPts (0.0, 0.0)
    | some derivJ => expectIntervalPairArrayE "PINN certificate.residual_bounds_deriv" derivJ nPts
    | _ => pure <| List.replicate nPts (0.0, 0.0)
  let ubA ← expectFieldArrayE "PINN certificate" "u_bounds" j
  let uTriples ← (List.finRange ubA.size).mapM (fun (i : Fin ubA.size) =>
    expectUBoundsEntryE s!"PINN certificate.u_bounds[{i.val}]" ubA[i.val])
  pure ({ pde := pdeStr, h := h, eps := eps, nPts := nPts, pts := pts }
    , resPairs, resPairsDeriv, uTriples)

/-- Finite-difference residual bounds for 1D second derivative. -/
def fdResidualBounds (u_minus : Float × Float) (u0 : Float × Float) (u_plus : Float × Float) (h :
  Float) : Float × Float :=
  let (lminus, hminus) := u_minus
  let (l0, h0) := u0
  let (lplus, hplus) := u_plus
  let num_lo := lplus - 2.0 * h0 + lminus
  let num_hi := hplus - 2.0 * l0 + hminus
  let s := Numbers.one / (h * h)
  (num_lo * s, num_hi * s)

/-- Generic param seeding (1D). -/
def seedParamsGeneric {α : Type} [Context α] : ParamStore α :=
  let W1 : Tensor α (.dim 16 (.dim 1 .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun _ => Tensor.scalar ((((i.val + 1 : Nat) : α)) *
      Numbers.pointone)))
  let eight : α := Numbers.four * Numbers.two
  let b1 : Tensor α (.dim 16 .scalar) :=
    Tensor.dim (fun i => Tensor.scalar (Numbers.pointfive * Numbers.pointone * ((((i.val : Nat) : α)
      - eight))))
  let Wm : Tensor α (.dim 16 (.dim 16 .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (if decide (i.val = j.val) then
      Numbers.one else (Numbers.pointfive * Numbers.pointone))))
  let bm : Tensor α (.dim 16 .scalar) := Tensor.dim (fun _ => Tensor.scalar Numbers.zero)
  let pointzeroone : α := Numbers.pointone * Numbers.pointone
  let W2 : Tensor α (.dim 1 (.dim 16 .scalar)) :=
    Tensor.dim (fun _ => Tensor.dim (fun j => Tensor.scalar (Numbers.pointone + pointzeroone *
      (((j.val : Nat) : α)))))
  let b2 : Tensor α (.dim 1 .scalar) := Tensor.dim (fun _ => Tensor.scalar Numbers.zero)
  let ps0 : ParamStore α := {}
  let ps1 := { ps0 with linearWB := ps0.linearWB.insert 1 ({ m := 16, n := 1,  w := W1, b := b1 }) }
  let ps2 := { ps1 with linearWB := ps1.linearWB.insert 3 ({ m := 16, n := 16, w := Wm, b := bm }) }
  let ps3 := { ps2 with linearWB := ps2.linearWB.insert 5 ({ m := 1,  n := 16, w := W2, b := b2 }) }
  ps3

/-- 2D generic param seeding (first layer 16x2). -/
def seedParamsGeneric2D {α : Type} [Context α] : ParamStore α :=
  let W1 : Tensor α (.dim 16 (.dim 2 .scalar)) :=
    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        let base := (((i.val + 1 : Nat) : α)) * (Numbers.pointfive * Numbers.pointone)
        let w := if decide (j.val = 0) then base * Numbers.two else base
        Tensor.scalar w))
  let eight : α := Numbers.four * Numbers.two
  let b1 : Tensor α (.dim 16 .scalar) :=
    Tensor.dim (fun i => Tensor.scalar (Numbers.pointfive * Numbers.pointone * ((((i.val : Nat) : α)
      - eight))))
  let Wm : Tensor α (.dim 16 (.dim 16 .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (if decide (i.val = j.val) then
      Numbers.one else (Numbers.pointfive * Numbers.pointone))))
  let bm : Tensor α (.dim 16 .scalar) := Tensor.dim (fun _ => Tensor.scalar Numbers.zero)
  let pointzeroone : α := Numbers.pointone * Numbers.pointone
  let W2 : Tensor α (.dim 1 (.dim 16 .scalar)) :=
    Tensor.dim (fun _ => Tensor.dim (fun j => Tensor.scalar (Numbers.pointone + pointzeroone *
      (((j.val : Nat) : α)))))
  let b2 : Tensor α (.dim 1 .scalar) := Tensor.dim (fun _ => Tensor.scalar Numbers.zero)
  let ps0 : ParamStore α := {}
  let ps1 := { ps0 with linearWB := ps0.linearWB.insert 1 ({ m := 16, n := 2,  w := W1, b := b1 }) }
  let ps2 := { ps1 with linearWB := ps1.linearWB.insert 3 ({ m := 16, n := 16, w := Wm, b := bm }) }
  let ps3 := { ps2 with linearWB := ps2.linearWB.insert 5 ({ m := 1,  n := 16, w := W2, b := b2 }) }
  ps3

/-- Generic input seeding for 1D. -/
def seedInputGeneric {α : Type} [Context α] (ps : ParamStore α) (x : α) (eps : α) : ParamStore α :=
  let x0 : Tensor α (.dim 1 .scalar) := Tensor.dim (fun _ => Tensor.scalar x)
  ps.seedLInfBall 0 x0 eps

/-- Generic input seeding for 2D. -/
def seedInputGeneric2D {α : Type} [Context α] (ps : ParamStore α) (x y : α) (eps : α) : ParamStore
  α :=
  let x0 : Tensor α (.dim 2 .scalar) :=
    Tensor.dim (fun i => Tensor.scalar (if decide (i.val = 0) then x else y))
  ps.seedLInfBall 0 x0 eps

end NN.Verification.PINN
