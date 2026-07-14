/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tests.Runtime.Floats.Utils
public import NN.Verification.Cert.Common
public import Lean.Data.Json

/-!
# CertificatePreconditions

Regression checks for the executable certificate boundary.

These tests cover side conditions that are easy for an external producer to get wrong: α-CROWN
slopes must be in `[0,1]`, binary elementwise bounds must have matching flattened dimensions, and
the true `log` relaxation must only be replayed on positive boxes.
-/

@[expose] public section

open Spec
open Lean
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open NN.Verification.Cert.Common

namespace Tests
namespace Floats
namespace CertificatePreconditions

def expect (msg : String) (b : Bool) : IO Unit := do
  unless b do
    throw <| IO.userError msg

def expectRejected {α : Type} (msg : String) (act : IO α) : IO Unit := do
  let mut rejected := false
  try
    let _ ← act
  catch _ =>
    rejected := true
  unless rejected do
    throw <| IO.userError msg

def parseJson! (s : String) : IO Json := do
  match Json.parse s with
  | .ok j => pure j
  | .error e => throw <| IO.userError s!"bad test JSON: {e}"

def flatBox (lo hi : Fin 2 → Float) : FlatBox Float :=
  { dim := 2
    lo := Spec.vectorTensor lo
    hi := Spec.vectorTensor hi }

def flatBox3 : FlatBox Float :=
  { dim := 3
    lo := Spec.vectorTensor (fun _ : Fin 3 => 0.0)
    hi := Spec.vectorTensor (fun _ : Fin 3 => 1.0) }

def inputNode (id dim : Nat) : _root_.NN.IR.Node :=
  { id := id, parents := [], kind := .input, outShape := .dim dim .scalar }

def addGraph : _root_.NN.IR.Graph :=
  { nodes := #[
      inputNode 0 2,
      inputNode 1 3,
      { id := 2, parents := [0, 1], kind := .add, outShape := .dim 2 .scalar }
    ] }

def logGraph : _root_.NN.IR.Graph :=
  { nodes := #[
      inputNode 0 2,
      { id := 1, parents := [0], kind := .log, outShape := .dim 2 .scalar }
    ] }

def run : IO Unit := do
  IO.println "certificate_preconditions: begin"

  let goodAlpha ← parseJson! "[0.0, 0.75]"
  let some _ ← parseAlphaVec? 2 goodAlpha
    | throw <| IO.userError "valid alpha vector was rejected"

  let badAlpha ← parseJson! "[2.0, 0.5]"
  expectRejected "alpha outside [0,1] was accepted" (parseAlphaVec? 2 badAlpha)

  let nonfiniteAlpha ← parseJson! "[1e999, 0.5]"
  expectRejected "non-finite alpha was accepted" (parseAlphaVec? 2 nonfiniteAlpha)

  let nonfiniteBox ← parseJson! "{\"lo\":[0.0,1e999],\"hi\":[1.0,2.0]}"
  expectRejected "non-finite interval certificate was accepted" (parseFlatBox? 2 nonfiniteBox)

  let b2 := flatBox (fun _ => 0.0) (fun _ => 1.0)
  let mismatchCert : Array (Option (FlatBox Float)) := #[some b2, some flatBox3, some b2]
  expect "binary elementwise dimension mismatch was accepted"
    (!(ibpNodePreconditionsOk addGraph mismatchCert 2))

  let nonPositive := flatBox (fun _ => 0.0) (fun _ => 1.0)
  let positive := flatBox (fun _ => 0.1) (fun _ => 2.0)
  expect "non-positive log input was accepted"
    (!(ibpNodePreconditionsOk logGraph #[some nonPositive, some nonPositive] 1))
  expect "positive log input was rejected"
    (ibpNodePreconditionsOk logGraph #[some positive, some positive] 1)

  let emptyStore : ParamStore Float := {}
  let nonPositiveRun := runIBP logGraph (emptyStore.seedInputBox 0 nonPositive)
  let positiveRun := runIBP logGraph (emptyStore.seedInputBox 0 positive)
  expect "IBP evaluated raw log across its nonpositive domain boundary"
    (nonPositiveRun[1]!.isNone)
  expect "IBP failed to evaluate raw log on a positive interval"
    (positiveRun[1]!.isSome)

  IO.println "certificate_preconditions: ok"

end CertificatePreconditions
end Floats
end Tests
