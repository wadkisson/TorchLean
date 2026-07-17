/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Artifacts.GridWorld.Position
public import NN.Runtime.RL.Artifacts.GridWorld.IO

/-!
# GridWorld Policy-Difference Artifacts

A `PolicyDiff` stores before/after greedy action maps for a fixed GridWorld. These files are small
run artifacts for visualization and regression checks, not a general RL dataset format.
-/

@[expose] public section

namespace Runtime.RL.Artifacts.GridWorld

open Lean
open Json
open Runtime.Training.JsonCodec

/--
Before/after greedy policy snapshots for a fixed `width × height` GridWorld.

`before` and `after` are flattened row-major arrays of action indices (0..3).
-/
structure PolicyDiff where
  width : Nat
  height : Nat
  before : Array Nat
  after : Array Nat
  notes : Array String := #[]
  deriving Inhabited

namespace PolicyDiff

def artifactLabel : String := "GridWorld policy artifact"

/--
Validate a `PolicyDiff` record (lengths and action ranges).

This is used defensively by IO readers/writers and widgets; it is scoped to IO
specification layer for policies.
-/
def validateE (p : PolicyDiff) : Except String Unit := do
  let expected := p.width * p.height
  if p.before.size != expected then
    throw s!"{artifactLabel}: `before` expected length {expected}, got {p.before.size}."
  if p.after.size != expected then
    throw s!"{artifactLabel}: `after` expected length {expected}, got {p.after.size}."
  if !(p.before.all (fun a => a < 4)) then
    throw s!"{artifactLabel}: `before` contains an out-of-range action (expected 0..3)."
  if !(p.after.all (fun a => a < 4)) then
    throw s!"{artifactLabel}: `after` contains an out-of-range action (expected 0..3)."

/-- JSON encoding for `PolicyDiff`. -/
def toJson (p : PolicyDiff) : Json :=
  Json.mkObj
    [ ("width", Json.num (JsonNumber.fromNat p.width))
    , ("height", Json.num (JsonNumber.fromNat p.height))
    , ("before", natArrayToJson p.before)
    , ("after", natArrayToJson p.after)
    , ("notes", stringArrayToJson p.notes)
    ]

/-- JSON decoding for `PolicyDiff`. -/
def ofJsonE (j : Json) : Except String PolicyDiff := do
  let o ←
    match j.getObj? with
    | .ok o => pure o
    | .error e => throw s!"{artifactLabel}: expected object: {e}"
  let widthJ ←
    match o.get? "width" with
    | some v => pure v
    | none => throw s!"{artifactLabel}: missing field `width`."
  let heightJ ←
    match o.get? "height" with
    | some v => pure v
    | none => throw s!"{artifactLabel}: missing field `height`."
  let beforeJ ←
    match o.get? "before" with
    | some v => pure v
    | none => throw s!"{artifactLabel}: missing field `before`."
  let afterJ ←
    match o.get? "after" with
    | some v => pure v
    | none => throw s!"{artifactLabel}: missing field `after`."
  let notesJ := (o.get? "notes").getD (.arr #[])
  let width ←
    match widthJ.getNat? with
    | .ok n => pure n
    | .error e => throw s!"{artifactLabel}: width expected Nat: {e}"
  let height ←
    match heightJ.getNat? with
    | .ok n => pure n
    | .error e => throw s!"{artifactLabel}: height expected Nat: {e}"
  let before ← natArrayOfJsonE (field := "before") beforeJ
  let after ← natArrayOfJsonE (field := "after") afterJ
  let notes :=
    match stringArrayOfJsonE (field := "notes") notesJ with
    | Except.ok xs => xs
    | Except.error _ => #[]
  let p : PolicyDiff := { width, height, before, after, notes }
  validateE p
  pure p

/-- Write a `PolicyDiff` JSON file to disk (creating parent directories if needed). -/
def writeJson (path : System.FilePath) (p : PolicyDiff) (pretty : Bool := true) : IO Unit :=
  writeValidatedJson validateE toJson path p pretty

/-- Read a `PolicyDiff` from a JSON file. -/
def readJson (path : System.FilePath) : IO PolicyDiff :=
  readDecodedJson artifactLabel ofJsonE path

end PolicyDiff

end Runtime.RL.Artifacts.GridWorld
