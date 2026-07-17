/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Artifacts.GridWorld.Position
public import NN.Runtime.RL.Artifacts.GridWorld.IO

/-!
# GridWorld Path-Difference Artifacts

A `PathDiff` stores before/after episode trajectories for a fixed GridWorld. It uses the shared
position codec and validates that every recorded `(row, col)` stays inside the declared grid.
-/

@[expose] public section

namespace Runtime.RL.Artifacts.GridWorld

open Lean
open Json
open Runtime.Training.JsonCodec

/--
Before/after episode path snapshots for a fixed `width × height` GridWorld.

Each position is stored as a pair `(row, col)` with `row < height` and `col < width`.
-/
structure PathDiff where
  width : Nat
  height : Nat
  before : Array (Nat × Nat)
  after : Array (Nat × Nat)
  notes : Array String := #[]
  deriving Inhabited

namespace PathDiff

def artifactLabel : String := "GridWorld path artifact"

/-- Validate a `PathDiff` record (positions are in bounds). -/
def validateE (p : PathDiff) : Except String Unit := do
  let inBounds (pos : Nat × Nat) : Bool :=
    pos.1 < p.height && pos.2 < p.width
  if !(p.before.all inBounds) then
    throw s!"{artifactLabel}: `before` contains an out-of-bounds position."
  if !(p.after.all inBounds) then
    throw s!"{artifactLabel}: `after` contains an out-of-bounds position."

/-- JSON encoding for `PathDiff`. -/
def toJson (p : PathDiff) : Json :=
  Json.mkObj
    [ ("width", Json.num (JsonNumber.fromNat p.width))
    , ("height", Json.num (JsonNumber.fromNat p.height))
    , ("before", posArrayToJson p.before)
    , ("after", posArrayToJson p.after)
    , ("notes", stringArrayToJson p.notes)
    ]

/-- JSON decoding for `PathDiff`. -/
def ofJsonE (j : Json) : Except String PathDiff := do
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
  let before ← posArrayOfJsonE (field := "before") beforeJ
  let after ← posArrayOfJsonE (field := "after") afterJ
  let notes :=
    match stringArrayOfJsonE (field := "notes") notesJ with
    | Except.ok xs => xs
    | Except.error _ => #[]
  let p : PathDiff := { width, height, before, after, notes }
  validateE p
  pure p

/-- Write a `PathDiff` JSON file to disk (creating parent directories if needed). -/
def writeJson (path : System.FilePath) (p : PathDiff) (pretty : Bool := true) : IO Unit :=
  writeValidatedJson validateE toJson path p pretty

/-- Read a `PathDiff` from a JSON file. -/
def readJson (path : System.FilePath) : IO PathDiff :=
  readDecodedJson artifactLabel ofJsonE path

end PathDiff

end Runtime.RL.Artifacts.GridWorld
