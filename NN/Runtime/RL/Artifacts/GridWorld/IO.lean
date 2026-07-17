/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Training.Log

/-!
# GridWorld Artifact JSON I/O

Shared validate/write/read helpers used by `PathDiff` and `PolicyDiff` artifacts.
-/

@[expose] public section

namespace Runtime.RL.Artifacts.GridWorld

open Lean
open Json

/-- Write a validated JSON artifact, creating parent directories when needed. -/
def writeValidatedJson {α : Type}
    (validateE : α → Except String Unit) (toJson : α → Json)
    (path : System.FilePath) (value : α) (pretty : Bool := true) : IO Unit := do
  match validateE value with
  | .error e => throw <| IO.userError e
  | .ok () =>
      match path.parent with
      | some parent => IO.FS.createDirAll parent
      | none => pure ()
      let j := toJson value
      let s := if pretty then Json.pretty j else Json.compress j
      IO.FS.writeFile path (s ++ "\n")

/-- Read a JSON artifact from disk and decode it with field-aware errors. -/
def readDecodedJson {α : Type}
    (label : String) (ofJsonE : Json → Except String α)
    (path : System.FilePath) : IO α := do
  let s ← IO.FS.readFile path
  let j ←
    match Json.parse s with
    | .ok j => pure j
    | .error e => throw <| IO.userError s!"{label}: parse error: {e}"
  match ofJsonE j with
  | .ok v => pure v
  | .error e => throw <| IO.userError e

end Runtime.RL.Artifacts.GridWorld
