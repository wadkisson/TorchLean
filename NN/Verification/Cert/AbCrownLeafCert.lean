/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.CLI
public import NN.Verification.Util.Array
public import NN.Verification.Util.Json

/-!
# AbCrown Leaf Artifact

Alpha-beta-CROWN (AbCrown) leaf-artifact checker.

This module checks a small TorchLean JSON schema (`abcrown_leaf_artifact_v0_1`). Vanilla
alpha-beta-CROWN does not emit this TorchLean schema directly; use
`scripts/verification/abcrown/export_leaf_artifact.py` to convert terminal leaf/domain data from an
external verifier into the checked schema.

The checker does **not** run bound propagation itself. It validates only the finite claims present
in the artifact:
- each leaf input box is nested inside the declared root input box, and
- each leaf contains a witness that refutes the unsafe threshold (`lb[i] > threshold[i]` for some
  `i`).

This is useful for:
- regression testing JSON export/import paths, and
- reviewer-friendly validation of the leaf data that TorchLean actually checks.

References:
- beta-CROWN paper (NeurIPS 2021): `https://arxiv.org/abs/2103.06624`
- alpha-beta-CROWN implementation: `https://github.com/Verified-Intelligence/alpha-beta-CROWN`

Run:
`lake exe verify -- abcrown-leaf [path/to/artifact.json]`
-/

@[expose] public section


namespace NN.Verification.Cert.AbCrownLeafCert

open Lean
open Data
open NN.Verification.Json

/-- Bundled sample alpha-beta-CROWN-style leaf artifact. -/
def defaultArtifactPath : String :=
  "NN/Examples/Verification/AbCrown/sample_abcrown_leaf_artifact_v0_1.json"

/--
Parse and validate a `abcrown_leaf_artifact_v0_1` JSON artifact.

On failure this throws `IO.userError` with a brief message.
-/
def checkAbCrownLeafArtifact (path : String) : IO Unit := do
  let topObj ← readJsonObjectFile path
  expectFormat topObj "abcrown_leaf_artifact_v0_1"
  let inputDim ← expectFieldNat topObj "input_dim" "top-level"

  let rootObj ← expectFieldObj topObj "root" "top-level"
  let rootLo ← expectFieldFiniteFloatArray rootObj "lo" "root"
  let rootHi ← expectFieldFiniteFloatArray rootObj "hi" "root"
  if rootLo.size ≠ inputDim || rootHi.size ≠ inputDim then
    throw <| IO.userError
      s!"root dimension mismatch: input_dim={inputDim}, lo={rootLo.size}, hi={rootHi.size}"

  let leaves ← expectFieldArray topObj "leaves" "top-level"
  if leaves.isEmpty then
    IO.println "[artifact] Warning: leaves list is empty (nothing to check)"

  let mut okCount := 0
  let mut badCount := 0
  for leaf in leaves do
    let leafObj ← expectObj leaf "leaf"
    let lo ← expectFieldFiniteFloatArray leafObj "lo" "leaf"
    let hi ← expectFieldFiniteFloatArray leafObj "hi" "leaf"
    let lb ← expectFieldFiniteFloatArray leafObj "lb" "leaf"
    let thr ← expectFieldFiniteFloatArray leafObj "threshold" "leaf"
    if lo.size ≠ inputDim || hi.size ≠ inputDim then
      throw <| IO.userError
        s!"leaf dimension mismatch: input_dim={inputDim}, lo={lo.size}, hi={hi.size}"
    if lb.size ≠ thr.size then
      throw <| IO.userError
        s!"leaf lower-bound/threshold length mismatch: lb={lb.size}, threshold={thr.size}"

    let within := NN.Verification.Util.Array.boxWithin rootLo rootHi lo hi
    let verified :=
      match ← optionalFieldNat? leafObj "witness_idx" "leaf" with
      | some wi =>
          NN.Verification.Util.Array.refutesThresholdAt lb thr wi ||
            NN.Verification.Util.Array.refutesThreshold lb thr
      | none => NN.Verification.Util.Array.refutesThreshold lb thr
    if within && verified then
      okCount := okCount + 1
    else
      badCount := badCount + 1

  IO.println s!"[artifact] Checked {leaves.size} leaves: ok={okCount}, bad={badCount}"
  if badCount > 0 then
    throw <| IO.userError s!"Artifact failed checks for {badCount} leaves"

/--
CLI entry point: `lake exe verify -- abcrown-leaf [artifact.json]`.

If no path is provided, checks a small bundled sample artifact under
`NN/Examples/Verification/AbCrown/`.
-/
def run (args : List String) : IO Unit := do
  let usage :=
    String.intercalate "\n" [
      "Usage:",
      "  lake exe verify -- abcrown-leaf [<path/to/artifact.json>]",
      "",
      "If no path is provided, runs a small bundled sample artifact:",
      s!"  {defaultArtifactPath}"
    ]

  if NN.API.CLI.hasHelp args then
    IO.println usage
    return

  let args := NN.API.CLI.dropDashDash args
  let (path, rest) ←
    match NN.API.CLI.takePositionalDefault args defaultArtifactPath with
    | .ok result => pure result
    | .error e => throw <| IO.userError s!"{e}\n\n{usage}"
  match NN.API.CLI.requireNoArgs rest with
  | .ok () => pure ()
  | .error e => throw <| IO.userError s!"{e}\n\n{usage}"
  checkAbCrownLeafArtifact path

end NN.Verification.Cert.AbCrownLeafCert
