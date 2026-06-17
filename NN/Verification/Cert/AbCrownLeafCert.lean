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
# AbCrown Leaf Certificate

Alpha-beta-CROWN (AbCrown) leaf-certificate checker.

This module checks a small JSON certificate format (`abcrown_leaf_cert_v0_1`) exported by a Python
verification pipeline. It does **not** run bound propagation itself; instead it validates that:
- each leaf input box is nested inside the declared root input box, and
- each leaf contains a witness that refutes the unsafe threshold (`lb[i] > threshold[i]` for some
  `i`).

This is useful for:
- regression testing JSON export/import paths, and
- reviewer-friendly certificate validation workflows.

References:
- beta-CROWN paper (NeurIPS 2021): `https://arxiv.org/abs/2103.06624`
- alpha-beta-CROWN implementation: `https://github.com/Verified-Intelligence/alpha-beta-CROWN`

Run:
`lake exe verify -- abcrown-leaf [path/to/cert.json]`
-/

@[expose] public section


namespace NN.Verification.Cert.AbCrownLeafCert

open Lean
open Data
open NN.Verification.Json

/-- Bundled sample alpha-beta-CROWN leaf certificate. -/
def defaultCertPath : String :=
  "NN/Examples/Verification/AbCrown/sample_abcrown_leaf_cert_v0_1.json"

/--
Parse and validate a `abcrown_leaf_cert_v0_1` JSON certificate.

On failure this throws `IO.userError` with a brief message.
-/
def checkAbCrownLeafCertV01 (path : String) : IO Unit := do
  let topObj ← readJsonObjectFile path
  expectFormat topObj "abcrown_leaf_cert_v0_1"

  let rootObj ← expectFieldObj topObj "root" "top-level"
  let rootLo ← expectFieldFloatArray rootObj "lo" "root"
  let rootHi ← expectFieldFloatArray rootObj "hi" "root"

  let leaves ← expectFieldArray topObj "leaves" "top-level"
  if leaves.isEmpty then
    IO.println "[cert] Warning: leaves list is empty (nothing to check)"

  let mut okCount := 0
  let mut badCount := 0
  for leaf in leaves do
    let leafObj ← expectObj leaf "leaf"
    let lo ← expectFieldFloatArray leafObj "lo" "leaf"
    let hi ← expectFieldFloatArray leafObj "hi" "leaf"
    let lb ← expectFieldFloatArray leafObj "lb" "leaf"
    let thr ← expectFieldFloatArray leafObj "threshold" "leaf"

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

  IO.println s!"[cert] Checked {leaves.size} leaves: ok={okCount}, bad={badCount}"
  if badCount > 0 then
    throw <| IO.userError s!"Certificate failed checks for {badCount} leaves"

/--
CLI entry point: `lake exe verify -- abcrown-leaf [cert.json]`.

If no path is provided, checks a small bundled sample certificate under
`NN/Examples/Verification/AbCrown/`.
-/
def run (args : List String) : IO Unit := do
  let usage :=
    String.intercalate "\n" [
      "Usage:",
      "  lake exe verify -- abcrown-leaf [<path/to/cert.json>]",
      "",
      "If no path is provided, runs a small bundled sample cert:",
      s!"  {defaultCertPath}"
    ]

  if NN.API.CLI.hasHelp args then
    IO.println usage
    return

  let args := NN.API.CLI.dropDashDash args
  let (path, rest) ←
    match NN.API.CLI.takePositionalDefault args defaultCertPath with
    | .ok result => pure result
    | .error e => throw <| IO.userError s!"{e}\n\n{usage}"
  match NN.API.CLI.requireNoArgs rest with
  | .ok () => pure ()
  | .error e => throw <| IO.userError s!"{e}\n\n{usage}"
  checkAbCrownLeafCertV01 path

end NN.Verification.Cert.AbCrownLeafCert
