/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Cert.AlphaCROWN
public import NN.MLTheory.CROWN.Graph
public import NN.Spec.Core.Utils
public import NN.Verification.Cert.IBPNodeCert
public import Lean.Data.Json

/-!
# CROWNNodeCert

Per-node α-CROWN certificate checking (graph dialect).

This mirrors `NN.Verification.IBPNodeCert`, but for affine bounds produced by a CROWN/DeepPoly pass
with optional α-parameters for the ReLU lower relaxation (α-CROWN).

Certificate JSON format:

```json
{
  "ctx": { "inputId": 0, "inputDim": 2 },
  "ibp": [ null | { "lo": [...], "hi": [...] }, ... ],
  "crown": [
    null |
      {
        "loA": [[...], ...], "loC": [...],
        "hiA": [[...], ...], "hiC": [...]
      },
    ...
  ],
  "alpha": [ null | [...], ... ] // optional per-node ReLU α vector
}
```

Trust boundary notes:
- The certificate is untrusted; we accept it only if Lean recomputation matches (within a float
  tolerance due to JSON decimal serialization).
- Transcendental relaxations are checked only via structural recomputation, not via a formal
  "libm is correct" guarantee.
-/

@[expose] public section


namespace NN.Verification.CROWNNodeCert

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
The helpers below are the JSON-facing boundary for the CROWN certificate checkers.  They parse
the artifact, compare decimal-serialized floats with an explicit tolerance, and check parent/shape
requirements before invoking the semantic checker.
-/

/-- In-memory representation of a node-wise α-CROWN certificate read from JSON. -/
abbrev CROWNNodeCertificate := CROWNNodeCoreCertificate

/-- Read a CROWN node certificate from JSON on disk. -/
def readCROWNNodeCertificate (g : Graph) (path : String) : IO CROWNNodeCertificate := do
  let topObj ← readJsonObjectFile path
  parseCROWNNodeCoreCertificate g topObj

/-- Check the local CROWN enclosure condition for one node against a certificate entry. -/
def checkCROWNNode (g : Graph) (ps : ParamStore Float)
    (certIbp : Array (Option (FlatBox Float)))
    (certAlpha : Array (Option (FlatVec Float)))
    (certCrown : Array (Option (FlatAffineBounds Float)))
    (ctx : AffineCtx)
    (id : Nat) (tol : Float) : IO Bool := do
  let computed? :=
    alphaCrownStepNode? (α := Float) g.nodes ps certIbp certAlpha certCrown ctx id
  checkCROWNLikeNode "CROWNNodeCert" g certIbp certCrown ctx id tol computed?

/--
Check a per-node α-CROWN certificate against Lean's propagation rules.

Returns `true` iff every supplied IBP box is locally recomputed by Lean and every node's affine
bounds agree (within `tol`) with Lean's CROWN step.
-/
def checkCROWNNodeCertificate (g : Graph) (ps : ParamStore Float) (path : String) (tol : Float := 1e-4) : IO Bool :=
  do
  let cert ← readCROWNNodeCertificate g path
  let mut ok := true
  for id in [0:g.nodes.size] do
    let okIbp ← NN.Verification.IBPNodeCert.checkIBPNode g ps cert.ibp id tol
    let okCrown ← checkCROWNNode g ps cert.ibp cert.alpha cert.crown cert.ctx id tol
    ok := ok && okIbp && okCrown
  if ok then
    IO.println "[CROWNNodeCert] artifact replay matched Lean IBP and α-CROWN steps."
  pure ok

end NN.Verification.CROWNNodeCert
