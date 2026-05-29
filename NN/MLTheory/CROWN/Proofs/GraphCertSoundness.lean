/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.Semantics
public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.CertificateStep
public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.IntervalLemmas
public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.NonlinearOps
public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.Main

/-!
# Graph IBP Certificate Soundness

Proof-level soundness for graph-dialect IBP certificates over `ℝ`.

The main theorem is `CertSoundness.cert_encloses_semantics`: if the certificate matches the local
IBP step at every node, and the semantic values are locally consistent with the graph evaluator,
then every certified box encloses the corresponding semantic value.
-/
