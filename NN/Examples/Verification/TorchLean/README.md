# Native TorchLean Verification

This folder is for workflows where the model starts inside TorchLean, is compiled to the verifier
IR, and is checked by Lean side bound propagation. That is different from importing a certificate
from an external verifier. Here the point is to keep the path visible:

```text
TorchLean model
  -> parameter payload
  -> verifier IR graph
  -> IBP/CROWN-style bound propagation
  -> checked margin, bound, or diagnostic report
```

Reusable workflow code belongs under `NN/Verification/TorchLean`. Reusable CROWN/LiRPA data
structures, transfer rules, and proof files belong under `NN/MLTheory/CROWN`.

Run the maintained entry points through the unified verifier:

```bash
lake exe verify -- torchlean-ibp
lake exe verify -- torchlean-crown-ops
lake exe verify -- torchlean-transformer-ibp
lake exe verify -- torchlean-mlp-workflow --dtype float
```

Implementation map:

- `torchlean-ibp`: `NN.Verification.TorchLean.IBPWorkflow`
- `torchlean-crown-ops`: `NN.Verification.TorchLean.CrownOpsWorkflow`
- `torchlean-transformer-ibp`: `NN.Verification.TorchLean.TransformerIBPWorkflow`
- `torchlean-mlp-workflow`: `NN.Verification.TorchLean.MlpTrainVerifyWorkflow`

The `Proved/` subtree is for theorem-backed fragments of this path. The docs and code should keep
the levels of support separate: a runtime report, a Lean checker, and a theorem-backed compiler
fragment are different kinds of evidence.

The model training examples elsewhere in `NN/Examples/Models` cover ordinary eager, compiled, and
CUDA training. The workflows here are verifier workflows: after training, the parameters must be
available as Lean tensors so the verifier can compile and check the graph. Keep generated runtime
logs, checkpoints, and exported artifacts out of this directory; put them under an ignored
`generated/`, `outputs/`, or `_external/` directory if needed.
