# Native TorchLean Verification

This folder contains runnable entry modules for workflows where the model is written in TorchLean,
compiled to the verifier IR, and checked by Lean bound propagation. Reusable workflow code belongs
under `NN/Verification/TorchLean`; reusable compiler, IR, CROWN, and certificate logic belongs under
`NN/Verification/TorchLean` or `NN/MLTheory/CROWN`.

Run the maintained entry points through the unified verifier:

```bash
lake exe verify -- torchlean-ibp
lake exe verify -- torchlean-crown-ops
lake exe verify -- torchlean-transformer-ibp
lake exe verify -- torchlean-mlp-workflow --dtype float
```

The `torchlean-ibp` implementation lives in `NN.Verification.TorchLean.IBPWorkflow`; the
`torchlean-crown-ops` implementation lives in `NN.Verification.TorchLean.CrownOpsWorkflow`.
The `torchlean-transformer-ibp` implementation lives in
`NN.Verification.TorchLean.TransformerIBPWorkflow`. The `torchlean-mlp-workflow` implementation
lives in `NN.Verification.TorchLean.MlpTrainVerifyWorkflow`.

The model training examples elsewhere in `NN/Examples/Models` cover ordinary CUDA eager training.
The workflows here are verifier workflows: after any training step, the parameters must be available
as Lean tensors so the verifier can compile and check the graph. Keep generated runtime logs,
checkpoints, and exported artifacts out of this directory; put them under an ignored `generated/`,
`outputs/`, or `_external/` directory if needed.
