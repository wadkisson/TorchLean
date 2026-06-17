# TorchLean Dependency Audit

Inspired by Li, Peng, Severini, and Shafto, "The Network Structure of Mathlib" (arXiv:2604.24797).

## Summary

- Modules: `1143`
- Import edges: `3872`
- Internal import edges: `3346`
- Public imports: `3562`
- Private imports: `310`
- Critical-path length over internal imports: `121`
- Findings: `0` (`0` errors, `0` warnings)
- Lean files: `1143`
- Lean source lines: `297255`
- Declaration headers: `12144`
- Theorem/lemma headers: `2099`

## Top Fan-In Modules

- `NN`: `77` incoming imports
- `NN.Spec.Core.Tensor`: `39` incoming imports
- `NN.Spec.Core.TensorOps`: `36` incoming imports
- `NN.MLTheory.CROWN.Graph`: `34` incoming imports
- `NN.Spec.Core.Context`: `32` incoming imports
- `NN.Spec.Core.TensorReductionShape`: `31` incoming imports
- `NN.Spec.Layers.Activation`: `31` incoming imports
- `NN.Floats.IEEEExec.Exec32`: `27` incoming imports
- `NN.API.Public`: `24` incoming imports
- `NN.Spec.Module.SpecModule`: `24` incoming imports

## Top Fan-Out Modules

- `NN.CI.All`: `333` imports
- `NN.Entrypoint.Proofs`: `42` imports
- `NN.Spec.Module`: `28` imports
- `NN.Spec.Models`: `26` imports
- `NN.Examples.Zoo`: `23` imports
- `NN.MLTheory.API`: `23` imports
- `NN.Tests.Runtime.Floats.ModelsCheck`: `19` imports
- `NN.Proofs.Autograd.Coverage`: `18` imports
- `NN.Spec.Layers`: `18` imports
- `NN.Entrypoint.Widgets`: `17` imports

## Layer Edges

- `NN.Runtime` -> `NN.Runtime`: `348`
- `NN.API` -> `NN.API`: `347`
- `NN.Spec` -> `NN.Spec`: `313`
- `NN.Proofs` -> `NN.Proofs`: `276`
- `NN.MLTheory` -> `NN.MLTheory`: `222`
- `NN.Floats` -> `NN.Floats`: `168`
- `NN.Verification` -> `NN.Verification`: `162`
- `NN.Examples` -> `NN.Examples`: `136`
- `NN.CI` -> `NN.Spec`: `88`
- `NN.MLTheory` -> `NN.Spec`: `76`
- `NN.CI` -> `NN.MLTheory`: `69`
- `NN.Tests` -> `NN.Tests`: `60`
- `NN.Proofs` -> `NN.Spec`: `57`
- `NN.Tests` -> `NN.Runtime`: `51`
- `NN.CI` -> `NN.Floats`: `49`
- `NN.Examples` -> `NN`: `49`
- `NN.Entrypoint` -> `NN.Proofs`: `42`
- `NN.API` -> `NN.Runtime`: `39`
- `NN.Runtime` -> `NN.Spec`: `37`
- `NN.API` -> `NN.Spec`: `36`

## Findings

No findings.
