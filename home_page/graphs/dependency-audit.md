# TorchLean Dependency Audit

Inspired by Li, Peng, Severini, and Shafto, "The Network Structure of Mathlib" (arXiv:2604.24797).

## Summary

- Modules: `1033`
- Import edges: `3573`
- Internal import edges: `3014`
- Public imports: `3228`
- Private imports: `345`
- Critical-path length over internal imports: `86`
- Findings: `0` (`0` errors, `0` warnings)
- Lean files: `1033`
- Lean source lines: `280863`
- Declaration headers: `10951`
- Theorem/lemma headers: `1970`

## Top Fan-In Modules

- `NN`: `64` incoming imports
- `NN.Spec.Core.Tensor`: `39` incoming imports
- `NN.Spec.Core.TensorOps`: `36` incoming imports
- `NN.Spec.Core.Context`: `32` incoming imports
- `NN.Spec.Core.TensorReductionShape`: `31` incoming imports
- `NN.Spec.Layers.Activation`: `31` incoming imports
- `NN.MLTheory.CROWN.Graph`: `29` incoming imports
- `NN.Floats.IEEEExec.Exec32`: `27` incoming imports
- `NN.Spec.Module.SpecModule`: `24` incoming imports
- `NN.MLTheory.CROWN.Core`: `23` incoming imports

## Top Fan-Out Modules

- `NN.CI.All`: `333` imports
- `NN.Entrypoint.Proofs`: `42` imports
- `NN.Spec.Module`: `28` imports
- `NN.Spec.Models`: `26` imports
- `NN.Examples.Zoo`: `24` imports
- `NN.MLTheory.API`: `23` imports
- `NN.Tests.Runtime.Floats.ModelsCheck`: `19` imports
- `NN.Spec.Layers`: `18` imports
- `NN.Entrypoint.Widgets`: `17` imports
- `NN.Floats.IEEEExec`: `17` imports

## Layer Edges

- `NN.Runtime` -> `NN.Runtime`: `346`
- `NN.Spec` -> `NN.Spec`: `313`
- `NN.Proofs` -> `NN.Proofs`: `261`
- `NN.MLTheory` -> `NN.MLTheory`: `221`
- `NN.Floats` -> `NN.Floats`: `168`
- `NN.Examples` -> `NN.Examples`: `142`
- `NN.API` -> `NN.API`: `140`
- `NN.CI` -> `NN.Spec`: `88`
- `NN.Verification` -> `NN.Verification`: `84`
- `NN.MLTheory` -> `NN.Spec`: `76`
- `NN.CI` -> `NN.MLTheory`: `69`
- `NN.Proofs` -> `NN.Spec`: `57`
- `NN.Tests` -> `NN.Tests`: `54`
- `NN.CI` -> `NN.Floats`: `49`
- `NN.Examples` -> `NN.API`: `48`
- `NN.Examples` -> `NN`: `45`
- `NN.Entrypoint` -> `NN.Proofs`: `42`
- `NN.Tests` -> `NN.Runtime`: `42`
- `NN.Runtime` -> `NN.Spec`: `37`
- `NN.API` -> `NN.Spec`: `36`

## Findings

No findings.
