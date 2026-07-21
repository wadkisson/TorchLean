# TorchLean Dependency Audit

This report measures the Lean import graph: which source modules import which other modules. It is an architecture and maintenance artifact. It is not the runtime graph IR used to represent neural-network computations, and it is not a declaration-level proof dependency graph.

The audit is still useful for TorchLean because the library has intentional layer boundaries: specifications should not depend on runtime backends, reusable runtime code should not depend on examples, and broad imports should stay at deliberate umbrella or test-aggregation entrypoints.

Inspired by Li, Peng, Severini, and Shafto, "The Network Structure of Mathlib" (arXiv:2604.24797).

## Summary

- Modules: `1272`
- Import edges: `4090`
- Internal import edges: `3548`
- Public imports: `3760`
- Private imports: `330`
- Critical-path length over internal imports: `133`
- Findings: `0` (`0` errors, `0` warnings)
- Lean files: `1272`
- Lean source lines: `319107`
- Declaration headers: `13312`
- Theorem/lemma headers: `2630`

## Top Fan-In Modules

- `NN.API`: `73` incoming imports
- `NN.Spec.Core.TensorOps`: `34` incoming imports
- `NN.Spec.Core.Tensor`: `33` incoming imports
- `NN.Spec.Layers.Activation`: `31` incoming imports
- `NN.Spec.Core.TensorReductionShape`: `30` incoming imports
- `NN.MLTheory.CROWN.Graph`: `28` incoming imports
- `NN.Floats.IEEEExec.Exec32`: `27` incoming imports
- `NN.Spec.Core.Context`: `27` incoming imports
- `NN.Tensor`: `22` incoming imports
- `NN.Spec.Module.SpecModule`: `21` incoming imports

## Top Fan-Out Modules

- `NN.CI.Theory`: `105` imports
- `NN.CI.Foundation`: `98` imports
- `NN.CI.Floats`: `55` imports
- `NN.Proofs`: `43` imports
- `NN.CI.Runtime`: `27` imports
- `NN.CI.Verification`: `25` imports
- `NN.Spec.Models`: `25` imports
- `NN.Spec.Module`: `25` imports
- `NN.MLTheory.API`: `24` imports
- `NN.Verification.CLI`: `20` imports

## Layer Edges

- `NN.Runtime` -> `NN.Runtime`: `383`
- `NN.Proofs` -> `NN.Proofs`: `345`
- `NN.API` -> `NN.API`: `320`
- `NN.Spec` -> `NN.Spec`: `310`
- `NN.Floats` -> `NN.Floats`: `279`
- `NN.MLTheory` -> `NN.MLTheory`: `230`
- `NN.Verification` -> `NN.Verification`: `176`
- `NN.Examples` -> `NN.Examples`: `159`
- `NN.CI` -> `NN.Spec`: `84`
- `NN.MLTheory` -> `NN.Spec`: `74`
- `NN.Proofs` -> `NN.Spec`: `69`
- `NN.CI` -> `NN.MLTheory`: `68`
- `NN.Tests` -> `NN.Tests`: `61`
- `NN.Examples` -> `NN.API`: `56`
- `NN.CI` -> `NN.Floats`: `54`
- `NN.Tests` -> `NN.Runtime`: `51`
- `NN.API` -> `NN.Spec`: `43`
- `Backend` -> `Backend`: `42`
- `NN.API` -> `NN.Runtime`: `39`
- `NN.CI` -> `NN.Proofs`: `38`

## Findings

No findings.
