# TorchLean Dependency Audit

This report measures the Lean import graph: which source modules import which other modules. It is an architecture and maintenance artifact. It is not the runtime graph IR used to represent neural-network computations, and it is not a declaration-level proof dependency graph.

The audit is still useful for TorchLean because the library has intentional layer boundaries: specifications should not depend on runtime backends, reusable runtime code should not depend on examples, and broad imports should mostly stay at public entrypoints or tutorial surfaces.

Inspired by Li, Peng, Severini, and Shafto, "The Network Structure of Mathlib" (arXiv:2604.24797).

## Summary

- Modules: `1178`
- Import edges: `3950`
- Internal import edges: `3415`
- Public imports: `3631`
- Private imports: `319`
- Critical-path length over internal imports: `121`
- Findings: `0` (`0` errors, `0` warnings)
- Lean files: `1178`
- Lean source lines: `311591`
- Declaration headers: `13146`
- Theorem/lemma headers: `2437`

## Top Fan-In Modules

- `NN`: `80` incoming imports
- `NN.Spec.Core.Tensor`: `34` incoming imports
- `NN.MLTheory.CROWN.Graph`: `34` incoming imports
- `NN.Spec.Core.TensorOps`: `32` incoming imports
- `NN.Spec.Layers.Activation`: `31` incoming imports
- `NN.Spec.Core.TensorReductionShape`: `30` incoming imports
- `NN.Spec.Core.Context`: `29` incoming imports
- `NN.Floats.IEEEExec.Exec32`: `27` incoming imports
- `NN.Spec.Module.SpecModule`: `24` incoming imports
- `NN.Verification.TorchLean.Compile`: `22` incoming imports

## Top Fan-Out Modules

- `NN.CI.All`: `327` imports
- `NN.Entrypoint.Proofs`: `43` imports
- `NN.Spec.Module`: `28` imports
- `NN.Spec.Models`: `26` imports
- `NN.MLTheory.API`: `25` imports
- `NN.Examples.Zoo`: `23` imports
- `NN.Verification.CLI`: `20` imports
- `NN.API.Public.Facade.Base.Runtime`: `18` imports
- `NN.Backend`: `18` imports
- `NN.Proofs.Autograd.Coverage`: `18` imports

## Layer Edges

- `NN.API` -> `NN.API`: `364`
- `NN.Runtime` -> `NN.Runtime`: `346`
- `NN.Spec` -> `NN.Spec`: `315`
- `NN.Proofs` -> `NN.Proofs`: `285`
- `NN.MLTheory` -> `NN.MLTheory`: `222`
- `NN.Floats` -> `NN.Floats`: `168`
- `NN.Verification` -> `NN.Verification`: `163`
- `NN.Examples` -> `NN.Examples`: `145`
- `NN.CI` -> `NN.Spec`: `88`
- `NN.MLTheory` -> `NN.Spec`: `71`
- `NN.CI` -> `NN.MLTheory`: `69`
- `NN.Proofs` -> `NN.Spec`: `62`
- `NN.Tests` -> `NN.Tests`: `58`
- `NN.Tests` -> `NN.Runtime`: `54`
- `NN.Examples` -> `NN`: `50`
- `NN.CI` -> `NN.Floats`: `49`
- `NN.Entrypoint` -> `NN.Proofs`: `43`
- `Backend` -> `Backend`: `42`
- `NN.API` -> `NN.Runtime`: `39`
- `NN.API` -> `NN.Spec`: `36`

## Findings

No findings.
