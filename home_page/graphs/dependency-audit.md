# TorchLean Dependency Audit

This report measures the Lean import graph: which source modules import which other modules. It is an architecture and maintenance artifact. It is not the runtime graph IR used to represent neural-network computations, and it is not a declaration-level proof dependency graph.

The audit is still useful for TorchLean because the library has intentional layer boundaries: specifications should not depend on runtime backends, reusable runtime code should not depend on examples, and broad imports should mostly stay at public entrypoints or tutorial surfaces.

Inspired by Li, Peng, Severini, and Shafto, "The Network Structure of Mathlib" (arXiv:2604.24797).

## Summary

- Modules: `1162`
- Import edges: `3947`
- Internal import edges: `3410`
- Public imports: `3629`
- Private imports: `318`
- Critical-path length over internal imports: `121`
- Findings: `0` (`0` errors, `0` warnings)
- Lean files: `1162`
- Lean source lines: `309083`
- Declaration headers: `12825`
- Theorem/lemma headers: `2467`

## Top Fan-In Modules

- `NN`: `80` incoming imports
- `NN.Spec.Core.Tensor`: `39` incoming imports
- `NN.Spec.Core.TensorOps`: `36` incoming imports
- `NN.MLTheory.CROWN.Graph`: `34` incoming imports
- `NN.Spec.Core.Context`: `32` incoming imports
- `NN.Spec.Core.TensorReductionShape`: `31` incoming imports
- `NN.Spec.Layers.Activation`: `31` incoming imports
- `NN.Floats.IEEEExec.Exec32`: `27` incoming imports
- `NN.Spec.Module.SpecModule`: `24` incoming imports
- `NN.MLTheory.CROWN.Core`: `23` incoming imports

## Top Fan-Out Modules

- `NN.CI.All`: `335` imports
- `NN.Entrypoint.Proofs`: `42` imports
- `NN.Spec.Module`: `28` imports
- `NN.Spec.Models`: `26` imports
- `NN.MLTheory.API`: `25` imports
- `NN.Examples.Zoo`: `24` imports
- `NN.Verification.CLI`: `20` imports
- `NN.Tests.Runtime.Floats.ModelsCheck`: `19` imports
- `NN.Proofs.Autograd.Coverage`: `18` imports
- `NN.Spec.Layers`: `18` imports

## Layer Edges

- `NN.API` -> `NN.API`: `365`
- `NN.Runtime` -> `NN.Runtime`: `348`
- `NN.Spec` -> `NN.Spec`: `315`
- `NN.Proofs` -> `NN.Proofs`: `282`
- `NN.MLTheory` -> `NN.MLTheory`: `226`
- `NN.Floats` -> `NN.Floats`: `168`
- `NN.Verification` -> `NN.Verification`: `162`
- `NN.Examples` -> `NN.Examples`: `145`
- `NN.CI` -> `NN.Spec`: `88`
- `NN.MLTheory` -> `NN.Spec`: `76`
- `NN.CI` -> `NN.MLTheory`: `71`
- `NN.Tests` -> `NN.Tests`: `61`
- `NN.Proofs` -> `NN.Spec`: `60`
- `NN.Tests` -> `NN.Runtime`: `52`
- `NN.Examples` -> `NN`: `50`
- `NN.CI` -> `NN.Floats`: `49`
- `NN.Entrypoint` -> `NN.Proofs`: `42`
- `NN.API` -> `NN.Runtime`: `39`
- `NN.Runtime` -> `NN.Spec`: `39`
- `NN.API` -> `NN.Spec`: `36`

## Findings

No findings.
