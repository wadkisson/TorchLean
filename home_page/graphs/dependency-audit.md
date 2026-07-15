# TorchLean Dependency Audit

This report measures the Lean import graph: which source modules import which other modules. It is an architecture and maintenance artifact. It is not the runtime graph IR used to represent neural-network computations, and it is not a declaration-level proof dependency graph.

The audit is still useful for TorchLean because the library has intentional layer boundaries: specifications should not depend on runtime backends, reusable runtime code should not depend on examples, and broad imports should stay at deliberate umbrella or test-aggregation entrypoints.

Inspired by Li, Peng, Severini, and Shafto, "The Network Structure of Mathlib" (arXiv:2604.24797).

## Summary

- Modules: `1259`
- Import edges: `4043`
- Internal import edges: `3502`
- Public imports: `3692`
- Private imports: `351`
- Critical-path length over internal imports: `131`
- Findings: `0` (`0` errors, `0` warnings)
- Lean files: `1259`
- Lean source lines: `311582`
- Declaration headers: `12924`
- Theorem/lemma headers: `2443`

## Top Fan-In Modules

- `NN.API`: `80` incoming imports
- `NN.Spec.Core.Tensor`: `33` incoming imports
- `NN.Spec.Core.TensorOps`: `32` incoming imports
- `NN.Spec.Core.TensorReductionShape`: `30` incoming imports
- `NN.Spec.Layers.Activation`: `30` incoming imports
- `NN.MLTheory.CROWN.Graph`: `29` incoming imports
- `NN.Spec.Core.Context`: `28` incoming imports
- `NN.Floats.IEEEExec.Exec32`: `27` incoming imports
- `NN.Spec.Module.SpecModule`: `22` incoming imports
- `NN.Tensor`: `22` incoming imports

## Top Fan-Out Modules

- `NN.CI.Theory`: `105` imports
- `NN.CI.Foundation`: `99` imports
- `NN.CI.Floats`: `55` imports
- `NN.Proofs`: `43` imports
- `NN.CI.Runtime`: `27` imports
- `NN.Spec.Module`: `26` imports
- `NN.CI.Verification`: `25` imports
- `NN.Spec.Models`: `25` imports
- `NN.MLTheory.API`: `24` imports
- `blueprint.TorchLeanBlueprint.Guide.Ch3_Backend.Floats`: `23` imports

## Layer Edges

- `NN.Runtime` -> `NN.Runtime`: `385`
- `NN.Proofs` -> `NN.Proofs`: `328`
- `NN.API` -> `NN.API`: `323`
- `NN.Spec` -> `NN.Spec`: `309`
- `NN.Floats` -> `NN.Floats`: `271`
- `NN.MLTheory` -> `NN.MLTheory`: `229`
- `NN.Verification` -> `NN.Verification`: `176`
- `NN.Examples` -> `NN.Examples`: `157`
- `NN.CI` -> `NN.Spec`: `85`
- `NN.MLTheory` -> `NN.Spec`: `71`
- `NN.CI` -> `NN.MLTheory`: `68`
- `NN.Proofs` -> `NN.Spec`: `62`
- `NN.Tests` -> `NN.Tests`: `59`
- `NN.CI` -> `NN.Floats`: `55`
- `NN.Examples` -> `NN.API`: `55`
- `NN.Tests` -> `NN.Runtime`: `54`
- `Backend` -> `Backend`: `42`
- `NN.API` -> `NN.Spec`: `42`
- `NN.API` -> `NN.Runtime`: `39`
- `NN.CI` -> `NN.Proofs`: `37`

## Findings

No findings.
