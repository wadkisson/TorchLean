# NeuralFloat (`NN/Floats/NeuralFloat`)

This folder provides generic rounded arithmetic over `ℝ`. It is the reusable layer underneath
TorchLean's proof-oriented floating-point models: precision metadata, exponent formats, rounding
modes, rounded scalar values, and small error lemmas.

The folder deliberately does not commit to a particular executable kernel. It is for proofs about
rounded arithmetic as a mathematical object. Executable binary32 behavior lives in
`NN/Floats/IEEEExec/`; CUDA, ATen/libtorch, and native runtime behavior are separate trust-boundary
topics.

That distinction is important. A theorem about `NeuralFloat` says something like "this expression is
the result of applying a declared rounding model to real arithmetic." Claims about a particular GPU
kernel, BLAS call, or libtorch operator should cite the executable bridge or trust-boundary statement
that connects the backend to that rounding model.

## Files

- `Core.lean`: core datatypes, radix/exponent scaffolding, and the `NeuralFloat` record.
- `Metadata.lean`: TorchLean metadata such as training phase, named precisions, and mixed-precision
  presets.
- `Formats.lean`: Flocq-style exponent functions and format predicates (`FIX`, `FLX`, `FLT`).
- `Rounding.lean`: rounding modes and `neural_round` validation classes.
- `NF.lean`: `NF`, the rounded scalar type used as a `Spec.Core.Context` instance.
- `NNOps.lean`: rounded scalar functions used by neural-network specs.
- `Conversion.lean`: conversion helpers with format and error metadata.
- `ErrorBounds.lean`: reusable error-bound lemmas for composing proofs.

## When To Use It

Use `NeuralFloat`/`NF` when the theorem should be parametric in a rounded arithmetic model. Use
`NN.Floats.FP32` when the theorem specifically needs the binary32-sized rounded-real model. Use
`IEEE32Exec` when the object is executable IEEE-754 binary32 behavior with special values.

A useful mental model is:

- `ℝ`: ideal mathematical semantics.
- `NeuralFloat` / `NF`: a configurable rounded-real semantics suitable for proofs.
- `FP32`: the binary32-sized rounded-real specialization used by many neural-network statements.
- `IEEE32Exec`: executable IEEE-754-style behavior, including finite/special-value cases.
- runtime floats: what the backend actually computed, subject to the runtime boundary documented in
  `TRUST_BOUNDARIES.md`.

## What Belongs Here

This layer is a good home for generic rounding definitions, format predicates, conversion metadata,
and reusable error lemmas. It is not the place for CUDA performance claims, ATen kernel behavior, or
end-to-end model accuracy results. Those claims need to name the backend and the evidence being used.
