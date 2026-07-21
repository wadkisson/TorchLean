import VersoManual

open Verso.Genre Manual

#doc (Manual) "Conclusion" =>
%%%
tag := "conclusion"
%%%


We began with a two-input regression model:

$$`
F_\theta(x)
=
W_2\operatorname{ReLU}(W_1x+b_1)+b_2.
`

It became several related objects:

- a shape-checked model declaration;
- a seeded parameter pack;
- an executable training run;
- an eager tape or compiled derivative graph;
- a canonical operation IR;
- a real-valued specification;
- a rounded or executable floating-point interpretation;
- a backend plan and capsule audit;
- a verification target or checked certificate.

The objects remain separate because they answer separate questions. Their value comes from the
explicit maps between them.

# Reproduce The Path

These commands retrace the central examples:

```
lake exe torchlean quickstart_tensors
lake exe torchlean quickstart_autograd
lake exe torchlean quickstart_mlp \
  --device cpu --steps 200 --seed 2026
lake exe torchlean graphspec --device cpu --backend eager
lake exe torchlean one_semantic_universe
lake exe torchlean float32_modes
lake exe torchlean numerical_certificate
```

Together they show:

1. one shape-indexed tensor API at several scalar semantics;
2. explicit VJPs, Jacobians, Hessians, and parameter gradients;
3. a complete training and prediction run;
4. structured model lowering;
5. one IR interpreted for values and interval bounds;
6. host Float versus executable binary32 forward/backward behavior;
7. a graph-level numerical certificate workflow.

Successful output is not the end of the argument. It tells us which artifact to inspect next.

# A Claim, Written Carefully

Suppose a report says:

> The trained model is robust on a box of inputs.

TorchLean encourages us to expand that sentence:

1. *Which model?* Identify architecture, parameter artifact, and graph payload.
2. *Which input box?* Give lower and upper tensors with checked shapes.
3. *Which robustness property?* State the output inequality or class-margin condition.
4. *Which scalar semantics?* Exact real, finite FP32, executable IEEE32, or a native runtime.
5. *Which method?* IBP, CROWN, α,β-CROWN artifact replay, branch-and-bound, or another checker.
6. *Which theorem?* Name the proposition obtained when the checker accepts.
7. *Which boundary remains trusted?* External search, parser, backend kernel, compiler, or hardware.

The longer sentence is not bureaucratic. It is the difference between a result that can be reused
and one that cannot be audited when the model or backend changes.

# What TorchLean Owns

TorchLean can own, within Lean:

- tensor shapes and layer composition;
- mathematical operator specifications;
- graph well-formedness and supported evaluation;
- primitive and graph-level autograd theorems;
- generic rounding and finite FP32 mathematics;
- executable IEEE32 reference algorithms and proved bridges;
- optimizer laws under their stated hypotheses;
- soundness of implemented certificate checkers;
- policy gates that reject missing or inadmissible evidence.

The exact scope is determined by each declaration. An import path or chapter heading does not
upgrade a partial theorem into a universal one.

# What Remains A Boundary

Real training also uses:

- native CUDA kernels;
- cuBLAS, cuFFT, and other accelerator libraries;
- LibTorch and ATen;
- C/C++ compilers, drivers, and hardware;
- Python export and conversion scripts;
- external verifier search;
- datasets and scientific simulators.

Wrapping these components in a Lean function does not prove them correct. Kernel capsules record
provider, device, shape/layout contracts, numerical policy, VJP ownership, and evidence. External
artifacts cross parsers and checkers. The remaining trust is named rather than disappearing into
the phrase “verified in Lean.”
