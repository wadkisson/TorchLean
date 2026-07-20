import VersoManual

open Verso.Genre Manual

#doc (Manual) "Where The Pieces Meet" =>
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

# The Numerical Story

The floating-point stack has three central levels:

```
NeuralFloat / NF
  generic radix, format, and rounding mathematics

FP32
  finite binary32 rounded-real proof semantics

IEEE32Exec
  executable 32-bit IEEE representation and operations
```

The runtime then adds CPU, CUDA, or external providers. A real-valued approximation theorem, a
half-ULP rounding theorem, an executable bit-pattern theorem, and a CUDA parity test are different
evidence.

This separation makes useful compositions possible. For example:

$$`
|F_{\mathrm{runtime}}(x)-f(x)|
\leq
|F_{\mathrm{runtime}}(x)-F_{\mathbb R}(x)|
+
|F_{\mathbb R}(x)-f(x)|.
`

The first term is numerical implementation error. The second is model approximation error. A
meaningful end-to-end result needs both, with compatible domains and semantics.

# The Scaling Story

TorchLean is not meant to replace every tuned numeric kernel with a slow Lean implementation.
Large models need industrial matrix multiplication, convolution, attention, FFT, and communication
libraries.

The architectural goal is:

```
one semantic operation graph
  -> several admissible kernel providers
  -> explicit contract and evidence per boundary
```

TorchLean can own graph structure, shapes, loss, optimizer meaning, and proof statements while a
provider supplies a fast value or local VJP. The assurance level may range from a proved internal
implementation to a checked or explicitly trusted external kernel.

Scaling and verification therefore meet at the backend contract, not by pretending that outsourced
numerics were executed inside the theorem prover.

# The Scientific-ML Story

A PINN or neural operator usually participates in a larger chain:

```
equation and domain
  -> discretization or simulator
  -> dataset
  -> model and training
  -> prediction artifact
  -> residual, invariant, or error certificate
  -> Lean checker and theorem
```

The neural network is only one part. Boundary conditions, quadrature, sampling coverage, simulator
accuracy, and interpolation between grid points can dominate the final claim.

TorchLean's role is to give each artifact a typed meaning and to make the accepted implication
precise. External computation can remain large; the checker and proposition should remain small
enough to audit.

# A Productive Development Loop

When adding an operation or model:

1. define the intended tensor and scalar semantics;
2. add the shape-checked public operation;
3. register forward and derivative behavior;
4. lower it to explicit IR when verification/export needs it;
5. add provider capsules for implemented runtimes;
6. state numerical and layout policies;
7. prove reusable semantic facts;
8. add executable positive and negative controls;
9. document unsupported paths and trust boundaries;
10. run the same small model through every claimed path.

Tests and proofs are complementary. Proofs establish universal propositions about formal objects.
Tests catch wiring, FFI, build, CLI, documentation, and platform regressions that are outside or not
yet covered by those propositions.

# What To Build Next

Several directions extend the same architecture:

- prove graph-wide exact-real enclosure from local interval transfers;
- expand proof-bearing reverse lowering and optimizer-error propagation;
- add narrower conformance checkers for native and external kernels;
- support more devices by registering real profiles and capsules, not only enum names;
- strengthen robustness and scientific certificate formats;
- connect quantization theory to packed runtime kernels;
- add model families through general tensor operations rather than private image or sequence types.

The criterion is not the number of features. A useful addition should reduce the distance between a
runnable model and a precise claim without hiding a new boundary.

# A Final Exercise

Choose one example from the model zoo and write down:

```
input and output shapes
parameter shapes
loss
data source and preprocessing
scalar semantics
execution mode
device and selected providers
forward and backward ownership
available theorem or checker
remaining trusted assumptions
```

Then run it with `--show-backend` and compare the report with your list. Any missing item is a
concrete documentation, logging, or verification task.

That habit is the central lesson of the guide: do not ask whether “the model” is verified as if it
were one indivisible thing. Ask which object carries the claim, which transformation produced it,
and which theorem or boundary connects it to what ran.

# References

- George et al., [*TorchLean: Formalizing Neural Networks in Lean*](https://arxiv.org/abs/2602.22631),
  2026.
- Boldo and Melquiond,
  [*Flocq: A Unified Library for Proving Floating-Point Algorithms in
  Coq*](https://doi.org/10.1109/ARITH.2011.40), 2011.
- George C. Necula, [*Proof-Carrying Code*](https://doi.org/10.1145/263699.263712), 1997.
- Odena et al., [*TensorFuzz*](https://proceedings.mlr.press/v97/odena19a.html), 2019.
- Liu et al., [*NNSmith*](https://arxiv.org/abs/2207.13066), 2022/2023.
