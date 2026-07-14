import VersoManual

open Verso.Genre Manual

#doc (Manual) "Backend Selection and Trust" =>
%%%
tag := "backend-selection"
%%%

Running a model on the CPU and running it on a GPU should not create two different model
definitions. The layers, parameters, tensor shapes, and loss remain the same. What changes is the
program that supplies each numeric value and gradient.

That distinction is easy to miss once an execution path becomes fast. A fused attention call may
replace several graph nodes. cuBLAS may perform a matrix multiply. LibTorch may choose among several
scaled-dot-product-attention implementations. These are valuable implementation choices, but none
of them is a theorem merely because the call was made from Lean.

TorchLean therefore records backend choices separately from model semantics.

# Four Choices That Should Not Be Confused

The public runtime has four related decisions:

1. *Execution mode* chooses eager or compiled TorchLean execution. Eager mode records operations
   as they run; compiled mode reuses a lowered graph.
2. *Device* chooses the hardware target, currently CPU or CUDA for implemented paths.
3. *Provider* names the implementation family for an operation, such as TorchLean, native CUDA,
   cuBLAS, cuFFT, or LibTorch.
4. *Trust policy* decides which kinds of evidence are acceptable for this run.

The first choice does not determine the others. Compiled execution is not automatically CUDA, and
CUDA does not automatically mean LibTorch. A run may use TorchLean's compiled graph while selected
matrix operations are supplied by cuBLAS and other operations use native TorchLean CUDA kernels.

At the semantic level, the division of responsibility is:

- the specification defines the mathematical operation;
- the graph records which operation occurs and with which payload;
- the runtime owns storage, scheduling, and execution state;
- a backend provider computes a selected value or local VJP;
- a checker or theorem carries the correctness claim.

# Kernel Capsules

Suppose a graph contains scaled dot-product attention. TorchLean currently knows several possible
implementations: a composed TorchLean expression, a native fused CUDA implementation, a LibTorch
forward bridge, and a LibTorch autograd bridge. They share TorchLean's hard boolean-mask semantics,
but differ in gradient ownership, runtime status, and trust boundary.

A `KernelCapsule` records those differences:

```
structure KernelCapsule where
  name : String
  op : BackendOp
  provider : Provider
  device : Device
  specName : String
  trustLevel : TrustLevel
  supportsForward : Bool
  vjpMode : VJPMode
  runtimeSupport : RuntimeSupport
  shapeContract : ContractDescriptor
  layoutContract : ContractDescriptor
  valueContract : ContractDescriptor
  vjpContract : ContractDescriptor
```

The capsule is not generated from a successful run. It is declared before planning and registered
with the backend. The planner may select it only when the requested device, provider preference,
gradient mode, and trust policy allow it. This makes an unsupported path a planning error rather
than an accidental change of semantics.

The complete installation and platform guide includes a
[worked capsule example](https://lean-dojo.github.io/TorchLean/installation/#what-a-kernel-capsule-looks-like).

# Evidence Is Not A Label

Each capsule has four obligations: shape, layout, forward value, and VJP. An obligation may point to
a Lean theorem, an executable checker, a native source symbol, a fuzz oracle, or an explicit trusted
boundary.

The obligation itself is a structured `ContractClaim`: shape safety, compatibility with a named
tensor layout, refinement of a forward specification, or refinement of a VJP specification. The
human-readable summary does not replace that claim, and provenance such as a source path does not
replace evidence.

Before planning, TorchLean checks that each descriptor has the right obligation kind and operation,
and that a VJP descriptor names the capsule's declared VJP mode. Thus a value theorem placed in the
shape field is rejected rather than counted as shape evidence. The registry author must still ensure
that the proposition carried by theorem or checker evidence is the intended formalization of the
structured claim; a string such as `specName` is documentation, not a proof link.

The distinction matters:

- *theorem evidence* contains a Lean proposition and its proof term;
- *checker evidence* contains an accepted result and a soundness proof for its Lean proposition;
- *runtime-guard evidence* records validation performed at an execution boundary;
- *test evidence* records a regression or differential test suite;
- *fuzz evidence* records sampled differential testing, not a universal statement;
- *trusted external evidence* names code whose correctness is assumed for the claim;
- *missing evidence* prevents acceptance under the normal strict policies.

Source paths and native symbols are provenance rather than evidence. They can identify the code that
ran, but their existence says nothing about its numerical correctness. Strict acceptance allows only
proof-bearing theorem or checker evidence; maintained runtime profiles may separately allow named
guards and test coverage.

# A Real Attention Theorem

TorchLean's FlashAttention specification illustrates the proof boundary. The following are genuine
Lean declarations:

```
#check Spec.flashAttention_eq_scaledDotProductAttention
#check Spec.flashAttentionBackward_eq_scaledDotProductAttentionBackward
```

They prove that the fused *Lean specification* has the same forward and backward denotation as
TorchLean's standard scaled-dot-product-attention specification. They are useful for semantic graph
rewrites and for stating the contract expected of a fused implementation.

They do not inspect PTX, CUDA machine code, cuBLAS, a GPU driver, or LibTorch. The native CUDA
attention capsule consequently uses runtime guards, regression tests, source provenance, and trust
level `checked`. A proof of
the native implementation would need an additional refinement theorem, or a replay checker with a
soundness theorem, connecting the executable result to the specification under explicit Float32,
layout, compiler, and hardware assumptions.

# Forward And Backward Ownership

Inference asks for a forward value. Training asks for more: the value must remain connected to the
derivative rule used by the optimizer.

TorchLean distinguishes four VJP modes:

- `none`: no gradient is requested;
- `torchLeanTape`: TorchLean records the node and applies its local backward rule;
- `backendVJP`: TorchLean owns the tape, while a named backend kernel computes the local VJP;
- `externalAutograd`: the external runtime owns the local autograd computation.

The preferred external-forward design is therefore precise: a provider may compute a fast forward
value, TorchLean records the same operation on its tape, and TorchLean applies the backward rule.
This requires enough forward information to be retained for that rule. If the bridge cannot provide
it, the implementation must fall back or expose a larger trust boundary.

External autograd is allowed as an explicit option, but it is a different contract. In that mode the
claim depends on the external forward implementation and its derivative implementation.

No-grad sessions request `none` automatically. During training, a differentiable operation cannot
select a forward-only capsule. Seeded random sources are the deliberate exception: they create
non-differentiable values, so they do not need a local VJP of their own.

# Boolean Attention Masks

TorchLean gives boolean attention masks one semantics across specifications and runtimes. A blocked
entry contributes exactly zero to the softmax numerator, as if its score were negative infinity.
Native CUDA skips blocked entries, while the LibTorch bridge passes a boolean mask directly to
scaled dot-product attention. Additive score biases remain a separate operation.

# Acceptance Gates

Planning and acceptance are separate steps. Planning finds capsules for graph operations. Auditing
turns their contract fields into obligation reports. An `AcceptancePolicy` then decides whether the
run may proceed.

The implementation follows one explicit path:

1. `Target` describes the operating system, architecture, accelerator, and compiled features.
2. `Availability` states the devices and providers declared for planning. Eager execution performs
   the separate linked-library and hardware probes before launching a kernel.
3. `Registry` supplies compatible capsules, and `Planner` chooses one `PlannedKernel` for each graph
   operation. Together these choices form an `ExecutionPlan`.
4. `Audit` and `Recheck` expose the selected evidence and any obligations that must be discharged
   again for this run.
5. `Gate` applies the requested acceptance policy. Eager execution receives an `AcceptedKernel` for
   each operation. Graph lowering can similarly produce an `AcceptedGraphPlan` for a later graph
   executor; the current eager runtime does not pretend that this data-level graph plan is executable.
6. `Report` renders the providers, devices, trust levels, and recheck dispositions for logs and
   benchmark records.

These are Lean data structures rather than an informal convention between command-line flags. The
eager runtime consumes the accepted per-operation value and records the capsule it actually used.
Inspection tools can retain rejected graph plans and explain why they failed.

`AcceptedKernel` and `AcceptedGraphPlan` carry the equality proof that their policy gate returned
`accepted`; they are not records that a caller can populate while omitting the gate result.

The strict policy allows only proof-bearing theorem and checker evidence. Runtime policies can allow
guards, regression tests, fuzzing, or named trusted boundaries explicitly. The gate
itself has a small Lean theorem:

```
#check ExecutionAudit.gate_eq_accepted_iff_gateFailures_eq_nil
```

This theorem proves a fact about TorchLean's acceptance function: acceptance is equivalent to an
empty failure list. It does not prove the kernels mentioned in the audit. The distinction between a
proved policy function and an assumed native implementation should remain visible in every backend
claim.

# Public Configuration

The model API stays independent of these implementation details:

```
def trainerFor (backend : TorchLean.Runtime.Backend) :=
  Trainer.new mkModel
    { task := .regression
      optimizer := optim.adam { lr := 0.01 }
      dtype := .float32
      backend := backend }

let eagerTrainer := trainerFor .eager
let compiledTrainer := trainerFor .compiled
```

Device and provider selection live in runtime configuration and command-line options. Backend
selection does not change the model's public forward function. This is similar to the separation
between calling a PyTorch model and wrapping it with `torch.compile`: compilation changes
execution, not the mathematical intention of the model.

# Choosing A Path

- Use *CPU eager* while inspecting individual operations and autograd behavior.
- Use *CPU compiled* for repeated execution of a supported fixed graph.
- Use *CUDA* when the required Float32 operations have registered executable capsules.
- Enable *LibTorch* only for the external kernels that have explicit bridges and capsules.
- Use *graph export* when the next consumer is a verifier, checker, or code generator.

Unsupported devices and providers should fail with a readable message. Silent CPU fallback is not
acceptable when the user requested an accelerator, because it makes benchmark and deployment claims
ambiguous.

# Reading Backend Claims

These statements have different strengths:

- "the example ran on CUDA" reports an execution path;
- "CUDA matched the CPU reference on this test suite" reports finite parity evidence;
- "the fused attention spec equals standard attention" cites a Lean semantic theorem;
- "the native attention kernel implements the fused spec" requires a native refinement argument;
- "the LibTorch result is correct" depends on the explicitly named LibTorch boundary unless a
  stronger checker or theorem covers it.

A backend report is useful because it records the selected provider and evidence. It should be
published beside benchmark results, but it should never be described as if every evidence row were
already a proof.

# Where To Continue

Read [Execution Modes and Runners](Runtime___-Autograd___-and-Interop/Execution-Modes-and-Runners/)
for the public runtime API. Read
[GPU and CUDA Boundaries](Floating-Point-and-Native-Boundaries/GPU-and-CUDA-Boundaries/)
for the native implementation details. The
[Installation page](https://lean-dojo.github.io/TorchLean/installation/) lists platform commands and
the profiles currently wired into the repository.

# References

- Paszke et al., ["PyTorch: An Imperative Style, High-Performance Deep Learning
  Library"](https://arxiv.org/abs/1912.01703), NeurIPS 2019.
- PyTorch, [`torch.compile` reference](https://docs.pytorch.org/docs/stable/generated/torch.compile.html).
- PyTorch, [C++ and LibTorch API](https://docs.pytorch.org/cppdocs/).
- NVIDIA, [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).
- Dao et al., ["FlashAttention: Fast and Memory-Efficient Exact Attention with
  IO-Awareness"](https://arxiv.org/abs/2205.14135), NeurIPS 2022.
- George C. Necula, ["Proof-Carrying Code"](https://doi.org/10.1145/263699.263712), POPL 1997.
- Lean, [validating proofs](https://lean-lang.org/doc/reference/latest/ValidatingProofs/).
