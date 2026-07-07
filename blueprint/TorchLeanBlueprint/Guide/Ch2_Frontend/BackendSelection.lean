import VersoManual

open Verso.Genre Manual

#doc (Manual) "Backend Selection and Trust" =>
%%%
tag := "backend-selection"
%%%

TorchLean has one model story and several execution stories.

This distinction matters. A user writes a model once, with fixed shapes, parameters, and layer
structure. The runtime then chooses how to execute supported tensor work: eager host execution,
compiled graph execution, CUDA device execution, or an external provider such as PyTorch/ATen
for selected kernels. Those choices should improve speed or interoperability. They should not change
which mathematical model the library is talking about, as long as the operation is supported and
the stated domain and trust-boundary assumptions apply.

# The Short Rule

Use this rule when reading backend code:

- the *spec* owns the mathematical meaning;
- the *graph* owns the inspectable operation structure;
- the *runtime* owns buffers, mode, allocation, and execution;
- the *backend* owns where selected numeric kernels run;
- the *certificate or theorem* says what claim has actually been checked.

TorchLean avoids a public API with separate names such as "compiled forward" or "CUDA forward" for
that reason. There should be one public model and one public prediction or training path. Backend
selection belongs in the runtime configuration.

# Eager, Compiled, CUDA, and ATen

The execution choices do different jobs:

- *Eager TorchLean* records a tape as operations run. It is the clearest path for debugging reverse
  mode, inspecting VJPs, and explaining a single training step.
- *Compiled TorchLean* builds a reusable graph artifact for a fixed model and loss shape.
  It is the repeated-execution path.
- *CUDA TorchLean* moves supported Float32 tensor work onto device buffers. Lean still records the
  graph/tape shape and names the CUDA agreement assumptions.
- *ATen or PyTorch interop* can provide fast runtime kernels or import/export paths. It should not
  own TorchLean's graph semantics or proof story by default.

The last point matters most during training. The intended design is not "ATen forward, ATen
backward, and TorchLean watches from the side." For training that is meant to connect back to
TorchLean proofs, ATen can be a fast forward provider only when TorchLean still records the
corresponding node and uses the TorchLean backward rule. If an operation cannot preserve that
relation, the training path should fall back to the TorchLean implementation until the bridge is made
explicit.

# What Changes When A Backend Changes

Backend selection may change:

- where buffers are allocated;
- whether a graph artifact is cached;
- whether a fused native kernel replaces a sequence of smaller runtime operations;
- which runtime assumptions are needed for Float32, CUDA, cuBLAS, cuFFT, libtorch, or FFI code;
- which tests or sanitizer runs are relevant evidence.

Backend selection should not silently change:

- tensor shapes;
- parameter layout;
- training/evaluation mode semantics;
- hard-mask meanings in the spec layer;
- graph node identities used by a verifier;
- the theorem statement attached to a certificate.

When the backend changes one of those semantic objects, it is no longer only a backend change. It is a
new contract and should be named as one.

# A Backend Decision Table

Use the following table as a practical reading guide:

- choose *eager CPU* when you want the clearest execution trace and easy autograd inspection;
- choose *compiled CPU* when the model/loss shape is fixed and repeated execution matters;
- choose *eager CUDA* when supported Float32 tensor work should live on device buffers;
- choose *PyTorch/ATen interop* when Python tooling, imported weights, or selected external kernels
  are the point of the example;
- choose *IR/export* when the next consumer is a checker, verifier, or generated code path.

Those choices can be combined only where the implementation says they can. If a CUDA path supports
only a fragment of operations, unsupported operations should fail clearly or fall back through a
named path. A quiet semantic change is worse than a runtime error.

# The Training Boundary

Training combines forward execution, a loss, a tape or compiled derivative artifact, an optimizer,
and state updates. Backend boundaries are sharper during training than during inference because the
forward value and the derivative rule must stay connected.

For inference, a backend result can often be treated as a value plus a stated agreement assumption. For
training, the same value must also be connected to the derivative rule used for the update. TorchLean's
preferred training contract is therefore:

1. the backend may compute the forward numeric value;
2. TorchLean still records the operation in its graph or tape;
3. TorchLean still owns the backward rule used by the optimizer;
4. the runtime cache records any forward data needed by that backward rule;
5. unsupported cases use the existing TorchLean path.

The result is a fast path that still leaves TorchLean in charge of the graph and backward semantics.

In code, the public training surface stays small:

```
def trainerFor (backend : TorchLean.Runtime.Backend) :=
  Trainer.new mkModel
    { task := .regression
      optimizer := optim.adam { lr := 0.01 }
      dtype := .float
      backend := backend }

let eagerTrainer := trainerFor .eager
let compiledTrainer := trainerFor .compiled
```

That is the shape TorchLean wants: one model, one public training path, a backend value in the
configuration.

# Reading Claims Carefully

The same run can support several different claims:

- "the example ran on CUDA" means the native path executed;
- "the CUDA parity test passed" means it matched a reference on tested inputs;
- "the graph compiler theorem applies" means a Lean theorem connects two Lean semantics;
- "the certificate checked" means a finite artifact passed its Lean checker;
- "the native kernel is verified" would mean a proof about the native kernel itself.

Domain failures are part of the claim too. For example, raw logarithm has the real-domain
precondition `x > 0`; use `safe_log` or `safeLog` when the model needs a total
epsilon-protected log-like operation.

Only the last claim proves the native kernel implementation. Most current backend work is one level
weaker: Lean owns the specification, and native code enters through named agreement assumptions plus
tests, sanitizers, and parity checks.

For example, these are different sentences:

- "The MLP example trained with `--backend compiled`."
- "The eager and compiled backends matched on this parity test."
- "A Lean theorem proves the compiled evaluator agrees with the IR evaluator for this operator
  fragment."
- "The CUDA DGEMM kernel implementation itself is verified."

The first three can all be true without the fourth being true. TorchLean's documentation should make
that visible, especially when native code is involved.

# A Clean Public API

The public API should make the common case direct:

```
let trainer := TorchLean.Trainer.new model cfg
let trained ← trainer.train data opts
trained.predict x
```

The configuration chooses dtype, backend, device, optimizer, logging, and runtime options. The model
API does not need a separate public function for every backend. This follows the same ergonomic
lesson as PyTorch's `model(x)` versus `torch.compile(model)`: compilation is a runtime
transformation around a model, not a second mathematical model.

# Where To Go Next

Read [Execution Modes and Runners](Runtime___-Autograd___-and-Interop/Execution-Modes-and-Runners/)
for the everyday API. Read
[GPU and CUDA Boundaries](Floating-Point-and-Native-Boundaries/GPU-and-CUDA-Boundaries/)
for the native CUDA contract. Read
[Verification and Certificates](Verification-and-Certificates/Verification/) for the difference
between runtime evidence, certificates, and Lean theorems.

# References

- PyTorch `torch.compile` reference: https://docs.pytorch.org/docs/stable/generated/torch.compile.html
- PyTorch FX reference: https://docs.pytorch.org/docs/stable/fx.html
- NVIDIA cuBLAS documentation: https://docs.nvidia.com/cuda/cublas/
