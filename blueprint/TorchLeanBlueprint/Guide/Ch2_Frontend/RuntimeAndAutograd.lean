import VersoManual

open Verso.Genre Manual

#doc (Manual) "Runtime Internals and Artifacts" =>
%%%
tag := "runtime-autograd"
%%%

The runtime layer is where a typed model becomes a run. It allocates values, records operations,
computes gradients, updates parameters, writes logs, and produces artifacts that can be inspected
later.

TorchLean has more than one runtime artifact. Eager execution produces a tape. Compiled execution
produces a reusable graph object. Verification uses an IR whose nodes name their operations. These
artifacts are related, but they are not the same data structure.

If a compact model has not run all the way through yet, *Training From Scratch* is the best first
stop; it makes the runtime layer much easier to ground.

# PyTorch Mental Model

PyTorch's default workflow is approximately:

1. create parameters and modules,
2. run a forward pass,
3. let autograd record the operations used by that forward pass,
4. call `loss.backward()` to compute gradients,
5. call `optimizer.step()` to update parameters,
6. zero gradients and repeat.

TorchLean keeps the rhythm but changes where the objects live:

- parameters and modules become explicit typed values and parameter bundles;
- the dynamic autograd tape becomes a Lean `Tape`;
- `.grad` accumulation becomes explicit gradient values returned by reverse mode;
- the normal user-facing loop is `trainer.train`; manual `optimizer.step()`-style loops live behind
  `Trainer.Manual.step`/`stepper` for runtime work;
- eager mode produces a tape, while compiled mode produces a reusable graph artifact.

That mapping lets a PyTorch reader recognize the workflow without treating the runtime state as
hidden global context.

# Two Execution Modes

TorchLean exposes one front end with two execution backends:

- *Eager*: tape recording and reverse-mode backprop in the style familiar from PyTorch.
- *Compiled*: a stable SSA/DAG artifact for repeated evaluation and proof alignment.

Many curated examples accept `--backend eager|compiled`.

```
lake env lean --run NN/Examples/Quickstart/SimpleMlpTrain.lean -- --steps 50 --dtype float --backend eager
lake env lean --run NN/Examples/Quickstart/SimpleMlpTrain.lean -- --steps 50 --dtype float --backend compiled
```

With matching seeds and supported operators, the forward computation should agree. What changes is
the runtime artifact:

- eager is easier to step through;
- compiled is easier to replay and connect to graph proof artifacts.

# Runtime Contexts And Named Values

Spec tensors are indexed by shape, but realistic training loops need registries:

- parameter maps such as `"w1" ↦ weights`;
- gradient maps such as `"w1" ↦ dL/dw1`;
- named values for debugging and widgets.

TorchLean therefore uses an existential container, `Runtime.AnyTensor α`, which pairs a `Shape` with
the corresponding tensor. That preserves the strongly typed spec layer while still supporting
runtime tooling.

See the [runtime context API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Context.lean)
and the runtime-context widgets for the concrete declarations.

# A Small Runtime Walkthrough

For a single training step, the objects move like this:

1. Build a model and parameter bundle through `NN.API`.
2. Run a forward pass with a chosen backend.
3. Produce a scalar loss.
4. Run reverse mode to obtain explicit gradients.
5. Pass parameters and gradients to an optimizer update.
6. Log metrics and, when needed, inspect the tape or compiled graph.

Those steps change the runtime state and the produced artifacts. They do not require rewriting the
architecture or the parameter-shape contract.

# Autograd Tape

The eager engine records operations into a `Tape`:

- nodes store forward values;
- nodes remember parent ids;
- nodes carry local VJP closures;
- reverse mode traverses node ids backward and accumulates gradients.

In PyTorch terms, this is the part of the system behind `loss.backward()`. TorchLean makes the tape
and gradient flow explicit. There is no hidden `.grad` mutation on tensor objects; the reverse pass
produces gradients directly.

The widgets make this visible:

- `#tape_view t` renders nodes, parents, and values;
- `#tape_grads_view t, outId` runs scalar backprop and shows which nodes receive gradients;
- `#tape_trace_view t, outId` shows the reverse traversal step by step;
- `#runtime_ctx_view ctx` shows the value and gradient registries.

The [widgets example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/DeepDives/Widgets.lean)
contains compact examples for these views.

A useful reading habit is to follow one scalar loss backward:

```
prediction -> loss
loss cotangent = 1
reverse traversal sends cotangents to parents
parameter cotangents become gradients
optimizer consumes parameters + gradients
```

TorchLean's eager tape exposes those intermediate objects. A theorem about the tape proves a
statement about the reverse traversal under its hypotheses. A training run merely executes the path
for the selected model, data, scalar backend, and runtime options.

# Compiled Graphs

The compiled runtime uses a typed SSA/DAG representation. Each node bundles the information needed
for forward evaluation and derivative propagation:

- `forward` computes a value;
- `jvp` computes a forward-mode pushforward;
- `vjp` computes a reverse-mode pullback.

The compiled form is separate from the eager tape for a practical reason. Eager mode is best for
debugging and interactive iteration. Compiled mode is best for a stable, replayable artifact that
proof code can reason about. Both paths are produced from the same public model definition.

API starting points:

- [compiled graph builder](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Compiled/GraphM.lean)
- [compiled runtime core](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Torch/Core.lean)
- [runtime overview](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Overview.lean)

The compiled path still has a derivative story. A compiled node records enough local structure to
evaluate forward values and propagate derivative information. That is why compiled execution is more
than a cache of numbers. It is a reusable executable representation of the same typed computation.

# IR Execution Bridge

For verification, TorchLean standardizes on `NN.IR.Graph`, the DAG described in *Graphs and IR*.
Runtime closures are good for execution, but a verifier needs explicit operation
tags, shapes, and parent ids.

The [IR execution compiler](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Compiled/IRExec.lean)
connects that IR to the compiled runtime backend. In words, `execGraphOfIR` produces
a compiled graph whose forward evaluation agrees with the IR evaluator on the same payload and
input, for the supported operator fragment.

That last phrase matters: *for the supported operator fragment*. If an imported or generated graph
contains an operation outside the fragment, the bridge must extend its semantics or reject the graph.
Otherwise a checker would be reasoning about a different program than the runtime executed.

# Proof Link

The proof layer follows a local-to-global pattern. Each primitive operation has a forward rule and a
VJP rule. If the local VJP rule is the adjoint derivative of the local forward rule, then reverse
traversal of a well-formed graph computes the adjoint derivative of the whole graph.

Later proof chapters state the exact Lean theorems. Here the runtime fact to remember is simpler:
the tape and compiled artifacts expose the structure those theorems need.

For a proof tour, use:

- [autograd proof overview](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Overview.lean)
- [tape algebra soundness API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Tape/Algebra/Soundness.lean)
- [runtime autograd link API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Runtime/Link.lean)

# CUDA Is A Backend Choice

CUDA details have their own guide page. The runtime rule is short: GPU mode accelerates supported
Float32 buffer operations, while Lean still owns the model structure, typed interfaces, logs, graph
artifacts, and proof/checker statements.

Use eager mode for stepping through compact examples. Use compiled mode for repeated evaluation over a
stable graph artifact. Use CUDA when the supported Float32 runtime should place numeric work
on device buffers.

# Where To Look

For runnable examples close to this runtime layer:

- [Float32 modes example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/DeepDives/Floats/Float32Modes.lean)
- [AutogradBasics](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/AutogradBasics.lean)
- [SimpleMlpTrain](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleMlpTrain.lean)
- [MinibatchMlpTrain](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/MinibatchMlpTrain.lean)

To read the runtime layer in dependency order, begin with eager tensors and tapes, then compiled
graph construction, then the IR execution bridge, and finally the curated model examples.

# References

- TorchLean paper: https://arxiv.org/abs/2602.22631
- Baydin et al., "Automatic differentiation in machine learning: a survey": https://arxiv.org/abs/1502.05767
- Griewank and Walther, *Evaluating Derivatives: Principles and Techniques of Algorithmic
  Differentiation*.
- PyTorch autograd docs: https://pytorch.org/docs/stable/autograd.html
- PyTorch FX docs: https://pytorch.org/docs/stable/fx.html
