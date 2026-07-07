import VersoManual

open Verso.Genre Manual

#doc (Manual) "A Tour of the API" =>
%%%
tag := "api_tour"
%%%

Read TorchLean's public API by following one model as it changes roles. At first, the code looks
familiar: tensors, layers, losses, optimizers, and training loops. The difference is what TorchLean
keeps around when the same model is trained, lowered to a graph, inspected, exported, or checked.

A tensor carries its shape. A model carries an input and output contract. A parameter payload is
ordinary data. A lowered graph has named operations and node ids. A certificate is checked against
the graph rather than treated as a comment beside it.

A typical workflow looks like this:

1. instantiate tensors and a model,
2. build parameters and run a training loop,
3. inspect the run through logs or graphs,
4. import or export a named artifact when Python belongs in the workflow,
5. state a claim about the resulting model and check the artifact that supports it.

The public entry point is `import NN`. The lower layer pages become relevant
when a chapter asks a more precise question about tensors, runtime execution, import/export,
floating point, or verification.

The API is intentionally layered. Most users should start with the public names and only descend
when the question demands it. A training tutorial can stay with `Tensor`, `nn`, `data`, `optim`, and
`Trainer`. A graph-inspection chapter opens the IR. A certificate chapter opens the verifier and
bound-propagation code. This keeps the common path readable while still leaving precise objects for
proof-oriented work.

Here is the map:

- When the user writes a tensor literal, TorchLean keeps the scalar type and shape.
- When the user builds a model, TorchLean keeps the architecture and parameter shapes.
- When the user runs training, TorchLean keeps parameter state, optimizer state, random state, and logs.
- When the user lowers a graph, TorchLean keeps operation names, node ids, parent links, and payloads.
- When the user checks a certificate, TorchLean keeps the checker predicate and scalar semantics.

# Public Names And Lower Names

The public API prefers names that describe the ML task. The lower layers prefer names that describe
the semantic object being manipulated. Both are useful.

```
import NN

open TorchLean

-- Public authoring layer: compact model construction.
def publicMlp : nn.M (nn.Sequential (Shape.vec 8) (Shape.vec 3)) :=
  nn.Sequential![
    nn.Linear 8 16,
    nn.ReLU,
    nn.Linear 16 3
  ]

-- Interactive inspection: ask Lean what object was built.
#check publicMlp
```

When the guide later mentions `NN.IR.Graph`, `NN.Spec`, `NN.MLTheory.CROWN`, or
`NN.Verification`, it is usually because the page has shifted from "how do I build and run this
model?" to "what does this artifact mean?" The module names are a map of that shift.

# Tensors And Models

The first visible difference from ordinary Python model code is that tensor shapes appear in the
Lean type. A value of type `Tensor Float (shape![32, 1])` is not interchangeable with a value of
type `Tensor Float (shape![32])`; a reshape, squeeze, or different loss must be named explicitly.

```
import NN

open TorchLean

def logits : Tensor.T Float (shape![32, 1]) :=
  tensorND! [32, 1] (List.replicate 32 0.0)

def labels : Tensor.T Float (shape![32]) :=
  tensorND! [32] (List.replicate 32 0.0)

-- A loss expecting matching shapes cannot silently reinterpret `labels`.
```

Broadcasting is sometimes intended, but a training loss with predictions shaped `[batch, 1]` and
targets shaped `[batch]` is often a modeling bug. TorchLean does not guess whether the right
convention is one-hot encoding, a singleton dimension squeeze, a different loss, or a different
model head.

The same shape discipline appears at the model level. A compact classifier states its input and output
shapes before it ever runs:

```
def classifier : nn.M (nn.Sequential (Shape.vec 16) (Shape.vec 4)) :=
  nn.Sequential![
    nn.Linear 16 32,
    nn.GELU,
    nn.Linear 32 4
  ]
```

Tensor constructors, literals, and model builders live under the root
[`NN`](https://github.com/lean-dojo/TorchLean/blob/main/NN.lean) umbrella.

# Parameters Are Part Of The Interface

In a proof-aware workflow, parameters are not just implementation details. A trained payload is the
difference between a family of networks and one concrete network. TorchLean therefore treats the
payload as data that can be initialized, saved, imported, lowered with a graph, and checked against
shape expectations.

This distinction is especially useful when a paper or report says "the model was verified." Usually
the intended object is not merely an architecture such as "two-layer MLP." It is:

$$`\text{architecture} + \text{parameter payload} + \text{input convention}
  + \text{scalar semantics}`

Leaving out the payload turns a concrete verification claim into a family-level statement that is
almost certainly false. Leaving out the input convention can make a true post-normalization claim
look like a raw-data claim. Leaving out scalar semantics can confuse a real-valued proof with a
float32 execution result.

# Building And Training

In TorchLean, model structure and parameters are separate values. Building a model chooses an
initial parameter payload, but it does not make those parameters hidden fields of a mutable object.

```
def task (seed : Nat) :=
  Trainer.new classifier { task := .classification, seed := seed }
```

Later chapters rely on that separation. A training step can be read as an explicit
state transition:

$$`\mathrm{step} :
  (\theta,\mathrm{optState},\mathrm{rng},\mathrm{mode},x,y)
  \longmapsto
  (\theta',\mathrm{optState}',\mathrm{rng}',\mathrm{log})`

The state may be large, but it is no longer implicit. Parameters, optimizer moments, random seeds,
training/evaluation mode, and metric reports are all ordinary data that can be passed, saved,
loaded, displayed, or mentioned in a theorem statement.

The training API follows the familiar pattern:

- `Trainer.new model { task := ... }` chooses the task,
- the trainer config carries optimizer, dtype, backend, and device,
- `Trainer.TrainOptions` carries training choices such as steps and logging,
- `trainer.train data trainOptions` is the normal training entrypoint,
- `optim.sgd`, `optim.adam`, and `optim.adamw` construct optimizer configurations.

Manual callbacks and explicit step loops live under `Trainer.Manual`; ordinary tutorials should not
need them. Those names live behind the same public API, with implementation details in the
[training runtime](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Train.lean) and the
[runtime optimizer definitions](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/TorchLean/Optim.lean).

# A Step Is Not A Theorem

Running a training step computes new parameters. It does not prove that the model is accurate,
robust, or even useful. That sounds obvious, but documentation often blurs the categories by
describing a successful run as "verified" because it passed a suite of checks.

TorchLean keeps the categories separate:

- the training API executes an optimization procedure;
- the logging API records what happened;
- graph checks validate structural properties such as well-formedness and shapes;
- theorem files state and prove semantic properties for supported fragments.

This separation is what lets a later theorem cite the trained payload without treating the training
run itself as a proof of the final property.

# Inspecting Runs

A runnable model should not become opaque after it trains. TorchLean keeps several inspection
tools close to the training path:

- structured training logs, persisted as ordinary JSON, are defined by the
  [training log API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Training/Log.lean);
- logger hooks for training loops are defined by the
  [training logger API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Train/Logging.lean);
- widgets can display tensors, logs, execution traces, and IR graphs through the
  [widget entrypoint](https://github.com/lean-dojo/TorchLean/blob/main/NN/Entrypoint/Widgets.lean).

Logs and graphs are audit artifacts. A loss curve records which run was performed, while a lowered
graph records which operations a verifier or compiler is about to interpret.

# Graphs, Export, And Import

The public model API is comfortable for writing models. Verification and interop need a more
explicit object: a graph whose nodes name their operations, together with a payload store.
[NN.IR.Graph](https://github.com/lean-dojo/TorchLean/blob/main/NN/IR/Graph.lean) is that object. Its denotation is the semantic reference used by graph
checkers, widgets, compiler bridges, and verifier passes.

The PyTorch boundary follows the same discipline. The supported path is a compact contract:

$$`\text{known architecture family}
\;+\;
\text{named tensor payload}
\;+\;
\text{shape checks}
\quad\leadsto\quad
\text{TorchLean parameters}`

That contract is implemented by the [PyTorch export](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/PyTorch/Export.lean) and
[PyTorch import](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/PyTorch/Import.lean) APIs, with runnable examples in the
[PyTorch interop tutorial](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Interop/PyTorch.lean). Python can remain the right tool
for data preparation or large-scale training, while Lean receives a payload with names, shapes, and
model family assumptions it can check.

The wider ecosystem has similar graph-export pressures. PyTorch's
[`torch.export`](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/export.html) aims to
capture a full graph representation for deployment and ahead-of-time workflows. ONNX describes an
open format for representing machine-learning models. TorchLean's graph layer is not trying to
replace those systems as an interchange standard. It is narrower: the graph is a Lean object with
operations, shape information, payload links, and denotational hooks that later proof and checker
code can cite.

# Verification Claims

After lowering, a verifier no longer has to guess which computation is being analyzed. It receives
a graph, a payload, an input region, and a scalar semantics. A robustness claim, for example, has
the form:

$$`\forall x\in B,\quad
\operatorname{margin}(\operatorname{denote}(g,\theta,x)) > 0`

A robustness certificate in TorchLean is a concrete claim about a graph, a parameter payload, an
input region, a scalar semantics, and a checker result. The [verification entrypoint](https://github.com/lean-dojo/TorchLean/blob/main/NN/Entrypoint/Verification.lean)
collects the public verification API, while the [CROWN graph API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/CROWN/Graph.lean)
shows the graph objects used by the bound propagation chapters.

The central API pattern is:

$$`\text{runnable model}
\;\longrightarrow\;
\text{explicit graph and payload}
\;\longrightarrow\;
\text{checked artifact}
\;\longrightarrow\;
\text{semantic claim}`

# Failure Modes The API Makes Visible

The API choices above come from concrete failure modes.

Shape mismatch is the simplest one. A target tensor with shape `[batch]` should not silently become a
`[batch, 1]` tensor because a loss function can broadcast. The type mismatch forces the
training script to state the intended convention.

Hidden runtime state is another. BatchNorm buffers, dropout mode, random seeds, optimizer moments,
tokenizer tables, and cache layouts affect the computation. TorchLean's functional style keeps
these objects in the data path instead of leaving them implicit inside a module instance.

Floating point semantics also need names. A real valued specification, the `FP32` proof model,
the executable `IEEE32Exec` model, and native CUDA kernels are related but not identical. The
[floating point entrypoint](https://github.com/lean-dojo/TorchLean/blob/main/NN/Entrypoint/Floats.lean) exists so a theorem, a verifier claim,
and a runtime run do not quietly use three different meanings of "float32."

Fast kernels are boundaries too. For attention, the mathematical contract is ordinary scaled
dot product attention, while fused FlashAttention implementations are optimized kernels that must be
related back to that contract. The relevant proof statements live near the attention and GPU
chapters: a fast path should preserve a slow, readable meaning, or else it is a different model.

# Reading The Rest Of The Guide

This tour should make the later chapters easier to navigate. *Building Models* explains the public
API in detail. *Runtime and Interop* shows training, logs, execution modes, and PyTorch
round trips. *Semantics and Graphs* fixes the graph denotation. *Verification* explains how checked
artifacts become claims about the model.

# References

- Szegedy et al., ["Intriguing properties of neural networks"](https://arxiv.org/abs/1312.6199),
  ICLR 2014.
- Zhang et al., ["Efficient Neural Network Robustness Certification with General Activation
  Functions"](https://arxiv.org/abs/1811.00866), NeurIPS 2018.
- Xu et al., ["Automatic Perturbation Analysis for Scalable Certified Robustness and
  Beyond"](https://arxiv.org/abs/2002.12920), NeurIPS 2020.
- Goldberg, ["What Every Computer Scientist Should Know About Floating-Point Arithmetic"](https://doi.org/10.1145/103162.103163),
  ACM Computing Surveys 1991.
- PyTorch `torch.export` documentation: https://docs.pytorch.org/docs/main/user_guide/torch_compiler/export.html
- ONNX project documentation: https://onnx.ai/
- IEEE 754-2019, *Standard for Floating-Point Arithmetic*.
