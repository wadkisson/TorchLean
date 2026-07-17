import VersoManual

open Verso.Genre Manual

#doc (Manual) "A Tour of the API" =>
%%%
tag := "api_tour"
%%%

Follow one model through the public API: tensors and layers, training, inspection, import/export,
then a claim about a checked artifact. Start with `import NN.API` (`Tensor`, `nn`, `data`, `optim`,
`Trainer`); open IR or verification modules only when the question demands it.

# Public Names And Lower Names

The public API prefers names that describe the ML task. The lower layers prefer names that describe
the semantic object being manipulated. Both are useful.

```
import NN.API

open TorchLean

-- Public authoring layer: compact model construction.
def publicMlp : nn.M (nn.Sequential (.dim 8 .scalar) (.dim 3 .scalar)) :=
  nn.Sequential![
    nn.Linear 8 16,
    nn.ReLU,
    nn.Linear 16 3
  ]

-- Interactive inspection: ask Lean what object was built.
#check publicMlp
```

`NN.IR.Graph`, `NN.Spec`, `NN.MLTheory.CROWN`, and `NN.Verification` shift the question from "how do
I build and run this model?" to "what does this artifact mean?"

# Tensors And Models

The first visible difference from ordinary Python model code is that tensor shapes appear in the
Lean type. A value of type `Tensor Float (shape![32, 1])` is not interchangeable with a value of
type `Tensor Float (shape![32])`; a reshape, squeeze, or different loss must be named explicitly.

```
import NN.API

open TorchLean

def logits : Tensor.T Float (shape![32, 1]) :=
  tensorOfList! [32, 1] (List.replicate 32 0.0)

def labels : Tensor.T Float (shape![32]) :=
  tensorOfList! [32] (List.replicate 32 0.0)

-- A loss expecting matching shapes cannot silently reinterpret `labels`.
```

Broadcasting is sometimes intended, but a training loss with predictions shaped `[batch, 1]` and
targets shaped `[batch]` is often a modeling bug. TorchLean does not guess whether the right
convention is one-hot encoding, a singleton dimension squeeze, a different loss, or a different
model head.

The same shape discipline appears at the model level. A compact classifier states its input and output
shapes before it ever runs:

```
def classifier : nn.M (nn.Sequential (.dim 16 .scalar) (.dim 4 .scalar)) :=
  nn.Sequential![
    nn.Linear 16 32,
    nn.GELU,
    nn.Linear 32 4
  ]
```

Tensor constructors, literals, and model builders are exported by
[`NN.API`](https://github.com/lean-dojo/TorchLean/blob/main/NN/API.lean).

# Parameters Are Part Of The Interface

In a proof-aware workflow, parameters are not just implementation details. A trained payload is the
difference between a family of networks and one concrete network. TorchLean therefore treats the
payload as data that can be initialized, saved, imported, lowered with a graph, and checked against
shape expectations.

A verification claim is about architecture + parameter payload + input convention + scalar
semantics—not an architecture name alone. See *Why Verification Matters* for the mismatch examples.

# Building And Training

In TorchLean, model structure and parameters are separate values. Building a model chooses an
initial parameter payload, but it does not make those parameters hidden fields of a mutable object.

```
def task (seed : Nat) :=
  Trainer.new classifier { task := .classification, seed := seed }
```

That separation makes a training step an explicit state transition:

$$`\mathrm{step} :
  (\theta,\mathrm{optState},\mathrm{rng},\mathrm{mode},x,y)
  \longmapsto
  (\theta',\mathrm{optState}',\mathrm{rng}',\mathrm{log})`

The variables are:

- `θ` — the current parameter payload;
- `optState` — optimizer state such as moments and step counters;
- `rng` — random-generator state used by stochastic layers or data order;
- `mode` — train or evaluation mode;
- `x` — the batch input;
- `y` — the batch target;
- `θ'`, `optState'`, `rng'` — the updated state after the step;
- `log` — the metric report produced by the step.

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
  [widget import](https://github.com/lean-dojo/TorchLean/blob/main/NN/Widgets.lean).

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

After lowering, a verifier receives a graph, payload, input region, and scalar semantics. Robustness
formulas are introduced in *Why Verification Matters*; the public APIs are
[NN.Verification](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification.lean) and the
[CROWN graph API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/CROWN/Graph.lean).

# Failure Modes The API Makes Visible

The API choices above come from concrete failure modes.

Shape mismatch is the simplest one. A target tensor with shape `[batch]` should not silently become a
`[batch, 1]` tensor because a loss function can broadcast. The type mismatch forces the
training script to state the intended convention.

Hidden runtime state is another. BatchNorm buffers, dropout mode, random seeds, optimizer moments,
tokenizer tables, and cache layouts affect the computation. TorchLean's functional style keeps
these objects in the data path instead of leaving them implicit inside a module instance.

Floating-point layer names live in
[NN.Floats](https://github.com/lean-dojo/TorchLean/blob/main/NN/Floats.lean) and *Floating Point and
Native Boundaries*.

Fast kernels are boundaries too. For attention, the mathematical contract is ordinary scaled
dot product attention, while fused FlashAttention implementations are optimized kernels that must be
related back to that contract. The relevant proof statements live near the attention and GPU
chapters: a fast path should preserve a slow, readable meaning, or else it is a different model.

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
