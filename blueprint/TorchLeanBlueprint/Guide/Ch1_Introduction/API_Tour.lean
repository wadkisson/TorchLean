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

The application entry point is `import NN.API`. The complete `NN` umbrella and the focused
subsystem imports become relevant when a chapter asks a more precise question about runtime
execution, graphs, floating point, proofs, or verification.

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

That separation makes a training step an explicit state transition:

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
[floating-point import](https://github.com/lean-dojo/TorchLean/blob/main/NN/Floats.lean) exists so a theorem, a verifier claim,
and a runtime run do not quietly use three different meanings of "float32."

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
