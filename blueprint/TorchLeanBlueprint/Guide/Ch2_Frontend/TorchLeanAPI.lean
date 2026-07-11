import VersoManual

open Verso.Genre Manual

#doc (Manual) "TorchLean API" =>
%%%
tag := "torchlean-api"
%%%

The public API starts with two lines:

```
import NN
open TorchLean
```

Those two lines are enough for ordinary examples. They give access to the names used throughout the
tutorials: `nn` for models, `Data` for datasets, `Trainer` for training and reports, `optim` for
optimizers, and `autograd` for explicit differentiation tools.

The public namespaces divide responsibilities as follows:

- `nn` builds models;
- `Data` builds datasets and loaders;
- `Trainer` runs training, prediction, callbacks, and reports;
- `optim` configures updates;
- `autograd` exposes explicit gradient tools when the trainer is not enough.

The first thing to keep in mind is the layering:

- `NN` is the canonical import; `TorchLean` remains the public namespace.
- `NN.API.Public` backs the `TorchLean.*` namespaces exported by `NN`.
- `NN.API.Runtime` exposes the executable runtime layer.
- `NN.API.TorchLean` is the namespace that the runtime layer re-exports for ordinary code.
- `NN.Entrypoint.*` modules provide focused imports for specialized proof or runtime paths.

Rule of thumb: use `import NN` and `open TorchLean` first. Drop to the runtime layer only
when the runtime itself is the subject.

# What Counts As An API Claim

TorchLean distinguishes four kinds of claims.

First, a *Lean snippet* is code meant to elaborate as Lean code when pasted into a file with the
right imports. These examples usually start with `import NN` and `open TorchLean`.

Second, a *runtime check* is an executable run: a training script, a data loader, a parity check, or
a CUDA smoke test. It is evidence about the implementation on the inputs that were run.

Third, a *certificate check* is a Lean program that validates a finite artifact such as a verifier
certificate. It is stronger than a printout, but its scope is exactly the checker and artifact
format that were used.

Fourth, a *theorem* is a Lean declaration proved in the proof layer. The theorem statement says
which semantics, scalar domain, graph fragment, and hypotheses are covered. A theorem about a Lean
graph evaluator is not automatically a theorem about a native CUDA kernel unless the bridge theorem
or trust boundary says so.

The public API lets these claims line up without conflating them. A model can train, lower to a
graph, produce an artifact, and appear in a theorem statement.

# The First Line Of A Tutorial

The shortest setup is:

```
import NN

open TorchLean

#check nn.Sequential
#check Trainer.new
#check optim.adam
#check Trainer.Config
#check Trainer.TrainOptions
```

That setup is enough for the ordinary tutorial path. Users should not have to choose between a dozen
internal imports before they have built their first model.

# The Public API

`TorchLean` is the canonical import for ordinary user code. Its top-level names are deliberately
close to the ones PyTorch readers already know:

- `nn` for layer builders and model constructors,
- `Data` for datasets, loaders, and small preprocessing APIs,
- `Trainer` for train/predict loops and training utilities,
- `optim` for optimizer configuration,
- `autograd` for gradient, VJP, and Jacobian APIs,
- `rand` for deterministic RNG APIs.

Tutorials, notebooks, and examples should prefer this layer. It does not hide the runtime; it gives
the runtime a stable and legible public interface.

The namespace map is:

- `nn`: layers, blocks, and model builders.
- `Data`: datasets, sources, loaders, and transforms.
- `Trainer`: training, prediction, callbacks, reports, and manual steps.
- `optim`: SGD, Adam, AdamW, and scheduler configuration.
- `autograd`: gradients, VJPs, Jacobians, and explicit differentiation.
- `NN.Entrypoint.IR`: graph examples and compiled artifacts.
- `NN.Entrypoint.Verification`: verifier and certificate examples.
- `NN.Entrypoint.Floats`: Float32 and numeric-semantics examples.

The user should not need to rewrite the model to move from a small training run to graph inspection
or to a verifier fixture. The runtime layer may require extra hypotheses, but it should be about the
same model.

The tutorial import remains short:

```
import NN
open TorchLean

def mkModel : nn.M (nn.Sequential (Shape.vec 2) (Shape.vec 1)) :=
  nn.Sequential![
    nn.Linear 2 8,
    nn.ReLU,
    nn.Linear 8 1
  ]

def trainer :=
  Trainer.new mkModel
    { task := .regression
      optimizer := optim.adam { lr := 0.03 }
      backend := .eager
      dtype := .float }
```

That record is a good example of the boundary. The model says what the architecture is. The trainer
record says how it should be run: task, optimizer, backend, dtype, seed, and device options. The
model definition does not fork into separate eager, compiled, or CUDA versions.

Two parts of the public API deserve special emphasis:

- `Data`
  includes typed CSV/NPY readers, deterministic shuffling, minibatch loaders, and
  constructors for supervised and labeled datasets;
- `Trainer`
  spans both full training and manual step APIs that feel familiar from PyTorch.

# Data And Transforms

TorchLean's data layer is closest in spirit to `torch.utils.data`, but the boundary is typed:

- file sources: `Data.TensorSource`, `Data.SupervisedSource`, `Data.LabeledSource`,
  `Data.TabularSupervisedSource`
- loading: `src.load` (on each source)
- deterministic minibatching: `Data.batchLoader` and `Data.BatchLoader.epoch`
- map-style transforms: `Data.Transforms` (combinators you can apply to samples or datasets)

The boundary format is the key design decision: TorchLean expects numeric tensors in `.npy` and
small numeric CSV for tabular data. For anything else, we usually convert once with Python tools.

The intended data contract is:

$$`\mathrm{loader}(\mathrm{file})
\in
\mathrm{Except}\;\mathrm{String}\;
  \left(\mathrm{Tensor}(\alpha,s_x)\times\mathrm{Tensor}(\alpha,s_y)\right)`

Loading can fail, and shape checks happen before a dataset becomes training data. The upfront check
is less convenient than accepting every file and hoping for the best, but later training and
verification code receives a typed object instead of an unchecked blob.

The small in-memory path has the same shape:

```
def xs : Tensor.T Float (shape![4, 2]) :=
  tensor! [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]

def ys : Tensor.T Float (shape![4, 1]) :=
  tensor! [[0.0], [1.0], [1.0], [0.0]]

def xorData : Trainer.Dataset (Shape.vec 2) (Shape.vec 1) :=
  Data.tensorDataset xs ys
```

The leading dimension of `xs` and `ys` is the number of examples. The dataset shape records the
per-sample contract, not the whole training table: one input has shape `Shape.vec 2`, and one target
has shape `Shape.vec 1`.

See the data contract documentation:

- [NN/API/Data/README.md](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Data/README.md)
- [NN/Examples/Data/README.md](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/README.md)

The public tutorials that exercise this layer are:

- [CSV loader example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Csv.lean)
- [NPY loader example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Npy.lean)
- [CIFAR image loader example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Cifar10Images.lean)

TorchLean examples now cover both small tutorial fixtures and data backed training runs.

# The Runtime Layer

`Runtime`, `Module`, `Supervised`, and `Ops` expose the runtime machinery behind the public
API. The corresponding subsystem modules live under `NN.API.Runtime` and `NN.API.TorchLean`. This
layer includes:

- typed and untyped tensor operations,
- backend-neutral programs and compiled outputs,
- session and execution control,
- losses, norms, optimizers, and training APIs,
- the explicit `eager` versus `compiled` execution distinction.

The inventory matters because the runtime layer keeps executable semantics visible enough for graph
inspection, verification, and interop work.

The runtime layer may look like ordinary infrastructure, but it is part of the scientific record of
a run. It answers questions such as:

- which scalar implementation was selected;
- whether the run used eager tape mode or compiled graph mode;
- whether a native backend was called;
- which values and gradients were cached;
- which log or graph artifact was produced.

Those facts matter when a result later becomes a plot, a regression test, or a verifier input.

For example, a public trainer configuration is just data:

```
def eagerCfg : Trainer.Config (Shape.vec 2) (Shape.vec 1) :=
  { dtype := .float
    backend := .eager
    optimizer := optim.sgd { lr := 0.05 }
    task := .regression
    seed := 7 }

def compiledCfg : Trainer.Config (Shape.vec 2) (Shape.vec 1) :=
  { eagerCfg with backend := .compiled }
```

Changing `backend` changes the execution artifact. It does not change the theorem you may later want
to prove about the model, and it does not change the parameter names or shapes.

Typical names in that layer include:

- `add`, `matmul`, `reshape`, `transpose2d`, `broadcastTo`,
- `linear`, `conv2d`, `layer_norm`, `multi_head_attention`,
- `trainCycleSGD`, `trainCycleOptim`, `meanLoss`,
- `Backend`, `Options`, plus the executable program/compiled graph types used by
  manual runtime code.

Those names represent the lower half of the same public API:

$$`\mathrm{Program}_\alpha
\;\xrightarrow{\mathrm{run}}\;
\mathrm{value},\mathrm{tape/log/graph},\mathrm{state'}`

The returned artifact matters. It is what widgets display, what compiled execution replays, and
what proof chapters relate back to the spec.

# Training, Reporting, And Schedulers

The training API has one normal public path, with a lower manual layer for implementation work.

## Main trainer path

- `Trainer.new model { task := .regression }`
- `Trainer.new model { task := .classification }`
- `Trainer.Config`
- `Trainer.TrainOptions`
- `trainer.train data trainOptions`

Public tutorials should copy this path. Data construction lives under `Data.*`; training stays on
the trainer object. We do not add dataset-specific trainer entrypoints for every example.

## Manual training loops

The manual runner API still exists for callbacks, custom stepping, and proof oriented runtime
experiments. Public examples should keep the normal path as one trainer object plus one config
record; manual loops are for cases that need direct control over each step.

## Optimizers and scheduler math

- `optim.sgd`, `optim.adam`
- `Trainer.constantLR`
- `Trainer.stepLR`
- `Trainer.exponentialLR`

The scheduler layer is pure and focused: it gives higher layer training code a stable
place to compute learning-rate policies without entangling that logic with runtime state.

# Execution Modes

TorchLean uses the same public model code with two execution styles:

- `eager` for debugging, gradients, and step by step inspection,
- `compiled` for a graph artifact that can be replayed or verified more directly.

That distinction matters because it keeps the choice of semantics explicit. The code does not change
when the backend changes; only the execution mode changes.

Device note: the same APIs also compose with an optional CUDA backed float32 path when the
project is built with `-K cuda=true`. Public tutorials often show `--device cpu` and `--device cuda` on the
`torchlean` runner; the public types do not change. Only the runtime buffer implementation does.
See *Runtime and Autograd* for the trust boundary and *Example Walkthroughs* for commands.

```
lake env lean --run NN/Examples/Quickstart/SimpleMlpTrain.lean -- --steps 200 --dtype float --backend compiled
```

The command above is a good reminder of one API invariant: the public API stays constant while the
backend varies.
Swapping `--backend eager` and `--backend compiled` shows the behavioral difference without rewriting
the model.

# Lean And PyTorch Background

- [Lean learning resources](https://lean-lang.org/learn/)
- Lean reference manual, including `do` notation and monadic syntax:
  [Lean reference manual](https://lean-lang.org/doc/reference/latest/)
- [PyTorch documentation](https://docs.pytorch.org/)
- [PyTorch autograd tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)

# When To Use A Lower Layer

Use `import NN` by default. Drop to a runtime layer only when the runtime layer is exactly what the
example is explaining:

1. `TorchLean` for ordinary examples.
2. `NN.API.Runtime` or `NN.API.TorchLean` only when the runtime API is the topic.
3. `NN.Entrypoint.*` for a narrow import tied to a specific topic (specs, runtime, floats, or
   verification).

This ordering keeps the manual coherent: the public API appears first, then the runtime assembly.
It also marks where TorchLean differs from a direct PyTorch clone. We do
not hide scalar-polymorphic code, shape-indexed parameter packs, graph denotations, or certificate
boundaries when those objects are the story. The rule is only that beginner examples should not need
to see them accidentally.

Concrete import choices:

```
-- Beginner model/training example
import NN
open TorchLean
```

```
-- Focused verifier example
import NN.Entrypoint.Verification
```

```
-- Focused Float32 example
import NN.Entrypoint.Floats
```

# Model Families Beyond The Smallest Tutorials

The public tutorials understandably emphasize MLPs and small CNNs, but the API and runtime now cover
more model families:

- residual CNN support via `nn.blocks.resnetBasicBlock`,
- transformer style blocks via `nn.multiheadAttention`, `nn.layerNorm`, and
  `nn.blocks.transformerEncoderBlock`,
- GraphSpec runtime programs such as `resnet18Program`,
- operator learning work such as the `fno1d` runtime model.

Not all of these families are beginner examples, but they are part of the model building API used by
larger tutorials and experiments.

# GraphSpec And Tooling

Three adjacent namespaces are easy to overlook from the broad `NN` umbrella alone.

## `NN.Entrypoint.*`

These are one-import entrypoints for focused subsystems:

- `NN.Entrypoint.Spec`
- `NN.Entrypoint.Runtime`
- `NN.Entrypoint.Floats`
- `NN.Entrypoint.Verification`
- `NN.Entrypoint.GraphSpec`

Use them in focused guide files or examples where the imports should say what the file is about.

## `NN.GraphSpec`

GraphSpec is the typed architecture DSL described in its own guide. It is not the default public
API, but it is already connected to the runtime model examples and to public examples such as
`lake exe torchlean graphspec`.

## Navigation Tools

The pattern is to start from the broad public namespace and move inward only when the example needs
to name a specific layer. Ordinary model code should feel like ML code first. The narrower
entrypoints are there for guide chapters, examples, and proof files that want their imports to say
exactly which part of TorchLean they are using.

# How The Pieces Fit Together

The split between public and runtime namespaces gives the project three practical benefits:

- tutorials stay concise,
- implementation details stay inspectable rather than hidden behind opaque objects,
- verification chapters can refer to the same names as the examples.

The API rule is simple: start broad, then narrow only when the topic demands it. Use
`import NN` for ordinary model code. Use `NN.Entrypoint.*` when a file is specifically about
graphs, floats, verification, widgets, or runtime internals. That keeps examples readable while
preserving precise entry points for the runtime and proof layers.

Read *Training From Scratch* for the first full training file, *Example Walkthroughs* for curated
commands, and *PyTorch Round Trip* then *TorchLean vs PyTorch* for interop.

# References

- TorchLean paper: https://arxiv.org/abs/2602.22631
- Lean module-system reference:
  https://lean-lang.org/doc/reference/latest/Source-Files-and-Modules/
- PyTorch documentation for the corresponding API concepts:
  https://pytorch.org/docs/stable/index.html
