import VersoManual

open Verso.Genre Manual

#doc (Manual) "TorchLean API" =>
%%%
tag := "torchlean-api"
%%%

An application file normally starts with two lines:

```
import NN.API
open TorchLean
```

This imports the model, data, training, and differentiation API without importing every proof and
verification module. The larger `import NN` umbrella includes the same public API together with the
rest of TorchLean; guide files use it when an example crosses those subsystem boundaries.

The public namespaces divide responsibilities as follows:

- `nn` builds models;
- `classical` provides KNN, tree, regression, mixture, and related model definitions;
- `Data` builds datasets and loaders;
- `Trainer` runs training, prediction, callbacks, and reports;
- `optim` configures updates;
- `autograd` exposes explicit gradient tools when the trainer is not enough.

`NN.API` is the centralized public import. The definitions behind it are divided into focused
modules so that TorchLean itself can be developed without import cycles, but application code does
not need to know that internal layout. Direct imports such as `NN.Spec`, `NN.Runtime`, and
`NN.Proofs` are available for files devoted to one subsystem.

# What Counts As An API Claim

TorchLean distinguishes four kinds of claims.

First, a *Lean snippet* is code meant to elaborate as Lean code when pasted into a file with the
right imports. Application examples usually start with `import NN.API` and `open TorchLean`.

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
import NN.API

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

`NN.API` is the application import, and `TorchLean` is the public namespace. Its top-level names
are deliberately close to the ones PyTorch readers already know:

- `nn` for layer builders and model constructors,
- `classical` for classical and statistical models,
- `Data` for datasets, loaders, and small preprocessing APIs,
- `Trainer` for train/predict loops and training utilities,
- `optim` for optimizer configuration,
- `autograd` for gradient, VJP, and Jacobian APIs,
- `rand` for deterministic RNG APIs.

Tutorials, notebooks, and examples should prefer this layer. It does not hide the runtime; it gives
the runtime a stable and legible public interface.

The namespace map is:

- `nn`: layers, blocks, and model builders.
- `classical`: KNN, forests, Naive Bayes, SVMs, PCA, GMMs, regressions, boosted trees, HMMs, and
  Hopfield networks.
- `Data`: datasets, sources, loaders, and transforms.
- `Trainer`: training, prediction, callbacks, reports, and manual steps.
- `optim`: SGD, Adam, AdamW, and scheduler configuration.
- `autograd`: gradients, VJPs, Jacobians, and explicit differentiation.
- `NN.IR`: graph examples and compiled artifacts.
- `NN.Verification`: verifier and certificate examples.
- `NN.Floats`: Float32 and numeric-semantics examples.

The user should not need to rewrite the model to move from a small training run to graph inspection
or to a verifier fixture. The runtime layer may require extra hypotheses, but it should be about the
same model.

The tutorial import remains short:

```
import NN.API
open TorchLean

def mkModel : nn.M (nn.Sequential (.dim 2 .scalar) (.dim 1 .scalar)) :=
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

def xorData : Trainer.Dataset (.dim 2 .scalar) (.dim 1 .scalar) :=
  Data.tensorDataset xs ys
```

The leading dimension of `xs` and `ys` is the number of examples. The dataset shape records the
per-sample contract, not the whole training table: one input has shape `.dim 2 .scalar`, and one target
has shape `.dim 1 .scalar`.

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
def eagerCfg : Trainer.Config (.dim 2 .scalar) (.dim 1 .scalar) :=
  { dtype := .float
    backend := .eager
    optimizer := optim.sgd { lr := 0.05 }
    task := .regression
    seed := 7 }

def compiledCfg : Trainer.Config (.dim 2 .scalar) (.dim 1 .scalar) :=
  { eagerCfg with backend := .compiled }
```

Changing `backend` changes the execution artifact. It does not change the theorem you may later want
to prove about the model, and it does not change the parameter names or shapes.

Typical names in that layer include:

- `add`, `matmul`, `reshape`, `transpose2d`, `broadcastTo`,
- `linear`, rank-generic `conv` and pooling, `layerNorm`, `multiheadAttention`,
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

Use `NN.API` for model construction, datasets, training, and the public classical models. Use `NN`
when one file also needs the specification, proof, verification, graph, or backend layers. A file
that belongs entirely to one of those layers can import its subsystem directly.

Concrete import choices:

```
-- Beginner model/training example
import NN.API
open TorchLean
```

```
-- Model code together with proofs or verification
import NN
open TorchLean
```

```
-- Focused verifier example
import NN.Verification
```

```
-- Focused Float32 example
import NN.Floats
```

# Model Families Beyond The Smallest Tutorials

The public tutorials understandably emphasize MLPs and small CNNs, but the API and runtime now cover
more model families:

- residual CNN support via `nn.blocks.resnetBasicBlock`,
- transformer style blocks via `nn.multiheadAttention`, `nn.layerNorm`, and
  `nn.blocks.transformerEncoderBlock`,
- rank-polymorphic residual networks via `nn.models.resnet`,
- operator learning work such as the `fno1d` runtime model.

Not all of these families are beginner examples, but they are part of the model building API used by
larger tutorials and experiments.

# Classical And Statistical Models

TorchLean also includes models that are not assembled from differentiable neural-network layers.
They remain useful library features: a PCA transform or a fitted random forest may be the model
under study, a preprocessing stage, or a reference computation beside a neural model. They are
available from the same import:

```
import NN.API
open TorchLean

#check classical.knn.Model
#check classical.knn.classify
#check classical.randomForest.regression.Model
#check classical.randomForest.classification.fitGini
#check classical.naiveBayes.fit
#check classical.svm.fit
#check classical.gmm.trainEM
#check classical.pca.fit
#check classical.linearRegression.trainStep
#check classical.logisticRegression.fit
#check classical.gradientBoostedTrees.trainStepAndFit
#check classical.hmm.baumWelchEpoch
#check classical.hopfield.energy
```

The public names are aliases, not wrappers with separate semantics. For example,
`classical.pca.Model` is `Spec.PCASpec`, and `classical.pca.forward` is
`Spec.pcaForwardSpec`. Proof files can name the `Spec` declaration directly while application code
uses the shorter family-oriented path.

# Focused Subsystem Imports

The main focused imports are:

- `NN.Spec`
- `NN.Runtime`
- `NN.Floats`
- `NN.Verification`
- `NN.GraphSpec`

These imports are useful in implementation, proof, and verification files that do not need the
complete umbrella.

## `NN.GraphSpec`

GraphSpec is TorchLean's typed architecture DSL. It connects architecture descriptions to runtime
model examples and can be inspected with `lake exe torchlean graphspec`.

# How The Pieces Fit Together

The public constructors, runtime objects, specifications, and proofs are separate declarations, but
they refer to the same model structures and operator names. This lets application code remain short
while proof and backend files state exactly which semantics they use.

Read *Training From Scratch* for the first full training file, *Example Walkthroughs* for curated
commands, and *PyTorch Round Trip* then *TorchLean vs PyTorch* for interop.

# References

- TorchLean paper: https://arxiv.org/abs/2602.22631
- Lean module-system reference:
  https://lean-lang.org/doc/reference/latest/Source-Files-and-Modules/
- PyTorch documentation for the corresponding API concepts:
  https://pytorch.org/docs/stable/index.html
