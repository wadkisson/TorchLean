import VersoManual

open Verso.Genre Manual

#doc (Manual) "Training" =>
%%%
tag := "training"
%%%


Training is a repeated state transition: parameters, optimizer memory, data order, and runtime
buffers all update together. TorchLean's high-level trainer packages that transition, while the
lower manual API exposes each piece.

We will train the running `2 → 8 → 1` MLP and then unpack what happened.

# The Smallest Complete Run

Execute:

```
lake exe torchlean quickstart_mlp \
  --device cpu --steps 200 --seed 2026
```

The current checkout reports:

```
dataset size = 25
mean_loss(before) = 0.761530
mean_loss(after) = 0.003234
heldout x=(0.25,-0.75), target=0.2,
prediction(after)=[0.210239]
```

The source is
[`SimpleMlpTrain.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleMlpTrain.lean).
Its public structure is:

```
model builder
  + trainer configuration
  + dataset
  + train options
  -> trained handle
```

The loss values are measurements from this run. The model and optimizer definitions, by contrast,
are reusable objects that can appear in theorem statements or another runtime profile.

# Declare The Architecture

```
import NN.API
open TorchLean

def model :
    nn.M (nn.Sequential (shape![2]) (shape![1])) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]
```

At this point we have:

- input and output shapes;
- layer structure;
- parameter shapes;
- seeded initialization actions.

We do not yet have a loss, optimizer, concrete parameter values, or device.

# Attach A Training Problem

```
def trainer (seed : Nat) :=
  Trainer.new model
    { task := .regression
      optimizer := optim.adam { lr := 0.03 }
      dtype := .float
      backend := .eager
      seed := seed }
```

`Trainer.new` runs the seeded builder and stores persistent choices. It still does not consume data
or update a parameter.

For regression, the default objective is mean-squared error. If a prediction and target each have
`n` entries:

$$`
L(\theta;x,y)
=
\frac1n\sum_{i=1}^{n}
\left(F_\theta(x)_i-y_i\right)^2.
`

Changing `.regression` to `.crossEntropy` changes the objective and target convention without
changing the architecture. A custom task supplies a checked scalar loss program.

# Build The Dataset

The quickstart uses a deterministic grid. A four-point example can be written directly:

```
def xs : Tensor.T Float (shape![4, 2]) :=
  tensor! [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
  ]

def ys : Tensor.T Float (shape![4, 1]) :=
  tensor! [[0.0], [1.0], [1.0], [0.0]]

def data : Trainer.Dataset (shape![2]) (shape![1]) :=
  Data.tensorDataset xs ys
```

The dataset type matches the model map `[2] → [1]`. A batched model would require a batched dataset
with a leading dimension in both item shapes.

# Call Train And Keep The Result

```
def run : IO Unit := do
  let trained ← (trainer 2026).train data
    { steps := 200
      batchSize := 4
      logEvery := 25 }

  trained.printSummary

  let heldout : Tensor.T Float (shape![2]) :=
    tensor! [0.25, -0.75]
  let yhat ← trained.predict heldout
  IO.println s!"prediction={Tensor.pretty yhat}"
```

# From Files To Typed Minibatches

A model type tells Lean the shape of one input and one output. A dataset must eventually provide
values of exactly those shapes, but real data begins in a less orderly form: rows in a CSV file,
arrays in an NPY file, text tokens, simulator output, or tensors generated on demand.

TorchLean treats loading as a boundary. Parsing and dimension checks happen before a value becomes
a typed training sample. Once the boundary succeeds, the training loop does not need to ask on
every step whether a row had the right number of columns.

# The Public Dataset Type

The public type is:

```
Trainer.Dataset inputShape targetShape
```

It describes one training item. The scalar type is intentionally absent. Its `build` field
materializes a concrete dataset after the trainer chooses `Float`, `IEEE32Exec`, or another
supported executable scalar.

This lets the same Float-authored data feed several scalar runtimes while keeping the conversion
visible at materialization. It also means a proof-level real tensor is not accidentally passed to
an IO training loop.

# Begin With Four Samples

The XOR table is small enough to see in full:

```
import NN.API
open TorchLean

def xs : Tensor.T Float (shape![4, 2]) :=
  tensor! [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
  ]

def ys : Tensor.T Float (shape![4, 1]) :=
  tensor! [[0.0], [1.0], [1.0], [0.0]]

def xorData : Trainer.Dataset (shape![2]) (shape![1]) :=
  Data.tensorDataset xs ys
```

The leading dimension of `xs` and `ys` is the sample count. `Data.tensorDataset` checks that both
counts agree and removes that leading axis from the item shapes:

```
whole input tensor    [4, 2]
one input sample         [2]

whole target tensor   [4, 1]
one target sample        [1]
```

Try changing the target annotation to `shape![3,1]` while leaving four rows in the literal. The
literal itself fails to elaborate. If the mismatch instead arrives from runtime files, the loader
returns an error before constructing the dataset.

`Data.samples`, `Data.singleton`, and `Data.floatSamples` serve list-backed or generated data.
`Data.regressionGrid` builds the deterministic grid used by the running MLP.

# The Public API

Most user programs need two lines:

```
import NN.API
open TorchLean
```

`NN.API` is the maintained application surface. It imports tensors, models, data, training,
optimizers, prediction, and explicit differentiation. It does not pull every proof, verifier,
floating-point implementation, widget, and backend-internal module into a small training file.

This chapter builds one program using only that public surface, then explains when a narrower or
lower import is appropriate.

# The Namespaces

The main public namespaces are:

| Namespace | Responsibility |
| --- | --- |
| `Tensor`, `Shape` | shape-indexed values and constructors |
| `nn` | layers, blocks, model families, functional operations |
| `Data` | datasets, loaders, batching, text and checkpoint helpers |
| `Trainer` | configuration, training, reports, prediction, manual loops |
| `optim` | optimizer configuration |
| `autograd` | function and model derivatives |
| `classical` | classical and statistical model APIs |

The lowercase `nn.linear`, `nn.relu`, and `optim.adam` names are the canonical public spellings.
Internal implementation namespaces may be longer because they distinguish specification, runtime,
and proof layers. Application code should not depend on those names unless it genuinely needs the
lower layer.

# Make A Scratch Program

Create `Scratch.lean` at the repository root:

```
import NN.API

open TorchLean

def model :
    nn.M (nn.Sequential (shape![2]) (shape![1])) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]

def xs : Tensor.T Float (shape![4, 2]) :=
  tensor! [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
  ]

def ys : Tensor.T Float (shape![4, 1]) :=
  tensor! [[0.0], [1.0], [1.0], [0.0]]

def data : Trainer.Dataset (shape![2]) (shape![1]) :=
  Data.tensorDataset xs ys

def trainer :=
  Trainer.new model
    { task := .regression
      optimizer := optim.adam { lr := 0.03 }
      dtype := .float
      backend := .eager
      seed := 2026 }

def main : IO Unit := do
  let trained ← trainer.train data
    { steps := 20
      batchSize := 4
      logEvery := 5 }
  trained.printSummary

  let heldout : Tensor.T Float (shape![2]) :=
    tensor! [0.25, -0.75]
  let yhat ← trained.predict heldout
  IO.println s!"prediction={Tensor.pretty yhat}"
```
