import VersoManual

open Verso.Genre Manual

#doc (Manual) "Datasets, Loaders, and Minibatches" =>
%%%
tag := "datasets-loaders"
%%%

One hard-coded tensor is enough to explain a loss or a gradient, but real examples need data
pipelines. TorchLean accepts in-memory samples, file-backed tensors, deterministic loaders, and
minibatches with shapes the model can see. The API stays close to `torch.utils.data`, while samples
carry Lean shapes.

The reader model is:

```
file or in-memory tensors
  -> typed dataset
  -> batch loader
  -> training loop
  -> report or saved curve
```

# Samples Have Shapes

A supervised sample has two tensors:

```
(x : Tensor alpha inputShape, y : Tensor alpha targetShape)
```

TorchLean packages that idea with the public `Data` and `sample` namespaces. In the MLP quickstart,
the dataset is built from two batched tensors:

```
import NN
open TorchLean

def X : Tensor.T Float (shape![25, 2]) :=
  Samples.squareGrid (-1.0) 1.0 5

def Y : Tensor.T Float (shape![25, 1]) :=
  Samples.regressionTargetsFloat X target

def dataset : Trainer.Dataset (Shape.vec 2) (Shape.vec 1) :=
  Data.tensorDataset X Y
```

The leading dimension is the sample dimension. The loader then turns per sample shapes into batched
shapes.

# TensorDataset Style Data

For small tutorials, the simplest path is fully in memory:

```
let dataset :=
  Data.Bands.dataset
```

Think of this as TorchLean's version of a small PyTorch `TensorDataset`: a finite dataset whose
elements are already tensors. It fits introductory examples, tests, and examples where the model or
proof interface matters more than file IO.

The image band dataset used by the CNN tutorial is exposed through the public `Data` API and is
used directly by
[NN.Examples.Quickstart.SimpleCnnTrain](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleCnnTrain.lean).

# File Sources

For real data, TorchLean uses small, predictable file contracts:

- `.npy` for numeric tensors;
- small numeric CSV for tabular examples;
- text files for sequence examples;
- conversion scripts for formats such as image folders, `.mat`, `.pt`, `.pth`, or `.npz`.

The public source types are:

- `Data.TensorSource` for one tensor file;
- `Data.SupervisedSource` for `(X, Y)` tensor files;
- `Data.LabeledSource` for inputs plus integer labels;
- `Data.TabularSupervisedSource` for numeric CSV with input and target columns.

Loading returns an error when the file or shape does not match the declared contract. A loader
failure is not delayed until the model sees a bad batch.

The exact constructor names vary by source, but the shape of a file-backed load looks like this:

```
def source : Data.SupervisedSource :=
  Data.SupervisedSource.ofPaths .npy "data/x.npy" "data/y.npy" 100 [2] [1]

def loadData : IO (Trainer.Dataset (Shape.vec 2) (Shape.vec 1)) := do
  match ← source.load (α := Float) with
  | .ok data => pure data
  | .error msg => throw <| IO.userError msg
```

That `Except String` boundary is not decoration. It is where a file-system object becomes a typed
training dataset. Once the loader succeeds, the trainer does not need to ask whether `x.npy` was
really two columns wide on every step.

For tabular CSV data, the same idea is column based:

```
def csvSource : Data.TabularSupervisedSource :=
  { path := "data/samples.csv"
    inDim := 2
    outDim := 1 }
```

The CSV convention is simple: each row contains the input columns followed by the target columns.
The table can be messy as a file, but the resulting dataset is not allowed to be vague. Either the
rows parse into the declared numeric shapes or the load fails with a concrete error.

The data contract references are:

- [NN/API/Data/README.md](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Data/README.md)
- [NN/Examples/Data/README.md](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/README.md)
- [CSV loader example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Csv.lean)
- [NPY loader example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Npy.lean)
- [CIFAR image loader example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Cifar10Images.lean)

# Minibatches and Epochs

The basic minibatch API is:

```
let loader :=
  Data.batchLoader dataset batch (shuffle := true) (seed := seed) (dropLast := true)
```

The arguments mean what a PyTorch reader expects:

- `batch` is the number of samples per minibatch;
- `shuffle := true` permutes the dataset at epoch boundaries;
- `seed` makes the order reproducible;
- `dropLast := true` keeps every batch at the same static shape.

That last point matters in Lean. A batch of size `5` has a different type from a batch of size `4`.
For tutorials that want one model type throughout training, `dropLast := true` is usually the clean
choice.

## Why `dropLast` Matters In Lean

In PyTorch, a final batch of size `3` after several batches of size `8` is usually fine. In
TorchLean, the batch dimension can appear in the type. If the model is written for
`Shape.mat 8 inDim`, then a final `Shape.mat 3 inDim` batch is not the same type. `dropLast := true`
keeps the tutorial simple by making every batch have the same static shape.

More flexible loaders can still be written, but then the file has to say how it handles the changing
batch dimension. The tradeoff is less implicit convenience and more visible shape information.

When a full epoch is needed directly, `Data.BatchLoader.epoch` materializes the batches and returns
the updated loader state. Most public examples stay one level higher and batch the dataset first:

```
let data := Data.batchDataset batch baseData (shuffle := true) (seed := seed)
let trained ← trainer.train data { steps := 200, batchSize := 16 }
trained.printSummary
```

The standard public path goes through `Data.batchDataset` and `trainer.train`. The loader still
exists under the hood, but the example does not need to own runner state, callbacks, or a separate
epoch loop.

# A Complete Minibatch Shape

Suppose a CSV row has:

- two input columns,
- one target column,
- and we train with `batch = 5`.

The per sample task is:

```
Shape.vec 2 -> Shape.vec 1
```

The minibatch model is:

```
Shape.mat 5 2 -> Shape.mat 5 1
```

For that reason, the quickstart writes:

```
def mkModel {batch : Nat} :
    nn.M (nn.Sequential (Shape.mat batch inDim) (Shape.mat batch outDim)) :=
  nn.Sequential![
    nn.Linear inDim hidDim,
    nn.ReLU,
    nn.Linear hidDim outDim
  ]
```

The model says it consumes a batch. The dataset says it contains individual samples. The loader
connects those two views.

This distinction is worth making explicit in code:

```
def perSample : Trainer.Dataset (Shape.vec 2) (Shape.vec 1) :=
  Data.tensorDataset xs ys

def batched : IO (Trainer.Dataset (Shape.mat 5 2) (Shape.mat 5 1)) := do
  Data.batchDataset 5 perSample (shuffle := true) (seed := 42)
```

The first value is a dataset of individual examples. The second value is a dataset of minibatches.
That is why a batched model has matrix-shaped inputs even though the original row had only two
features.

If the dataset size is not divisible by the batch size, a fixed-size typed batch must decide what to
do with the remainder. The beginner path drops the remainder. More advanced examples can pad,
bucket by length, or use a dynamic batch wrapper, but they must say so in the code.

# Hooks And Curves

Good runnable commands should still leave an artifact behind. The public trainer result writes a
two-point TrainLog when JSON logging is enabled, and it still exposes that same before/after summary
as its quick terminal report.

```
let trained ← trainer.train data
  { steps := 200
    log := .json outPath }
```

The returned `trained` value keeps the trained runtime handle alive. Public examples can immediately
run `trained.predict ...` without reopening the manual runner API.

Lower-level callbacks still exist for runtime-module tutorials and custom training loops:

```
Trainer.Manual.onTrainStart do
  Trainer.Manual.Report.reportMeanLoss (task := task) runner dataset "before"
```

The model examples also accept a log path through the shared CLI flags. The JSON log records the
quantities needed for later plots or checks, so the examples can answer the practical question: did
the model learn on the dataset we gave it?

# Text and Sequence Data

Text models use the same principle, but the sample builder is different. The sequence examples read
a corpus, tokenize or encode it, and create causal samples. The relevant files are:

- [NN/API/Text.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Text.lean)
- [NN/Examples/Models/Sequence/Transformer.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Sequence/Transformer.lean)
- [NN/Examples/Models/Sequence/TextGpt2.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Sequence/TextGpt2.lean)

The shapes are still the guide. A language model sample is a tensor with a sequence length and a
target convention, rather than an unstructured list of tokens.

# What the Data Layer Guarantees

The data layer gives the rest of the library a stable contract for examples and experiments:

- it checks that tensors entering training have the shapes the model expects;
- it keeps minibatches reproducible when the seed is fixed;
- it gives training loops stable batch shapes when the example requests them;
- it produces logs and reports that can be inspected later.

The data layer is deliberately modest: enough to train real examples, but small enough that the
reader can still see the path from file to tensor to model update.

# Data Is Evidence, Not A Proof

It is tempting to overstate what a clean training log says. The data layer can show that a file was
parsed, shapes matched, batches were reproducible, and loss moved during a run. Those are useful
runtime facts. They are not the same as a theorem about generalization, a proof of optimizer
convergence, or a certified robustness bound.

Later chapters use the same datasets and models as inputs to stronger checks. The point of Chapter 2
is to make sure the ordinary executable path is visible and typed before those stronger claims enter
the story.
