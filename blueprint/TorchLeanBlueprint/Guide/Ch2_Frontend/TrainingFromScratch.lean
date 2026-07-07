import VersoManual

open Verso.Genre Manual

#doc (Manual) "Training From Scratch" =>
%%%
tag := "training-from-scratch"
%%%

A TorchLean training file usually has five blocks: constants, model, trainer, data, and train/report.
Once you can recognize those blocks, most examples in the repository become readable.

The smallest workflow is:

1. define a typed model,
2. attach a loss by choosing a trainer,
3. choose an optimizer,
4. build or load a dataset,
5. run training and inspect the report.

No theorem is needed yet. The objects that later become graph artifacts and verification targets
should still feel like ordinary model code.

# A Minimal Complete Script

The shortest way to get a feel for the public API is to run the focused MLP example:

- [SimpleMlpTrain source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleMlpTrain.lean)
- [SimpleCnnTrain source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleCnnTrain.lean)
- [AutogradBasics source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/AutogradBasics.lean)

```
lake env lean --run NN/Examples/Quickstart/SimpleMlpTrain.lean
lake env lean --run NN/Examples/Quickstart/SimpleMlpTrain.lean -- --steps 200 --dtype float --backend eager
```

A few details to notice in that file:

- `import NN` is the supported tutorial entrypoint.
- `open TorchLean` gives you the public namespaces: `nn`, `Trainer`, `optim`, `Data`, and CLI parsers.
- `Trainer.new mkModel { task := .regression, seed := seed }` fixes initialization and returns the training task.
- The dataset is ordinary Lean data with tensor shapes.
- The report prints loss before and after training, then probes the trained model on a few inputs.

# Building Models With `nn`

Here is the core model definition pattern used across the tutorials:

```
import NN
open TorchLean

def mkModel : nn.M (nn.Sequential (Shape.vec 2) (Shape.vec 1)) :=
  nn.Sequential![
    nn.Linear 2 8,
    nn.ReLU,
    nn.Linear 8 1
  ]

def trainer (seed : Nat) :=
  Trainer.new mkModel { task := .regression, seed := seed }
```

Think of `nn.M` as a small “model builder” monad: it allocates parameters and wires layers
together, while keeping the end result purely functional.

## PyTorch Similarity

The PyTorch training pattern and the TorchLean training pattern are close by design:

```
import torch

model = torch.nn.Sequential(
    torch.nn.Linear(2, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 1),
)
opt = torch.optim.Adam(model.parameters(), lr=0.03)
```

TorchLean writes the same structure in Lean, but makes the parameter bundle explicit:

```
import NN
open TorchLean

def mkModel : nn.M (nn.Sequential (Shape.vec 2) (Shape.vec 1)) :=
  nn.Sequential![
    nn.Linear 2 8,
    nn.ReLU,
    nn.Linear 8 1
  ]

def trainer (seed : Nat) :=
  Trainer.new mkModel { task := .regression, seed := seed }
```

The loop has the same rhythm as PyTorch: forward pass, loss, backward pass, optimizer step.
TorchLean's difference is where the state lives. Parameters, optimizer state, runtime mode, and logs
are explicit values rather than hidden fields of a module object.

Here is a fuller public configuration:

```
def task :=
  Trainer.new mkModel
    { task := .regression
      optimizer := optim.adam { lr := 0.03, beta1 := 0.9, beta2 := 0.999 }
      dtype := .float
      backend := .compiled
      seed := 17 }

def trainOpts : Trainer.TrainOptions :=
  { steps := 200
    batchSize := 16
    logEvery := 25
    title := "two-layer regression" }
```

Persistent choices live on the trainer: task family, optimizer, scalar dtype, backend, seed, and
device/runtime options. Per-call choices live in `TrainOptions`: how many steps, how often to log,
where to write logs, and whether to load or save parameters.

That division keeps repeated experiments legible. If two runs differ only in `backend`, the
architecture and optimizer should be identical in the file. If two runs differ in `steps`, the
trainer should not have to be rebuilt.

# Workflow Lessons

The first training example is small, but it is already exercising the decisions that matter later:

- parameters are explicit data, so compilation and verification can find them without inspecting a
  hidden Python object;
- the model builder records structure separately from scalar execution, so the same architecture can
  be reused across backends;
- losses and optimizers are ordinary Lean functions over typed tensors, not implicit methods attached
  to mutable modules;
- the tutorial path and the verifier path share the same public model definitions instead of using
  unrelated examples.

A successful training run establishes that the model, data, loss, optimizer, and selected backend
execute together and produce a report. Convergence, robustness, and CUDA-correctness claims live in
later layers: optimizer theorems, certificate checkers, runtime boundary docs, or backend-specific
tests.

In a run log, read each line for what it actually says:

- a decreasing loss says the selected training loop made progress on the selected data;
- a saved parameter file says a runtime artifact was written in a known format;
- a backend parity check says two implementations agreed on tested inputs;
- a theorem in the proof layer says its exact Lean statement was proved;
- a certificate check says the finite certificate passed the checker.

All of these are useful. None of them silently upgrades into the others.

After the MLP example, move to a data-backed training tutorial:

- [CSV loader example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Csv.lean)
- [NPY loader example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Npy.lean)
- [CIFAR image loader example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Cifar10Images.lean)

If you want more than an MLP:

- [SimpleCnnTrain source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleCnnTrain.lean)
- [Vision CNN source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Vision/Cnn.lean)

# Autograd and Training

TorchLean’s differentiation is functional: tensors do not carry mutable `.grad` fields. Instead,
TorchLean provides `autograd` primitives that transform functions.

For the smallest autograd-only example, see:

- [AutogradBasics source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/AutogradBasics.lean)

The current tutorial files stay short by using the public API directly:

- `Data.*` builds shape-indexed datasets from small tensors, generated grids, CSV files, or image
  loaders.
- `Trainer.new` and `trainer.train` own the runtime scalar and backend
  selection.
- `Tensor`, `TensorPack`, and the `tensor!` / `tensorND!` / `tensorpack!` macros keep examples
  typed without making each tutorial reopen the runtime callback layer.

Public tutorials should describe the model, data, optimizer, and training options. The
`NN.API.Runtime` callback layer still exists for runtime implementers and proof work, but it is no
longer the path a first example has to copy.

Runnable examples that show the public style:

- [Float32 modes example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/DeepDives/Floats/Float32Modes.lean)
- [SimpleMlpTrain source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleMlpTrain.lean)

There is also a small public data layer that keeps examples self-contained without importing
external datasets or Python preprocessing:

- `Data.regressionGrid` for 2D regression grids,
- `Data.tensorDataset` for already-materialized input/target tensors,
- `Data.tabularCsvDataset` for small supervised CSV files,
- `Data.cifar10Dataset` and related loader APIs for image examples,
- `Trainer.Probe.*` constructors for before/after prediction reports.

That keeps the introductory tutorials focused on the training loop instead of on repetitive data setup.

## A Direct Loop Comparison

PyTorch usually looks like this:

```
for step in range(steps):
    opt.zero_grad()
    pred = model(x)
    loss = mse(pred, y)
    loss.backward()
    opt.step()
```

TorchLean’s equivalent is the same loop shape, but the gradients and the model state are explicit in
the API:

```
let loop ← Trainer.Manual.stepper (task := task) runner (optim.adam { lr := 0.03 })

for step in [0:steps] do
  let sample := getSample step
  let loss ← Trainer.Manual.step (task := task) loop sample
  if step % 25 = 0 then
    IO.println s!"step {step}: loss={loss}"
```

TorchLean keeps the training rhythm familiar while making the state explicit enough for later graph,
runtime, and proof chapters.

# The Anatomy Of A Training File

Most training examples have the same five blocks.

## 1. Constants

```
def inDim : Nat := 2
def hidden : Nat := 8
def outDim : Nat := 1
```

These are ordinary Lean definitions. They are not hidden inside a module object, so the shapes that
depend on them are visible to the type checker.

## 2. Model builder

```
def mkModel : nn.M (nn.Sequential (Shape.vec inDim) (Shape.vec outDim)) :=
  nn.Sequential![
    nn.Linear inDim hidden,
    nn.ReLU,
    nn.Linear hidden outDim
  ]
```

This definition is only the architecture. It does not train anything by itself.

## 3. Trainer

```
def trainer (seed : Nat) :=
  Trainer.new mkModel { task := .regression, seed := seed }
```

This line chooses the public model and loss family. For classification examples, use
`Trainer.new model { task := .classification }`.

## 4. Dataset or loader

```
let data : Trainer.Dataset (Shape.vec inDim) (Shape.vec outDim) :=
  Data.tensorDataset xs ys
```

For tiny examples this is an in memory dataset. For real examples it is often loaded from CSV or
NPY, then wrapped in a `Data.batchLoader`.

## 5. Train and report

```
let trainer := Trainer.new mkModel
  { task := .regression, optimizer := optim.adam { lr := 0.03 }, seed := seed }
let trained ← trainer.train data { steps := steps, batchSize := 16 }
trained.printSummary
let yhat ← trained.predict (tensorND! [2] [0.25, -0.75])
IO.println s!"heldout={yhat}"
```

The public shape stays stable even when the model family changes: define a model, define data,
choose a trainer, pass one config, call `trainer.train`, then reuse the trained handle for inference.

If a file needs callbacks, custom step scheduling, or runtime access for proofs, it can drop into
`Trainer.Manual`. That should read as manual code. The beginner path should remain the same
four-value shape: model, data, trainer, config.

The returned handle is the important object after training:

```
let trained ← trainer.train data trainOpts

trained.printSummary

let probe : Tensor.T Float (Shape.vec inDim) :=
  tensorND! [inDim] [0.25, -0.75]

let pred ← trained.predict probe
IO.println s!"prediction={pred}"
```

The original `trainer` still names the architecture and run configuration. The `trained` handle owns
the updated parameters and any runtime state needed for prediction.

A training file should make four things visible:

- the tensor shapes;
- the minibatch shape;
- the selected backend;
- the trained handle produced by training.

If you want the autograd side of the same example, jump to:

- [AutogradBasics source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/AutogradBasics.lean)

# Datasets and Minibatching

TorchLean supports fully in-memory datasets (TensorDataset-style), plus typed minibatching.

Examples:

- [MinibatchMlpTrain source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/MinibatchMlpTrain.lean)
- [CSV loader example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Csv.lean)
- [NPY loader example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Npy.lean)

For the data boundary contract (`.npy`, small numeric CSV, UTF-8 text), see:

- [NN/API/Data/README.md](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Data/README.md)
- [NN/Examples/Data/README.md](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/README.md)

# Step Streams

Not every training job is a finite dataset pass.  Some workloads produce the next batch from a
rule, a simulator, a replay buffer, or a file-backed token window.  For those cases TorchLean has a
typed step stream:

```
Trainer.Manual.StepBatchStream α inputShapes
```

A `StepBatchStream` is a function from the optimizer step number to the already-collated input list
for the module. The shape list is still checked by Lean. A stream for `[x, y]` samples and a stream
for `[tokens, targets]` samples have different types, even if both are loaded from external files.

The corresponding training APIs are:

```
Trainer.Manual.trainModuleStreamStepsWith
Trainer.Manual.trainModuleStreamStepsReport
Trainer.Manual.trainModuleStreamStepsCurveFloat
```

Use a loader when the dataset is a fixed finite table. Use a step stream when the batch is produced
on demand: RL rollouts, collocation points, generated windows from a large text file, or synthetic
diagnostic inputs. The public loop is model-agnostic; only the stream knows where the next
batch comes from.

The stream boundary is especially useful for scientific ML and reinforcement learning. In both
settings, the "dataset" may be a rule:

```
def collocationStream : Trainer.Manual.StepBatchStream Float inputShapes :=
  fun step => do
    -- Generate points from `step`, evaluate boundary conditions,
    -- and return tensors with the same checked shapes every time.
    pure batch
```

The shape list on the stream says what the training loop will receive. The generator decides where
the values come from.

# Explicit Training Loops

If you want something that looks closer to “manual PyTorch training code”, use the quickstart
training files directly. They keep the same explicit rhythm (build model, choose optimizer, run
steps), while the PyTorch interop folder stays focused on actual Python import/export:

- [SimpleMlpTrain source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleMlpTrain.lean)
- [SimpleCnnTrain source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleCnnTrain.lean)
- [AutogradBasics source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/AutogradBasics.lean)

Once you are comfortable running and training models, continue with “Runtime and Autograd” for the
execution modes (eager vs compiled graphs), what gets cached, and where the trust boundary sits.
