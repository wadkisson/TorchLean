import VersoManual

open Verso.Genre Manual

#doc (Manual) "Execution Modes and Runners" =>
%%%
tag := "execution-modes"
%%%

By now we can build a model and give it data. The next question is how that same model runs.

Most TorchLean examples make three runtime choices: the scalar type, the execution backend, and the
device. These choices affect the artifact produced by the run. Eager mode produces a tape that is
easy to inspect. Compiled mode produces a reusable graph artifact. CUDA mode places supported
Float32 work on device buffers. The model architecture and parameter shapes remain fixed.

# The Three Runtime Choices

Most runnable examples answer three questions.

1. Which scalar type should the program use?
2. Which execution backend should run the graph?
3. Which device should hold the numeric buffers?

Typical command line flags look like:

```
lake env lean --run NN/Examples/Quickstart/SimpleMlpTrain.lean -- \
  --dtype float --backend eager --steps 100

lake env lean --run NN/Examples/Quickstart/SimpleMlpTrain.lean -- \
  --dtype float --backend compiled --steps 100
```

With CUDA enabled, model examples that support device buffers add a device choice:

```
lake -R -K cuda=true exe torchlean mlp --device cuda --steps 100
lake -R -K cuda=true exe torchlean mlp --device cuda --steps 1000 --cuda-mem-watch 100
```

The flags change how the model is evaluated. They do not change the layer structure or parameter
shapes. There is one public model definition; backend selection chooses the runtime artifact used to
run it. The `--cuda-mem-watch` flag is only a runtime diagnostic: it asks the example to print CUDA
allocator samples during training so that long runs expose memory drift while they are still
running. The public model examples use the same step-counted training convention here: `--steps`
means optimizer updates, and loader-based examples stop after that many updates rather than after an
accidental number of data-loader passes.

The runtime choices separate into four decisions:

- Scalar choice: `--dtype float` or `--dtype float32` changes the numeric representation. The model
  shape and architecture stay fixed.
- Backend choice: `--backend eager` or `--backend compiled` changes whether the run produces an
  eager tape or a reusable graph artifact.
- Device choice: `--device cpu` or `--device cuda` changes whether supported numeric buffers live on the host or
  on CUDA device memory.
- CUDA diagnostics: `--cuda-mem-watch N` samples native allocator state every `N` training updates.
  Long CUDA model runs choose a small default cadence so the terminal shows whether device memory is
  steady, growing slowly, or approaching exhaustion.
- Mode choice: train/eval mode changes runtime behavior for mode-sensitive layers such as dropout
  or normalization. The declared model and parameter payload stay visible.

# What The Public Runtime Owns

Most tutorial code does not instantiate a runner manually. It builds one trainer with a public
`Trainer.Config`, then lets `trainer.train data trainOptions` run with those scalar/backend/device
settings.

That public runtime path still creates one executable runtime object under the hood:

- the declared model stays the same,
- the runtime layer chooses the scalar/backend/device interpretation,
- the instantiated runtime object owns parameters, buffers, mode, and any compiled executable
  artifact.

Compiled execution is the same model run through a reusable runtime object. It is a backend choice,
not a second public forward API.

The quickstart API is:

- `Trainer.Config`
- `Trainer.TrainOptions`
- `Trainer.new`
- `trainer.train data trainOptions`
- `trained.predict ...` / `trained.printPrediction ...`

The manual runner API still exists, but it sits under
`Trainer.Manual`.  Use that path when you really need manual callbacks, explicit mode changes, or
custom runtime loops.

The same choice can be made in Lean code:

```
def eagerTrainer :=
  Trainer.new mkModel
    { task := .regression
      optimizer := optim.sgd { lr := 0.05 }
      dtype := .float
      backend := .eager }

def compiledTrainer :=
  Trainer.new mkModel
    { task := .regression
      optimizer := optim.sgd { lr := 0.05 }
      dtype := .float
      backend := .compiled }
```

The two trainer values differ in runtime policy, not in architecture. If their results disagree,
the disagreement belongs to the runtime/backend layer, not to a new model definition.

# Eager Mode

Eager mode is the most transparent backend while debugging. It records operations as they run, keeps
a tape of parent links and local reverse rules, and returns explicit gradients.

The closest PyTorch analogy is:

```
loss = model(x).loss(y)
loss.backward()
```

In TorchLean, the corresponding data remains explicit:

- the tape is a Lean value,
- gradients are returned as tensors or parameter bundles,
- widgets can inspect the recorded graph.

Use eager mode to understand one step, inspect a gradient, or explain what a small example is doing.

Eager mode is also the natural place to inspect failure. If a gradient is unexpectedly zero, or a
shape conversion is not doing what you think, run the small case eagerly first. A compiled graph is
better once you know which computation you want to repeat.

# Compiled Mode

Compiled mode builds a stable graph artifact before repeated execution. It is the
better default for longer runs when the model and loss are fixed, because the graph structure is
constructed once and then reused.

The closest PyTorch analogy is the intuition behind `torch.compile`: keep the model code, but first
turn its computation into a reusable graph. TorchLean's compiled path is more explicit because the
artifact is a Lean runtime object and can be related to the IR and proof layers.

Use compiled mode when the model and loss are fixed and the training loop will run many steps or
many batches.

Compiled mode should be read as a runtime transformation:

```
same model + same loss + fixed input/target shapes
  -> compiled executable artifact
  -> repeated evaluation
```

It is not a license to skip shape checks. The compiled artifact is valuable precisely because the
model, parameter layout, and input shapes have already been made explicit.

# Train Mode and Eval Mode

Some layers behave differently during training than they do while evaluating. Dropout and batch
normalization are the common examples. TorchLean keeps this state in the runner because these layers
are stateful at runtime.

This matches PyTorch's `model.train()` and `model.eval()` convention, but the mode is part of the
explicit runtime object.

A prediction call should therefore say whether it is using training or evaluation behavior when the
model contains mode-sensitive layers. This is a runtime distinction. It is not a new tensor shape,
and it is not a theorem about statistical performance.

# CPU, CUDA, and Float32

Device selection is separate from dtype selection. A CPU run over executable float32 semantics and a
CUDA run over host `Float` values exercise different runtime paths. Current CUDA-facing examples
use the supported Float32 buffer path and report when a requested combination is unsupported.

In code, the public shape is:

```
def cudaTrainer :=
  Trainer.new mkModel
    { task := .regression
      optimizer := optim.adam { lr := 0.01 }
      dtype := .float
      backend := .eager
      device := .cuda }
```

In command-line examples, the same choice appears as `--device cuda`. If an example says CUDA currently
requires `--dtype float`, read that as a runtime support constraint, not as a change to the
mathematical tensor type in the spec layer.

# Loaders and Epochs

Training calls run the same runner repeatedly over datasets or loaders. The runtime choice is
independent of whether examples arrive as one batch or many minibatches.

The public declarations to read first are:

- `Trainer.Config`
- `Trainer.TrainOptions`
- `Trainer.new`
- `trainer.train data trainOptions`

Lower level dataset and loader loops still exist for custom runtime code, but ordinary examples
should keep dtype, backend, device, and optimizer in the config passed to `Trainer.new`, and put
the step count and logging options in `Trainer.TrainOptions`.

# Same Architecture, Different Artifact

It helps to keep this small map in mind:

- `--dtype float` versus `--dtype float32` changes the scalar representation. Tensor shapes and
  model structure stay fixed.
- `--backend eager` changes the runtime artifact to tape style execution. The public model
  definition stays fixed.
- `--backend compiled` changes the runtime artifact to a reusable compiled graph. The public model
  definition stays fixed.
- `--device cpu` versus `--device cuda` changes where supported numeric buffers live. The Lean specification and
  declared runtime assumptions stay fixed.

That separation is the reason TorchLean can be used as a tutorial framework, an executable
experiment harness, and a verification codebase at the same time.

# Runnable Sources

For a small runnable example, open [SimpleMlpTrain](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleMlpTrain.lean).
For minibatches and epochs, open
[MinibatchMlpTrain](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/MinibatchMlpTrain.lean). For the declarations behind
the runner, open [NN.API.Runtime](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Runtime.lean).
