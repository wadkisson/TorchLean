import VersoManual

open Verso.Genre Manual

#doc (Manual) "Training, One State Transition At A Time" =>
%%%
tag := "training-from-scratch"
%%%

Training is not a magical property attached to a model. It is a repeated state transition involving
parameters, optimizer memory, data order, and runtime state. TorchLean's high-level trainer packages
that transition, while the lower manual API exposes each piece.

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

The returned handle retains:

- final parameters;
- runtime model state;
- before/after summary;
- prediction and public-verification closures.

Prediction accepts public Float tensors and performs the runtime scalar conversion selected by the
trainer. It does not rebuild or reinitialize the model.

# What One Update Computes

Let `θ_t` be the parameter pack at update `t`. A plain SGD update is:

$$`
g_t=\nabla_\theta L(\theta_t;x_t,y_t),
\qquad
\theta_{t+1}=\theta_t-\eta g_t.
`

Every symbol is structured:

- `θ_t` is a heterogeneous pack of tensors whose shapes are fixed by the model;
- `g_t` has exactly the same pack structure;
- `L` is a scalar tensor program;
- the subtraction and scaling occur coordinatewise in the selected scalar semantics.

An optimizer is therefore not merely a function from a flat vector to a flat vector. It owns
shape-aligned state.

# Adam's Hidden State Is Explicit

Adam maintains first and second moment estimates:

$$`
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
`

$$`
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
`

With bias correction:

$$`
\widehat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\widehat v_t=\frac{v_t}{1-\beta_2^t},
`

and the parameter update is:

$$`
\theta_{t+1}
=
\theta_t-\eta
\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
`

The two moment packs have the same dependent tensor shapes as `θ`. The step counter matters because
it changes the bias correction. Restoring only parameters from a checkpoint but not Adam state is
not the same continuation of training.

TorchLean also provides SGD, momentum SGD, AdamW, AdaGrad, RMSProp, Adadelta, and Muon-related
runtime configuration. Their public constructors share one trainer interface; their state and laws
remain optimizer-specific.

# What Does `steps` Count?

For the unbatched model `[2] → [1]`, the current in-memory loop interprets:

```
steps := 200
batchSize := 4
```

as 200 outer loop steps, each consuming a group of four samples. An optimizer update is currently
applied per sample in that group. Logging follows the outer step cadence.

For a true vectorized minibatch, define:

```
def batchedModel :
    nn.M
      (nn.Sequential
        (shape![5, 2])
        (shape![5, 1])) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]

def batchedData :
    Trainer.Dataset (shape![5, 2]) (shape![5, 1]) :=
  Data.batchDataset 5 data
    (shuffle := true)
    (seed := 2026)
```

Now one forward/backward operation sees five samples in one tensor. This can change reduction order,
optimizer cadence, and performance. It is not an optimization flag applied to the first model.

Run the maintained version:

```
lake exe torchlean quickstart_minibatch_mlp \
  --device cpu --batch 5 --steps 5 --seed 2026
```

The printed model confirms:

```
Sequential: [5, 2] -> [5, 1], layers=3, params=33
```

The parameter count remains 33 because linear layers preserve the batch prefix; the batch axis does
not create separate weights per sample.

# Eager And Compiled Are Execution Choices

Compare:

```
lake exe torchlean quickstart_mlp \
  --device cpu --backend eager --steps 20 --seed 2026

lake exe torchlean quickstart_mlp \
  --device cpu --backend compiled --steps 20 --seed 2026
```

Eager mode records operations and local reverse rules as they execute. Compiled mode builds a typed
forward/derivative graph and replays it with current parameters and inputs.

The public method remains `train`; compilation is a property of the configured runner, analogous to
wrapping a PyTorch model with an execution transform rather than renaming the model's `forward`
method.

Compiled trainer execution is currently CPU-only. A non-CPU compiled request is rejected rather
than silently falling back to a different semantics.

# Device Selection Is A Profile

Device is not just an enum passed to every tensor. The execution profile includes:

- target device and platform;
- provider preference;
- numerical and assurance policy;
- forward and VJP ownership;
- available kernel capsules.

Programmatically:

```
def cudaRun : Trainer.RunConfig :=
  ({ optimizer := optim.adam { lr := 0.03 }
     dtype := .float
     backend := .eager } :
    Trainer.RunConfig).cuda

def cudaTrainer :=
  Trainer.new model
    (Trainer.Config.fromRunConfig
      cudaRun .regression
      (seed := 2026))
```

A maintained CUDA run requires a CUDA-enabled build:

```
lake -R -K cuda=true exe torchlean quickstart_mlp \
  --device cuda --steps 20 --seed 2026 --show-backend
```

`--show-backend` prints the selected capsules as operations first execute. This is how a run reports
which provider owned matmul, ReLU, loss, and their VJPs.

Named future devices such as Metal, ROCm, TPU, or Trainium may be represented in configuration, but
`withDevice` fails if the current build has no maintained profile. A name in an enum is not an
implementation.

# Scalar Semantics Are Part Of The Run

The common executable selections are:

```
--dtype float
--dtype ieee754exec
```

Host `Float` is fast and relies on the platform runtime. `IEEE32Exec` is TorchLean's bit-level
binary32 reference and is much slower, but exposes precise finite and exceptional behavior.
Proof-level `Real` and finite rounded-real `FP32` are not executable trainer choices.

A loss curve without its scalar semantics is incomplete. The same architecture and seed may round
differently in binary32, binary64, a fused CUDA kernel, or an external provider.

# Save Enough State To Resume Honestly

`Trainer.TrainOptions` can select JSON logging and exact-bit parameter checkpoint paths. Persistent
configuration includes optimizer and runtime choices; per-call options include:

- number of steps;
- sample grouping and logging cadence;
- output path and notes;
- checkpoint load/save;
- CUDA allocator watch cadence.

For a faithful resume, preserve:

1. parameters;
2. optimizer state and step number;
3. data-loader or stream state;
4. model and preprocessing configuration;
5. scalar/backend/device profile;
6. RNG state where stochastic layers or sampling are used.

A parameter-only checkpoint is still useful for inference. It simply should not be described as an
exact continuation of Adam training.

# Manual Training

The high-level trainer is intended for common runs. `Trainer.Manual` exposes:

```
stepper
step
trainMode
evalMode
callbacks
loader loops
prediction
```

Use it when the program needs gradient accumulation, custom scheduling, multiple losses, generated
batches, reinforcement-learning interaction, or detailed instrumentation.

`Trainer.Manual.StepBatchStream α shapes` supplies already collated tensors as a function of the
step. PINN collocation points and simulator batches naturally fit this interface.

The lower API does not change the model or autograd semantics. It exposes the runner state that the
high-level trainer normally manages.

# Four Useful Experiments

## Initialization only

```
lake exe torchlean quickstart_mlp \
  --device cpu --steps 0 --seed 2026
```

This isolates initialization and the initial loss.

## Seed sensitivity

Run 20 steps with seeds `2026`, `2027`, and `2028`. Keep the dataset order fixed if you want to
study only initialization.

## Optimizer sensitivity

In the quickstart source, replace Adam with SGD while keeping the model, seed, and steps fixed.
Compare both the initial and final losses; the initial values should agree when initialization and
data order agree.

## Backend report

Add `--show-backend`. On CPU, inspect the reference capsules. On a CUDA build, inspect which native
capsules are selected and whether any trusted external provider appears.

# What Training Establishes

A successful run establishes that one configured pipeline executed:

```
data -> forward -> loss -> reverse pass -> optimizer -> parameters
```

It can produce valuable evidence:

- loss and prediction traces;
- exact parameter artifacts;
- capsule audit rows;
- reproducible configuration;
- runtime errors or successful completion.

It does not automatically prove:

- convergence for all initializations;
- generalization to unseen data;
- robustness to an input region;
- equality of eager, compiled, CUDA, and LibTorch paths;
- correctness of every native instruction.

TorchLean's contribution is not to rename those observations as proofs. It gives the run enough
structure that a theorem, numerical bound, backend contract, or verification certificate can refer
to the same model without erasing the boundary between them.
