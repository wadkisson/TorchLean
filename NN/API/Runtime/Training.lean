/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Core
public import NN.API.Rand
public import NN.API.TorchLean.ParamIO
public import NN.API.TorchLean.Schedulers
public import NN.GraphSpec.Models.TorchLean
public import NN.Runtime.Autograd.TorchLean
public import NN.Runtime.Autograd.Train.Dataset
public import NN.Runtime.RL
public import NN.Spec.RL.MDP
public import NN.Spec.RL.MarkovMDP
public import NN.Spec.RL.FiniteStochasticMDP
public import NN.API.Runtime.Core
public import NN.API.Runtime.Layers
public import NN.API.Runtime.Autograd
public import NN.API.Runtime.Module

import Mathlib.Algebra.Order.Algebra
import NN.Spec.Autograd.AutogradSpec

@[expose] public section

namespace NN
namespace API
namespace TorchLean

/-!
# Runtime Supervised Training

Supervised tasks, runners, steppers, optimizer configs, trainer aliases, and the low-level session
exports that back executable examples.
-/

namespace Supervised

/-
Supervised training helpers built directly on `ScalarModule`.

This is a slightly lower-level layer than `NN.API.Public.train`: it is designed around a
`SeqTask σ τ` (model + loss) and produces a `Runner` + `Stepper` that can be used in scripts.
-/

/-- Built-in loss choices for `SeqTask`. -/
inductive SeqLoss where
  | mse (reduction : API.TorchLean.Loss.Reduction := .mean)
  | crossEntropyOneHot (reduction : API.TorchLean.Loss.Reduction := .mean)

/-- A supervised task is just a model plus a choice of loss. -/
structure SeqTask (σ τ : Spec.Shape) where
  /-- Model to run. -/
  model : API.TorchLean.NN.Seq σ τ
  /-- Loss function. -/
  loss : SeqLoss

/--
Build a `ScalarModuleDef` for a task, choosing an explicit model mode (train/eval).

This is the underlying "instantiate me as a runnable module" step for training.
-/
def SeqTask.moduleDefWithMode {σ τ : Spec.Shape} (task : SeqTask σ τ)
    (mode : API.TorchLean.NN.Mode) :
    API.TorchLean.Module.ScalarModuleDef (API.TorchLean.NN.Seq.paramShapes task.model) [σ, τ] :=
  match task.loss with
  | .mse reduction =>
      API.TorchLean.NN.Seq.mseScalarModuleDefWithMode mode (model := task.model) (reduction :=
        reduction)
  | .crossEntropyOneHot reduction =>
      API.TorchLean.NN.Seq.crossEntropyOneHotScalarModuleDefWithMode mode
        (model := task.model) (reduction := reduction)

/-- Default module definition for a task (training mode). -/
def SeqTask.moduleDef {σ τ : Spec.Shape} (task : SeqTask σ τ) :
    API.TorchLean.Module.ScalarModuleDef (API.TorchLean.NN.Seq.paramShapes task.model) [σ, τ] :=
  task.moduleDefWithMode .train

namespace SeqTask

/-- Constructor: regression task (MSE loss). -/
def mse {σ τ : Spec.Shape} (model : API.TorchLean.NN.Seq σ τ)
    (reduction : API.TorchLean.Loss.Reduction := .mean) :
    SeqTask σ τ :=
  { model := model, loss := .mse reduction }

/-- Constructor: one-hot classification task (cross-entropy loss). -/
def crossEntropyOneHot {σ τ : Spec.Shape} (model : API.TorchLean.NN.Seq σ τ)
    (reduction : API.TorchLean.Loss.Reduction := .mean) :
    SeqTask σ τ :=
  { model := model, loss := .crossEntropyOneHot reduction }

end SeqTask

/-- Parameter shapes for a task (delegates to `Seq.paramShapes`). -/
abbrev paramShapes {σ τ : Spec.Shape} (task : SeqTask σ τ) : List Spec.Shape :=
  API.TorchLean.NN.Seq.paramShapes task.model

/--
Optimizer hyperparameter configuration for the supervised training helpers.

This configuration covers the optimizer choices exposed by the public training helpers. It mirrors
a few common PyTorch optimizers by name/defaults, but it does not try to cover the full option surface of
  `torch.optim.*`.
-/
inductive OptimizerConfig where
  /--
  SGD optimizer config.

  PyTorch analogy: `torch.optim.SGD(..., lr=..., momentum=...)` when `momentum > 0`,
  and plain SGD when `momentum = 0`.
  -/
  | sgd (lr : Float) (momentum : Float := 0.0)
  /-- Adam optimizer config. -/
  | adam (lr : Float) (beta1 : Float := 0.9) (beta2 : Float := 0.999) (epsilon : Float := 1e-8)
  /-- AdamW optimizer config (decoupled weight decay). -/
  | adamw (lr : Float) (weightDecay : Float := 0.01)
      (beta1 : Float := 0.9) (beta2 : Float := 0.999) (epsilon : Float := 1e-8)
  deriving Repr

/--
Step-based training configuration for `trainSamples` / `trainDataset`.

Fields:
- `steps`: number of parameter updates,
- `batchSize`: number of samples consumed by one public step for in-memory datasets,
- `optimizer`: optimizer hyperparameters,
- `scheduler`: optional learning-rate schedule (applied per step),
- `logEvery`: progress printing frequency (`0` disables logging).
-/
structure TrainConfig where
  /-- Number of training steps. -/
  steps : Nat
  /--
  Number of samples consumed by one public training step.

  The in-memory supervised loop applies one optimizer update per sample in the group, while
  reporting/logging at the outer step cadence. Loader-backed training uses the loader batches
  directly.
  -/
  batchSize : Nat := 1
  /-- Optimizer configuration. -/
  optimizer : OptimizerConfig := .sgd 0.01
  /-- Scheduler configuration. -/
  scheduler : Option API.TorchLean.Schedulers.Config := none
  /-- Log once every this many steps. -/
  logEvery : Nat := 1
  deriving Repr

/--
Small summary returned by lower training helpers.

By default, `before` and `after` are mean loss values, but the type is polymorphic so callers can
report other scalars in the same shape.
-/
structure TrainReport (α : Type) where
  /-- Metrics before training. -/
  before : α
  /-- Metrics after training. -/
  after : α

/--
Epoch-based training configuration for `trainLoader` (data-loader training).

Fields:
- `epochs`: number of epochs (each epoch iterates once over the loader),
- `optimizer`: optimizer hyperparameters,
- `scheduler`: optional learning-rate schedule (applied per step/epoch depending on helper),
- `logEvery`: progress printing frequency (`0` disables logging).
-/
structure LoaderTrainConfig where
  /-- Number of epochs to train for. -/
  epochs : Nat
  /-- Optimizer configuration. -/
  optimizer : OptimizerConfig := .sgd 0.01
  /-- Scheduler configuration. -/
  scheduler : Option API.TorchLean.Schedulers.Config := none
  /-- Log once every this many steps. -/
  logEvery : Nat := 1
  deriving Repr

/-- Extract the base learning rate encoded in an optimizer configuration. -/
def optimizerLR : OptimizerConfig → Float
  | .sgd lr _ => lr
  | .adam lr _ _ _ => lr
  | .adamw lr _ _ _ _ => lr

/--
Resolve the learning rate to use at a given training step.

If a scheduler is present, it takes precedence over the optimizer's baked-in base learning rate.
Otherwise this simply returns `optimizerLR cfg`.
-/
def stepLR (scheduler : Option API.TorchLean.Schedulers.Config) (cfg : OptimizerConfig)
    (step : Nat) : Float :=
  match scheduler with
  | some sched => API.TorchLean.Schedulers.lrAt sched step
  | none => optimizerLR cfg

/-- Map a state update over every optimizer-state entry in a shape-indexed parameter list. -/
def mapStateList {State : Type → Spec.Shape → Type} {α : Type} :
    {ss : List Spec.Shape} →
    ({s : Spec.Shape} → State α s → State α s) →
    API.TorchLean.Optim.StateList State α ss →
    API.TorchLean.Optim.StateList State α ss
  | [], _, .nil => .nil
  | _ :: ss, f, .cons st rest => .cons (f st) (mapStateList (ss := ss) f rest)

/-- Set the learning rate field of every Adam optimizer state entry to `lr`. -/
def adamStateWithLR {α : Type} (lr : α) {paramShapes : List Spec.Shape} :
    API.TorchLean.Optim.StateList _root_.Optim.Adam.State α paramShapes →
    API.TorchLean.Optim.StateList _root_.Optim.Adam.State α paramShapes :=
  mapStateList (ss := paramShapes) (fun st => { st with lr := lr })

/-- Set the learning rate field of every momentum-SGD optimizer state entry to `lr`. -/
def momentumSGDStateWithLR {α : Type} (lr : α) {paramShapes : List Spec.Shape} :
    API.TorchLean.Optim.StateList _root_.Optim.MomentumSGD.State α paramShapes →
    API.TorchLean.Optim.StateList _root_.Optim.MomentumSGD.State α paramShapes :=
  mapStateList (ss := paramShapes) (fun st => { st with lr := lr })

/-- Set the learning rate field of every AdamW optimizer state entry to `lr`. -/
def adamwStateWithLR {α : Type} (lr : α) {paramShapes : List Spec.Shape} :
    API.TorchLean.Optim.StateList _root_.Optim.AdamW.State α paramShapes →
    API.TorchLean.Optim.StateList _root_.Optim.AdamW.State α paramShapes :=
  mapStateList (ss := paramShapes) (fun st => { st with lr := lr })

/--
A fully instantiated supervised task runner.

This bundles:
- the imperative `ScalarModule` (parameters/buffers stored in refs),
- compiled predictors and loss functions for both `.train` and `.eval` modes (so switching mode is
  low-overhead),
- and the current mode stored in an `IO.Ref`.

The mode influences both operator behavior (e.g. dropout/batchnorm) and whether buffers are updated
during training.
-/
structure Runner (α : Type) [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    {σ τ : Spec.Shape} (task : SeqTask σ τ) where
  /-- Instantiated scalar module storing parameters/buffers in mutable refs. -/
  module : API.TorchLean.Module.ScalarModule α (paramShapes task) [σ, τ]
  /-- Compiled forward predictor specialized to training-mode behavior. -/
  predictorTrain : _root_.Runtime.Autograd.Torch.CompiledOut α (paramShapes task ++ [σ]) τ
  /-- Compiled forward predictor specialized to eval-mode behavior. -/
  predictorEval : _root_.Runtime.Autograd.Torch.CompiledOut α (paramShapes task ++ [σ]) τ
  /-- Compiled loss function for training-mode behavior. -/
  lossTrain : _root_.Runtime.Autograd.Torch.CompiledScalar α (paramShapes task ++ [σ, τ])
  /-- Compiled loss function for eval-mode behavior. -/
  lossEval : _root_.Runtime.Autograd.Torch.CompiledScalar α (paramShapes task ++ [σ, τ])
  /-- Mutable mode flag (`.train` / `.eval`) used by stateful layers (e.g. dropout/batchnorm). -/
  mode : IO.Ref API.TorchLean.NN.Mode

/--
Instantiate a `Runner` by explicitly providing a `Float → α` cast and backend.

Use this when you want to run the same task over different numeric backends (e.g. `Float` vs
`IEEE32Exec`) or when you want custom literal injection.
-/
def instantiateWithOptions {σ τ : Spec.Shape} (task : SeqTask σ τ)
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (cast : Float → α) (opts : API.TorchLean.Options := {}) :
    IO (Runner α task) := do
  let module ← API.TorchLean.Module.instantiateWithOptions (α := α) task.moduleDef cast opts
  let predictorTrain ← API.TorchLean.NN.Seq.compileOutWithMode .train (α := α) task.model
  let predictorEval ← API.TorchLean.NN.Seq.compileOutWithMode .eval (α := α) task.model
  let lossTrain ← API.TorchLean.Autodiff.compileLoss (α := α)
    (paramShapes := paramShapes task) (inputShapes := [σ, τ])
    (task.moduleDefWithMode .train).loss
  let lossEval ← API.TorchLean.Autodiff.compileLoss (α := α)
    (paramShapes := paramShapes task) (inputShapes := [σ, τ])
    (task.moduleDefWithMode .eval).loss
  let mode : IO.Ref API.TorchLean.NN.Mode ← IO.mkRef .eval
  pure {
    module := module
    predictorTrain := predictorTrain
    predictorEval := predictorEval
    lossTrain := lossTrain
    lossEval := lossEval
    mode := mode
  }

/--
Instantiate a `Runner` by explicitly providing a `Float → α` cast and a backend selector.

Instantiate a module with explicit runtime options.
-/
def instantiateWith {σ τ : Spec.Shape} (task : SeqTask σ τ)
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (cast : Float → α) (backend : API.TorchLean.Backend := .eager) :
    IO (Runner α task) := do
  instantiateWithOptions (task := task) (α := α) cast { backend := backend }

/--
Instantiate a `Runner` using the standard runtime literal injection `API.Runtime.ofFloat`.

This is the common entrypoint for executable examples.
-/
def instantiateWithRuntimeOptions {σ τ : Spec.Shape} (task : SeqTask σ τ)
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [API.Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (opts : API.TorchLean.Options := {}) :
    IO (Runner α task) :=
  instantiateWithOptions (task := task) (α := α) API.Runtime.ofFloat opts

/--
Instantiate a `Runner` using the standard runtime literal injection `API.Runtime.ofFloat` and a
backend selector.

Instantiate a module after parsing runtime options.
-/
def instantiate {σ τ : Spec.Shape} (task : SeqTask σ τ)
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [API.Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (backend : API.TorchLean.Backend := .eager) :
    IO (Runner α task) :=
  instantiateWithRuntimeOptions (task := task) (α := α) { backend := backend }

/--
Run a TorchLean task with CLI-style dtype/backend selection, then call `k` with a fully constructed
  runner.

This is used by `lake exe` entrypoints: `run` takes care of parsing dtype flags and instantiating
the underlying module/compiled programs.
-/
def run {σ τ : Spec.Shape} (task : SeqTask σ τ) (args : List String)
    (k :
      ∀ {α : Type}, [API.Semantics.Scalar α] → [DecidableEq Spec.Shape] → [ToString α] →
        [API.Runtime.Scalar α] →
        Runner α task → List String → IO Unit) :
    IO Unit := do
  API.TorchLean.Module.withModuleRuntime task.moduleDef args (fun {α} _ _ _ _ module rest => do
    let predictorTrain ← API.TorchLean.NN.Seq.compileOutWithMode .train (α := α) task.model
    let predictorEval ← API.TorchLean.NN.Seq.compileOutWithMode .eval (α := α) task.model
    let lossTrain ← API.TorchLean.Autodiff.compileLoss (α := α)
      (paramShapes := paramShapes task) (inputShapes := [σ, τ])
      (task.moduleDefWithMode .train).loss
    let lossEval ← API.TorchLean.Autodiff.compileLoss (α := α)
      (paramShapes := paramShapes task) (inputShapes := [σ, τ])
      (task.moduleDefWithMode .eval).loss
    let mode : IO.Ref API.TorchLean.NN.Mode ← IO.mkRef .eval
    k (α := α)
      { module := module
        predictorTrain := predictorTrain
        predictorEval := predictorEval
        lossTrain := lossTrain
        lossEval := lossEval
        mode := mode }
      rest
  )

/-- Read the current parameter list from a runner. -/
def params {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) :
    IO (API.TorchLean.TensorPack α (paramShapes task)) :=
  API.TorchLean.Module.params runner.module

/-- Read the runner's current mode (`.train` or `.eval`). -/
def mode {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) : IO API.TorchLean.NN.Mode :=
  runner.mode.get

/-- Set the runner mode (`.train` or `.eval`). -/
def setMode {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (value : API.TorchLean.NN.Mode) : IO Unit :=
  runner.mode.set value

/-- Convenience: `setMode runner .train`. -/
def trainMode {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) : IO Unit :=
  setMode runner .train

/-- Convenience: `setMode runner .eval`. -/
def evalMode {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) : IO Unit :=
  setMode runner .eval

/-- Predicate: are we in training mode? -/
def isTraining {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) : IO Bool :=
  do
    pure ((← mode runner) == .train)

/-- Pick the predictor compiled for the runner's current mode (`.train` or `.eval`). -/
def activePredictor {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) :
    IO (_root_.Runtime.Autograd.Torch.CompiledOut α (paramShapes task ++ [σ]) τ) := do
  match (← mode runner) with
  | .train => pure runner.predictorTrain
  | .eval => pure runner.predictorEval

/-- Pick the loss program compiled for the runner's current mode (`.train` or `.eval`). -/
def activeLoss {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) :
    IO (_root_.Runtime.Autograd.Torch.CompiledScalar α (paramShapes task ++ [σ, τ])) := do
  match (← mode runner) with
  | .train => pure runner.lossTrain
  | .eval => pure runner.lossEval

/--
Refresh mode-dependent runner buffers using one supervised sample.

This mutates the module parameters only in `.train` mode, mirroring PyTorch-style buffer updates
for layers such as normalization. In `.eval` mode it is a no-op.
-/
def updateRunnerBuffers {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (sample : API.TorchLean.TensorPack α [σ, τ]) : IO Unit := do
  let currentMode ← mode runner
  if currentMode == .train then
    match sample with
    | .cons x (.cons _y .nil) => do
        let ps ← params runner
        let ps' ← API.TorchLean.NN.Seq.updateBuffers currentMode task.model ps x
        API.TorchLean.Module.setParams runner.module ps'
  else
    pure ()

/--
Run one forward/backward pass on a single supervised sample and return gradients for all parameters.

This is the TorchLean analogue of the `loss.backward()` payload in PyTorch, except TorchLean returns
the gradients explicitly instead of storing them in `.grad` fields.
-/
def backward {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (sample : API.TorchLean.TensorPack α [σ, τ]) :
    IO (API.TorchLean.TensorPack α (paramShapes task)) := do
  -- The instantiated scalar module always uses the training-mode program; keep the runner mode
  -- aligned so `updateRunnerBuffers` is not accidentally skipped.
  trainMode runner
  updateRunnerBuffers runner sample
  API.TorchLean.Module.backward runner.module sample

/-- Predict on one input tensor using the runner's active mode (`.train` or `.eval`). -/
def predict {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (x : Spec.Tensor α σ) :
    IO (Spec.Tensor α τ) := do
  let ps ← params runner
  let predictor ← activePredictor runner
  pure (API.TorchLean.NN.Seq.predict1 task.model predictor ps x)

/-- Predict on a list of inputs by repeatedly calling `predict`. -/
def predictBatch {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (xs : List (Spec.Tensor α σ)) :
    IO (List (Spec.Tensor α τ)) := do
  let ps ← params runner
  let predictor ← activePredictor runner
  pure <| xs.map (API.TorchLean.NN.Seq.predict1 task.model predictor ps)

/-- For classification heads: run `predict`, then take `argmax` over the logits (if defined). -/
def predictClass? {σ : Spec.Shape} {n : Nat} {task : SeqTask σ (.dim n .scalar)}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (x : Spec.Tensor α σ) :
    IO (Option (Fin n)) := do
  let logits ← predict runner x
  pure <| API.TorchLean.Metrics.argmax? (α := α) (n := n) logits

/-- Compute `(correct, total)` for a one-hot classification dataset. -/
def accuracyOneHot {σ : Spec.Shape} {n : Nat} {task : SeqTask σ (.dim n .scalar)}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (samples : List (API.TorchLean.TensorPack α [σ, .dim n .scalar])) :
    IO (Nat × Nat) := do
  let rec go (correct total : Nat) :
      List (API.TorchLean.TensorPack α [σ, .dim n .scalar]) → IO (Nat × Nat)
    | [] => pure (correct, total)
    | sample :: rest =>
        do
          let (x, y) :=
            match sample with
            | .cons x (.cons y .nil) => (x, y)
          let logits ← predict runner x
          let ok := API.TorchLean.Metrics.correctOneHot? (α := α) (n := n) logits y
          go (if ok = some true then correct + 1 else correct) (total + 1) rest
  go 0 0 samples

/-- Mean scalar loss over a list of supervised samples (uses the runner's active mode). -/
def meanLoss {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    (runner : Runner α task) (samples : List (API.TorchLean.TensorPack α [σ, τ])) :
    IO α := do
  let compiled ← activeLoss runner
  let ps ← params runner
  let values ← samples.mapM (fun sample => do
    let args : API.TorchLean.TensorPack α (paramShapes task ++ [σ, τ]) :=
      _root_.Runtime.Autograd.Torch.Proofs.Autograd.Algebra.TList.append
        (α := α) (ss₁ := paramShapes task) (ss₂ := [σ, τ]) ps sample
    pure (Spec.Tensor.toScalar <| _root_.Runtime.Autograd.Torch.CompiledScalar.forward compiled
      args))
  match values with
  | [] => pure 0
  | xs => pure (xs.foldl (· + ·) 0 / (xs.length : α))

/-- Mean scalar loss over a dataset (materialized via `dataset.toList`). -/
def meanLossDataset {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    (runner : Runner α task)
    (dataset : _root_.Runtime.Autograd.Train.Dataset (API.TorchLean.TensorPack α [σ, τ])) :
    IO α :=
  meanLoss runner dataset.toList

/-- Scalar loss for one sample through the instantiated runtime module. -/
def moduleLoss {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (sample : API.TorchLean.TensorPack α [σ, τ]) : IO α := do
  let loss ← API.TorchLean.Module.forward runner.module sample
  pure (Spec.Tensor.toScalar loss)

/-- Treat `0` as the conservative single-sample step size. -/
def effectiveTrainBatchSize (n : Nat) : Nat :=
  if n = 0 then 1 else n

/-- Take the next cyclic group of samples from an in-memory training set. -/
def nextCyclicBatch {α : Type} (context : String)
    (samples : List α) (restRef : IO.Ref (List α)) (batchSize : Nat) : IO (List α) := do
  if samples.isEmpty then
    throw <| IO.userError s!"{context}: empty sample cycle"
  let mut batch : List α := []
  for _ in [0:batchSize] do
    let mut rest ← restRef.get
    if rest.isEmpty then
      rest := samples
    match rest with
    | [] => throw <| IO.userError s!"{context}: empty sample cycle"
    | sample :: rest' =>
        restRef.set rest'
        batch := sample :: batch
  pure batch.reverse

/--
Train on a small in-memory list of supervised samples for a fixed number of steps.

This is the simplest training-loop helper: it is intended for examples and small synthetic datasets.
For loader-based training, see `trainLoader`.
-/
def trainSamples {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [API.Runtime.Scalar α]
    (runner : Runner α task) (cfg : TrainConfig) (samples : List (API.TorchLean.TensorPack α [σ, τ])) :
    IO (TrainReport α) := do
  trainMode runner
  let before ← meanLoss runner samples
  unless samples.isEmpty do
    let batchSize := effectiveTrainBatchSize cfg.batchSize
    let restRef ← IO.mkRef samples
    let nextBatch : IO (List (API.TorchLean.TensorPack α [σ, τ])) :=
      nextCyclicBatch "Supervised.train" samples restRef batchSize
    let logBatchLoss (stepIdx : Nat) (batch : List (API.TorchLean.TensorPack α [σ, τ])) :
        IO Unit := do
      if cfg.logEvery > 0 && stepIdx % cfg.logEvery = 0 then
        let loss ← meanLoss runner batch
        IO.println s!"step {stepIdx}: loss={loss}"
    match cfg.optimizer with
    | .sgd lr momentum =>
        if momentum == 0.0 then
          for stepIdx in [0:cfg.steps] do
            let batch ← nextBatch
            let lrα := API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer stepIdx)
            for sample in batch do
              updateRunnerBuffers runner sample
              API.TorchLean.Module.step runner.module lrα sample
            logBatchLoss stepIdx batch
        else
          let opt := API.TorchLean.Optim.momentumSGD
            (α := α) (paramShapes := paramShapes task)
            (lr := API.Runtime.ofFloat lr) (momentum := API.Runtime.ofFloat momentum)
          let st0 : API.TorchLean.Optim.StateList _root_.Optim.MomentumSGD.State α
            (paramShapes task) ←
            API.TorchLean.Module.initOptim runner.module opt
          let mut st := st0
          for stepIdx in [0:cfg.steps] do
            let batch ← nextBatch
            let lrα := API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer stepIdx)
            st := momentumSGDStateWithLR (paramShapes := paramShapes task) lrα st
            for sample in batch do
              updateRunnerBuffers runner sample
              st ← API.TorchLean.Module.stepWith runner.module opt st sample
            logBatchLoss stepIdx batch
    | .adam lr beta1 beta2 epsilon =>
        let opt := API.TorchLean.Optim.adam
          (α := α) (paramShapes := paramShapes task)
          (lr := API.Runtime.ofFloat lr)
          (beta1 := API.Runtime.ofFloat beta1)
          (beta2 := API.Runtime.ofFloat beta2)
          (epsilon := API.Runtime.ofFloat epsilon)
        let st0 : API.TorchLean.Optim.StateList _root_.Optim.Adam.State α (paramShapes task) ←
          API.TorchLean.Module.initOptim runner.module opt
        let mut st := st0
        for stepIdx in [0:cfg.steps] do
          let batch ← nextBatch
          let lrα := API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer stepIdx)
          st := adamStateWithLR (paramShapes := paramShapes task) lrα st
          for sample in batch do
            updateRunnerBuffers runner sample
            st ← API.TorchLean.Module.stepWith runner.module opt st sample
          logBatchLoss stepIdx batch
    | .adamw lr weightDecay beta1 beta2 epsilon =>
        let opt := API.TorchLean.Optim.adamw
          (α := α) (paramShapes := paramShapes task)
          (lr := API.Runtime.ofFloat lr) (weightDecay := API.Runtime.ofFloat weightDecay)
          (beta1 := API.Runtime.ofFloat beta1)
          (beta2 := API.Runtime.ofFloat beta2)
          (epsilon := API.Runtime.ofFloat epsilon)
        let st0 : API.TorchLean.Optim.StateList _root_.Optim.AdamW.State α (paramShapes task) ←
          API.TorchLean.Module.initOptim runner.module opt
        let mut st := st0
        for stepIdx in [0:cfg.steps] do
          let batch ← nextBatch
          let lrα := API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer stepIdx)
          st := adamwStateWithLR (paramShapes := paramShapes task) lrα st
          for sample in batch do
            updateRunnerBuffers runner sample
            st ← API.TorchLean.Module.stepWith runner.module opt st sample
          logBatchLoss stepIdx batch
  let after ← meanLoss runner samples
  pure { before := before, after := after }

/-- Train over a dataset by materializing it as a list. -/
def trainDataset {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [API.Runtime.Scalar α]
    (runner : Runner α task) (cfg : TrainConfig)
    (dataset : _root_.Runtime.Autograd.Train.Dataset (API.TorchLean.TensorPack α [σ, τ])) :
    IO (TrainReport α) :=
  trainSamples runner cfg dataset.toList

/--
Train over a `DataLoader` for `cfg.epochs` epochs, returning the final report and updated loader.

This corresponds to the common PyTorch pattern:
`for epoch in ...: for batch in loader: step(batch)`.
-/
def trainLoader {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [API.Runtime.Scalar α]
    (runner : Runner α task) (cfg : LoaderTrainConfig)
    (dl : _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ])) :
    IO (TrainReport α × _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ])) := do
  trainMode runner
  let nextEpoch
      (loader : _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ])) :
      IO (_root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ]) ×
        List (List (API.TorchLean.TensorPack α [σ, τ]))) :=
    match _root_.Runtime.Autograd.Train.DataLoader.epoch "Supervised.trainLoader" loader with
    | .ok out => pure out
    | .error msg => throw <| IO.userError s!"Supervised.trainLoader: {msg}"

  let before ← meanLossDataset runner dl.dataset

  match cfg.optimizer with
  | .sgd lr momentum =>
      if momentum == 0.0 then
        let rec trainSgdBatches
            (epoch : Nat)
            (batches : List (List (API.TorchLean.TensorPack α [σ, τ]))) :
            IO Unit := do
          match batches with
          | [] => pure ()
          | batch :: rest =>
              let lrα := API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer epoch)
              for sample in batch do
                updateRunnerBuffers runner sample
                API.TorchLean.Module.step runner.module lrα sample
              trainSgdBatches epoch rest

        let rec runSgdEpochs (remaining : Nat)
            (epoch : Nat)
            (loader : _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ])) :
            IO (_root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ])) := do
          match remaining with
          | 0 => pure loader
          | n + 1 =>
              let (loader', batches) ← nextEpoch loader
              trainSgdBatches epoch batches
              runSgdEpochs n (epoch + 1) loader'

        let dl' ← runSgdEpochs cfg.epochs 0 dl
        let after ← meanLossDataset runner dl'.dataset
        pure ({ before := before, after := after }, dl')
      else
        let opt := API.TorchLean.Optim.momentumSGD
          (α := α) (lr := API.Runtime.ofFloat lr) (momentum := API.Runtime.ofFloat momentum)
        let st0 ← API.TorchLean.Module.initOptim runner.module opt

        let rec trainMomSamples (epoch stepIdx : Nat) (state : opt.State)
            (samples : List (API.TorchLean.TensorPack α [σ, τ])) :
            IO (opt.State × Nat) := do
          match samples with
          | [] => pure (state, stepIdx)
          | sample :: rest =>
              updateRunnerBuffers runner sample
              let state' ← API.TorchLean.Module.stepWith runner.module opt state sample
              if cfg.logEvery > 0 && stepIdx % cfg.logEvery = 0 then
                let loss ← API.TorchLean.Module.forward runner.module sample
                IO.println s!"step {epoch}:{stepIdx}: loss={Spec.Tensor.toScalar loss}"
              trainMomSamples epoch (stepIdx + 1) state' rest

        let rec trainMomBatches (epoch stepIdx : Nat) (state : opt.State)
            (batches : List (List (API.TorchLean.TensorPack α [σ, τ]))) :
            IO (opt.State × Nat) := do
          match batches with
          | [] => pure (state, stepIdx)
          | batch :: rest =>
              let (state', stepIdx') ← trainMomSamples epoch stepIdx state batch
              trainMomBatches epoch stepIdx' state' rest

        let rec runMomEpochs (epoch remaining : Nat)
            (loader : _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ]))
            (st : opt.State) :
            IO (_root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ]) × opt.State)
              := do
          match remaining with
          | 0 => pure (loader, st)
          | n + 1 =>
              let (loader', batches) ← nextEpoch loader
              let stSched : opt.State :=
                match cfg.scheduler with
                | none => st
                | some _ =>
                    momentumSGDStateWithLR
                      (paramShapes := paramShapes task)
                      (API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer epoch))
                      st
              let (st', _) ← trainMomBatches epoch 0 stSched batches
              runMomEpochs (epoch + 1) n loader' st'

        let (dl', _) ← runMomEpochs 0 cfg.epochs dl st0
        let after ← meanLossDataset runner dl'.dataset
        pure ({ before := before, after := after }, dl')

  | .adam lr beta1 beta2 epsilon =>
      let opt := API.TorchLean.Optim.adam
        (α := α) (lr := API.Runtime.ofFloat lr)
        (beta1 := API.Runtime.ofFloat beta1)
        (beta2 := API.Runtime.ofFloat beta2)
        (epsilon := API.Runtime.ofFloat epsilon)
      let st0 ← API.TorchLean.Module.initOptim runner.module opt

      let rec trainAdamSamples (epoch stepIdx : Nat) (state : opt.State)
          (samples : List (API.TorchLean.TensorPack α [σ, τ])) :
          IO (opt.State × Nat) := do
        match samples with
        | [] => pure (state, stepIdx)
        | sample :: rest =>
            updateRunnerBuffers runner sample
            let state' ← API.TorchLean.Module.stepWith runner.module opt state sample
            if cfg.logEvery > 0 && stepIdx % cfg.logEvery = 0 then
              let loss ← API.TorchLean.Module.forward runner.module sample
              IO.println s!"step {epoch}:{stepIdx}: loss={Spec.Tensor.toScalar loss}"
            trainAdamSamples epoch (stepIdx + 1) state' rest

      let rec trainAdamBatches (epoch stepIdx : Nat) (state : opt.State)
          (batches : List (List (API.TorchLean.TensorPack α [σ, τ]))) :
          IO (opt.State × Nat) := do
        match batches with
        | [] => pure (state, stepIdx)
        | batch :: rest =>
            let (state', stepIdx') ← trainAdamSamples epoch stepIdx state batch
            trainAdamBatches epoch stepIdx' state' rest

      let rec runAdamEpochs (epoch remaining : Nat)
          (loader : _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ]))
          (st : opt.State) :
          IO (_root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ]) × opt.State)
            := do
        match remaining with
        | 0 => pure (loader, st)
        | n + 1 =>
            let (loader', batches) ← nextEpoch loader
            let stSched : opt.State :=
              match cfg.scheduler with
              | none => st
              | some _ =>
                  adamStateWithLR
                    (paramShapes := paramShapes task)
                    (API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer epoch))
                    st
            let (st', _) ← trainAdamBatches epoch 0 stSched batches
            runAdamEpochs (epoch + 1) n loader' st'

      let (dl', _) ← runAdamEpochs 0 cfg.epochs dl st0
      let after ← meanLossDataset runner dl'.dataset
      pure ({ before := before, after := after }, dl')

  | .adamw lr weightDecay beta1 beta2 epsilon =>
      let opt := API.TorchLean.Optim.adamw
        (α := α) (lr := API.Runtime.ofFloat lr) (weightDecay := API.Runtime.ofFloat weightDecay)
        (beta1 := API.Runtime.ofFloat beta1)
        (beta2 := API.Runtime.ofFloat beta2)
        (epsilon := API.Runtime.ofFloat epsilon)
      let st0 ← API.TorchLean.Module.initOptim runner.module opt

      let rec trainAdamWSamples (epoch stepIdx : Nat) (state : opt.State)
          (samples : List (API.TorchLean.TensorPack α [σ, τ])) :
          IO (opt.State × Nat) := do
        match samples with
        | [] => pure (state, stepIdx)
        | sample :: rest =>
            updateRunnerBuffers runner sample
            let state' ← API.TorchLean.Module.stepWith runner.module opt state sample
            if cfg.logEvery > 0 && stepIdx % cfg.logEvery = 0 then
              let loss ← API.TorchLean.Module.forward runner.module sample
              IO.println s!"step {epoch}:{stepIdx}: loss={Spec.Tensor.toScalar loss}"
            trainAdamWSamples epoch (stepIdx + 1) state' rest

      let rec trainAdamWBatches (epoch stepIdx : Nat) (state : opt.State)
          (batches : List (List (API.TorchLean.TensorPack α [σ, τ]))) :
          IO (opt.State × Nat) := do
        match batches with
        | [] => pure (state, stepIdx)
        | batch :: rest =>
            let (state', stepIdx') ← trainAdamWSamples epoch stepIdx state batch
            trainAdamWBatches epoch stepIdx' state' rest

      let rec runAdamWEpochs (epoch remaining : Nat)
          (loader : _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ]))
          (st : opt.State) :
          IO (_root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ]) × opt.State)
            := do
        match remaining with
        | 0 => pure (loader, st)
        | n + 1 =>
            let (loader', batches) ← nextEpoch loader
            let stSched : opt.State :=
              match cfg.scheduler with
              | none => st
              | some _ =>
                  adamwStateWithLR
                    (paramShapes := paramShapes task)
                    (API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer epoch))
                    st
            let (st', _) ← trainAdamWBatches epoch 0 stSched batches
            runAdamWEpochs (epoch + 1) n loader' st'

      let (dl', _) ← runAdamWEpochs 0 cfg.epochs dl st0
      let after ← meanLossDataset runner dl'.dataset
      pure ({ before := before, after := after }, dl')

/--
Stateful training loop object: a `Runner` plus an optimizer state and a step counter.

This is the TorchLean analogue of holding a PyTorch `optimizer` object plus the model, ready to
`step()` on batches.
-/
structure Stepper (α : Type) [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    {σ τ : Spec.Shape} (task : SeqTask σ τ) where
  /-- Underlying task runner (module + compiled predictors/losses). -/
  runner : Runner α task
  /-- Run a single optimization step on one supervised sample, returning the loss value. -/
  stepSample : API.TorchLean.TensorPack α [σ, τ] → IO α
  /-- Run an epoch over an explicit list of samples, returning the per-step loss values. -/
  epochSamples : List (API.TorchLean.TensorPack α [σ, τ]) → IO (List α)
  /-- Read the total number of `stepSample` calls performed so far. -/
  stepCount : IO Nat

/--
Construct a `Stepper` for a runner, optimizer config, and optional scheduler.

This is the recommended way to build custom training loops without reimplementing the optimizer
logic: call `stepper`, then choose `stepSample` for single batches or `epochSamples` for explicit
sample lists.
-/
def stepper {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α] [API.Runtime.Scalar α]
    (runner : Runner α task) (optimizer : OptimizerConfig)
    (scheduler : Option API.TorchLean.Schedulers.Config := none) :
    IO (Stepper α task) := do
  trainMode runner
  let stepRef ← IO.mkRef 0
  match optimizer with
  | .sgd lr momentum =>
      if momentum == 0.0 then
        let runStep := fun (sample : API.TorchLean.TensorPack α [σ, τ]) => do
          trainMode runner
          let stepIdx ← stepRef.get
          let loss ← moduleLoss runner sample
          updateRunnerBuffers runner sample
          let lrα := API.Runtime.ofFloat (stepLR scheduler optimizer stepIdx)
          API.TorchLean.Module.step runner.module lrα sample
          stepRef.set (stepIdx + 1)
          pure loss
        pure {
          runner := runner
          stepSample := runStep
          epochSamples := fun samples => samples.mapM runStep
          stepCount := stepRef.get
        }
      else
        let opt := API.TorchLean.Optim.momentumSGD
          (α := α) (paramShapes := paramShapes task)
          (lr := API.Runtime.ofFloat lr) (momentum := API.Runtime.ofFloat momentum)
        let st0 : API.TorchLean.Optim.StateList _root_.Optim.MomentumSGD.State α (paramShapes task)
          ←
          API.TorchLean.Module.initOptim runner.module opt
        let stRef ← IO.mkRef st0
        let runStep := fun (sample : API.TorchLean.TensorPack α [σ, τ]) => do
          trainMode runner
          let stepIdx ← stepRef.get
          let loss ← moduleLoss runner sample
          updateRunnerBuffers runner sample
          let lrα := API.Runtime.ofFloat (stepLR scheduler optimizer stepIdx)
          let st0 ← stRef.get
          let st := momentumSGDStateWithLR (paramShapes := paramShapes task) lrα st0
          let st' ← API.TorchLean.Module.stepWith runner.module opt st sample
          stRef.set st'
          stepRef.set (stepIdx + 1)
          pure loss
        pure {
          runner := runner
          stepSample := runStep
          epochSamples := fun samples => samples.mapM runStep
          stepCount := stepRef.get
        }
  | .adam lr beta1 beta2 epsilon =>
      let opt := API.TorchLean.Optim.adam
        (α := α) (paramShapes := paramShapes task)
        (lr := API.Runtime.ofFloat lr)
        (beta1 := API.Runtime.ofFloat beta1)
        (beta2 := API.Runtime.ofFloat beta2)
        (epsilon := API.Runtime.ofFloat epsilon)
      let st0 : API.TorchLean.Optim.StateList _root_.Optim.Adam.State α (paramShapes task) ←
        API.TorchLean.Module.initOptim runner.module opt
      let stRef ← IO.mkRef st0
      let runStep := fun (sample : API.TorchLean.TensorPack α [σ, τ]) => do
        trainMode runner
        let stepIdx ← stepRef.get
        let loss ← moduleLoss runner sample
        updateRunnerBuffers runner sample
        let lrα := API.Runtime.ofFloat (stepLR scheduler optimizer stepIdx)
        let st0 ← stRef.get
        let st := adamStateWithLR (paramShapes := paramShapes task) lrα st0
        let st' ← API.TorchLean.Module.stepWith runner.module opt st sample
        stRef.set st'
        stepRef.set (stepIdx + 1)
        pure loss
      pure {
        runner := runner
        stepSample := runStep
        epochSamples := fun samples => samples.mapM runStep
        stepCount := stepRef.get
      }
  | .adamw lr weightDecay beta1 beta2 epsilon =>
      let opt := API.TorchLean.Optim.adamw
        (α := α) (paramShapes := paramShapes task)
        (lr := API.Runtime.ofFloat lr) (weightDecay := API.Runtime.ofFloat weightDecay)
        (beta1 := API.Runtime.ofFloat beta1)
        (beta2 := API.Runtime.ofFloat beta2)
        (epsilon := API.Runtime.ofFloat epsilon)
      let st0 : API.TorchLean.Optim.StateList _root_.Optim.AdamW.State α (paramShapes task) ←
        API.TorchLean.Module.initOptim runner.module opt
      let stRef ← IO.mkRef st0
      let runStep := fun (sample : API.TorchLean.TensorPack α [σ, τ]) => do
        trainMode runner
        let stepIdx ← stepRef.get
        let loss ← moduleLoss runner sample
        updateRunnerBuffers runner sample
        let lrα := API.Runtime.ofFloat (stepLR scheduler optimizer stepIdx)
        let st0 ← stRef.get
        let st := adamwStateWithLR (paramShapes := paramShapes task) lrα st0
        let st' ← API.TorchLean.Module.stepWith runner.module opt st sample
        stRef.set st'
        stepRef.set (stepIdx + 1)
        pure loss
      pure {
        runner := runner
        stepSample := runStep
        epochSamples := fun samples => samples.mapM runStep
        stepCount := stepRef.get
      }

/-- Run one optimization step on a single supervised sample. -/
def step {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (loop : Stepper α task) (sample : API.TorchLean.TensorPack α [σ, τ]) : IO α :=
  loop.stepSample sample

/-- Run one epoch over a list of supervised samples, returning the per-step losses. -/
def epoch {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (loop : Stepper α task) (samples : List (API.TorchLean.TensorPack α [σ, τ])) : IO (List α) :=
  loop.epochSamples samples

end Supervised

namespace Trainer

/-!
`NN.API.TorchLean.Trainer` is the stable public namespace for supervised training machinery.

Usage note:

If you are writing ordinary model code, prefer the `NN` umbrella:
`import NN; open TorchLean; Trainer.new ...`. This namespace is the lower runtime
layer underneath that facade, and it exposes lower-level controls for custom training loops.

The intended workflow is:
- pick a `Task` (regression / classification),
- call `instantiate` to get a `Runner` (parameters + buffers + backend state),
- call `trainSamples` / `trainDataset` / `trainLoader`, or build a `Stepper` for custom loops.

This API is backend-agnostic: the same code can run in `.eager` mode or via a compiled backend,
depending on the `backend` argument passed to `instantiate`.
-/

@[inherit_doc Supervised.SeqTask]
abbrev Task := Supervised.SeqTask
@[inherit_doc Supervised.Runner]
abbrev Runner := Supervised.Runner
@[inherit_doc Supervised.OptimizerConfig]
abbrev Optimizer := Supervised.OptimizerConfig
@[inherit_doc Supervised.TrainConfig]
abbrev TrainConfig := Supervised.TrainConfig
@[inherit_doc Supervised.LoaderTrainConfig]
abbrev LoaderTrainConfig := Supervised.LoaderTrainConfig
@[inherit_doc Supervised.TrainReport]
abbrev TrainReport := Supervised.TrainReport
@[inherit_doc Supervised.Stepper]
abbrev Stepper := Supervised.Stepper

/-- Lower-runtime regression task with mean-squared error loss by default. -/
def runtimeRegressionTask {σ τ : Spec.Shape} (model : API.TorchLean.NN.Seq σ τ)
    (reduction : API.TorchLean.Loss.Reduction := .mean) :
    Task σ τ :=
  Supervised.SeqTask.mse model reduction

/-- Lower-runtime classifier task with cross-entropy loss by default. -/
def runtimeClassifierTask {σ τ : Spec.Shape} (model : API.TorchLean.NN.Seq σ τ)
    (reduction : API.TorchLean.Loss.Reduction := .mean) :
    Task σ τ :=
  Supervised.SeqTask.crossEntropyOneHot model reduction

/--
SGD optimizer config.

PyTorch analogue: `torch.optim.SGD`
  (`https://pytorch.org/docs/stable/generated/torch.optim.SGD.html`).
-/
def sgd (lr : Float) (momentum : Float := 0.0) : Optimizer := .sgd lr momentum

/-- Momentum SGD optimizer config. Pass `lr` and optionally override `momentum`. -/
def momentumSGD (lr : Float) (momentum : Float := 0.9) : Optimizer := .sgd lr momentum

/--
Adam optimizer config. Pass `lr`; override `beta1`, `beta2`, or `epsilon` when the run needs it.

PyTorch analogue: `torch.optim.Adam`
  (`https://pytorch.org/docs/stable/generated/torch.optim.Adam.html`).
-/
def adam (lr : Float) (beta1 : Float := 0.9) (beta2 : Float := 0.999) (epsilon : Float := 1e-8) :
    Optimizer :=
  .adam lr beta1 beta2 epsilon

/--
AdamW optimizer config. Pass `lr`; override weight decay or moment parameters when the run needs it.

PyTorch analogue: `torch.optim.AdamW`
  (`https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html`).
-/
def adamw (lr : Float) (weightDecay : Float := 0.01)
    (beta1 : Float := 0.9) (beta2 : Float := 0.999) (epsilon : Float := 1e-8) :
    Optimizer :=
  .adamw lr weightDecay beta1 beta2 epsilon

/-- Optimizer algorithm accepted by simple CLI commands that expose a `--optim` flag. -/
inductive OptimizerKind where
  /-- Stochastic gradient descent. -/
  | sgd
  /-- Adam with TorchLean's usual public defaults. -/
  | adam
  /-- AdamW with decoupled weight decay. -/
  | adamw
deriving DecidableEq, Repr

namespace OptimizerKind

/-- Parse an optimizer name accepted by a command-line `--optim` flag. -/
def parse (s : String) : Except String OptimizerKind :=
  if s == "sgd" then
    pure .sgd
  else if s == "adam" then
    pure .adam
  else if s == "adamw" then
    pure .adamw
  else
    throw s!"bad --optim {s}; expected sgd, adam, or adamw"

/-- Human-readable optimizer name used in logs. -/
def name : OptimizerKind → String
  | .sgd => "SGD"
  | .adam => "Adam"
  | .adamw => "AdamW"

/-- Build a public optimizer config for this optimizer kind and learning rate. -/
def toOptimizer (kind : OptimizerKind) (lr : Float) : Optimizer :=
  match kind with
  | .sgd => Trainer.sgd lr
  | .adam => Trainer.adam (lr := lr) (beta1 := 0.9) (beta2 := 0.95) (epsilon := 1e-8)
  | .adamw =>
      Trainer.adamw (lr := lr) (weightDecay := 0.1) (beta1 := 0.9) (beta2 := 0.95)
        (epsilon := 1e-8)

end OptimizerKind

/-- Fixed-step training config over an in-memory sample list or dataset. -/
def steps (count : Nat) (optimizer : Optimizer := sgd 0.01) (logEvery : Nat := 1) : TrainConfig :=
  { steps := count, optimizer := optimizer, logEvery := logEvery }

/-- Epoch-based training config over a data loader. -/
def epochs (count : Nat) (optimizer : Optimizer := sgd 0.01) (logEvery : Nat := 1) :
    LoaderTrainConfig :=
  { epochs := count, optimizer := optimizer, logEvery := logEvery }

/-- Attach a scheduler to a step-based training config. -/
def withScheduler (cfg : TrainConfig) (scheduler : API.TorchLean.Schedulers.Config) : TrainConfig :=
  { cfg with scheduler := some scheduler }

/-- Attach a scheduler to an epoch-based loader training config. -/
def withEpochScheduler (cfg : LoaderTrainConfig) (scheduler : API.TorchLean.Schedulers.Config) :
    LoaderTrainConfig :=
  { cfg with scheduler := some scheduler }

/-- Step-based constant learning-rate schedule. -/
def constantLR (cfg : TrainConfig) (lr : Float) : TrainConfig :=
  withScheduler cfg (.constant lr)

/-- Step-based step-decay schedule. -/
def stepLR (cfg : TrainConfig) (base : Float) (stepSize : Nat) (gamma : Float := 0.1) : TrainConfig :=
  withScheduler cfg (.step base stepSize gamma)

/-- Step-based exponential-decay schedule. -/
def exponentialLR (cfg : TrainConfig) (base : Float) (gamma : Float) : TrainConfig :=
  withScheduler cfg (.exponential base gamma)

/-- Epoch-based constant learning-rate schedule. -/
def constantEpochLR (cfg : LoaderTrainConfig) (lr : Float) : LoaderTrainConfig :=
  withEpochScheduler cfg (.constant lr)

/-- Epoch-based step-decay schedule. -/
def stepEpochLR (cfg : LoaderTrainConfig) (base : Float) (stepSize : Nat) (gamma : Float := 0.1) :
    LoaderTrainConfig :=
  withEpochScheduler cfg (.step base stepSize gamma)

/-- Epoch-based exponential-decay schedule. -/
def exponentialEpochLR (cfg : LoaderTrainConfig) (base : Float) (gamma : Float) : LoaderTrainConfig :=
  withEpochScheduler cfg (.exponential base gamma)

/--
Instantiate a runner under explicit Torch options (`backend`, `useGpu`, `fastKernels`, ...).

This is the recommended entrypoint when you want CUDA eager execution from the training helpers
without dropping down to `TorchLean.Module` directly.
-/
def instantiateWithOptions {σ τ : Spec.Shape} (task : Task σ τ)
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [API.Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (opts : API.TorchLean.Options := {}) :
    IO (Runner α task) :=
  Supervised.instantiateWithRuntimeOptions (task := task) (α := α) opts

/--
Instantiate a runner (parameters + buffers + backend state) for the given task.

This allocates and initializes model parameters (via `Seq.initParams`) and sets up the chosen
execution backend (`.eager` vs `.compiled`).
-/
def instantiate {σ τ : Spec.Shape} (task : Task σ τ)
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [API.Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (backend : API.TorchLean.Backend := .eager) :
    IO (Runner α task) :=
  instantiateWithOptions (task := task) (α := α) { backend := backend }

/--
CLI-oriented runner entry point.

This parses dtype/backend flags (via `NN.API.DType` / `Module.ExecConfig`) and then calls the
continuation `k` under the selected scalar backend.
-/
def run {σ τ : Spec.Shape} (task : Task σ τ) (args : List String)
    (k :
      ∀ {α : Type}, [API.Semantics.Scalar α] → [DecidableEq Spec.Shape] → [ToString α] →
        [API.Runtime.Scalar α] → Runner α task → List String → IO Unit) :
    IO Unit :=
  Supervised.run task args k

/-- Get the current model parameters from a runner. -/
def params {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) :
    IO (API.TorchLean.TensorPack α (Supervised.paramShapes task)) :=
  Supervised.params runner

/-- Read the current mode (train vs eval). -/
def mode {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) : IO API.TorchLean.NN.Mode :=
  Supervised.mode runner

/-- Set the mode (train vs eval). -/
def setMode {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (value : API.TorchLean.NN.Mode) : IO Unit :=
  Supervised.setMode runner value

/-- Put the runner in training mode so layers such as dropout/batchnorm use training behavior. -/
def trainMode {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) : IO Unit :=
  Supervised.trainMode runner

/-- Put the runner in evaluation mode so stateful layers use inference behavior. -/
def evalMode {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) : IO Unit :=
  Supervised.evalMode runner

/-- Check whether the runner is in training mode. -/
def isTraining {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) : IO Bool :=
  Supervised.isTraining runner

/-- Run forward+backward on one supervised sample and return gradients for all parameters. -/
def backward {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (sample : API.TorchLean.TensorPack α [σ, τ]) :
    IO (API.TorchLean.TensorPack α (Supervised.paramShapes task)) :=
  Supervised.backward runner sample

/--
Predict on a single input tensor.

This runs the forward pass under the runner's current mode.
-/
def predict {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (x : Spec.Tensor α σ) :
    IO (Spec.Tensor α τ) :=
  Supervised.predict runner x

/-- Predict on a list of inputs (runs the forward pass repeatedly). -/
def predictBatch {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (xs : List (Spec.Tensor α σ)) :
    IO (List (Spec.Tensor α τ)) :=
  Supervised.predictBatch runner xs

/--
Predict the argmax class for a classification task, if `argmax` is well-defined for `α`.

It runs `predict` and then applies `Metrics.argmax?` to the output vector.
-/
def predictClass? {σ : Spec.Shape} {n : Nat} {task : Task σ (.dim n .scalar)}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (x : Spec.Tensor α σ) :
    IO (Option (Fin n)) :=
  Supervised.predictClass? (task := task) runner x

/-- Count correct predictions in a one-hot labeled sample list (returns `(correct, total)`). -/
def accuracyOneHot {σ : Spec.Shape} {n : Nat} {task : Task σ (.dim n .scalar)}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (samples : List (API.TorchLean.TensorPack α [σ, .dim n .scalar])) :
    IO (Nat × Nat) :=
  Supervised.accuracyOneHot (task := task) runner samples

/-- Mean loss over an explicit list of samples. -/
def meanLoss {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    (runner : Runner α task) (samples : List (API.TorchLean.TensorPack α [σ, τ])) :
    IO α :=
  Supervised.meanLoss runner samples

/-- Mean loss over an entire `Dataset`. -/
def meanLossDataset {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    (runner : Runner α task)
    (dataset : _root_.Runtime.Autograd.Train.Dataset (API.TorchLean.TensorPack α [σ, τ])) :
    IO α :=
  Supervised.meanLossDataset runner dataset

/--
Train on an explicit list of samples for a fixed number of steps.

Returns a small report with mean loss before/after.
-/
def trainSamples {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α] [API.Runtime.Scalar α]
    (runner : Runner α task) (cfg : TrainConfig)
    (samples : List (API.TorchLean.TensorPack α [σ, τ])) :
    IO (TrainReport α) :=
  Supervised.trainSamples runner cfg samples

/-- Train on a `Dataset` for a fixed number of steps. -/
def trainDataset {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α] [API.Runtime.Scalar α]
    (runner : Runner α task) (cfg : TrainConfig)
    (dataset : _root_.Runtime.Autograd.Train.Dataset (API.TorchLean.TensorPack α [σ, τ])) :
    IO (TrainReport α) :=
  Supervised.trainDataset runner cfg dataset

/-- Train using a `DataLoader` for a fixed number of epochs. -/
def trainLoader {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α] [API.Runtime.Scalar α]
    (runner : Runner α task) (cfg : LoaderTrainConfig)
    (loader : _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ])) :
    IO (TrainReport α ×
      _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ])) :=
  Supervised.trainLoader runner cfg loader

/--
Construct a stateful stepper for custom loops.

This is useful if you want to control:
- evaluation cadence,
- logging,
- validation, early stopping, etc.

The returned `Stepper` still uses TorchLean's optimizer/scheduler implementations.
-/
def stepper {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α] [API.Runtime.Scalar α]
    (runner : Runner α task) (optimizer : Optimizer)
    (scheduler : Option API.TorchLean.Schedulers.Config := none) :
    IO (Stepper α task) :=
  Supervised.stepper runner optimizer scheduler

/-- Run a single training step on one sample using a `Stepper`. -/
def step {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (loop : Stepper α task) (sample : API.TorchLean.TensorPack α [σ, τ]) : IO α :=
  Supervised.step loop sample

/-- Run an epoch over a list of samples using a `Stepper` (returns the per-step losses). -/
def epoch {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (loop : Stepper α task) (samples : List (API.TorchLean.TensorPack α [σ, τ])) : IO (List α) :=
  Supervised.epoch loop samples

/--
Convenience: instantiate + train on a list of samples.

Returns both the `Runner` (so you can keep using the trained parameters) and the train report.
-/
def train {σ τ : Spec.Shape} {_task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α] [API.Runtime.Scalar α]
    (task : Task σ τ) (cfg : TrainConfig)
    (samples : List (API.TorchLean.TensorPack α [σ, τ]))
    (backend : API.TorchLean.Backend := .eager) :
    IO (Runner α task × TrainReport α) := do
  let runner ← instantiate (task := task) (α := α) backend
  let report ← trainSamples runner cfg samples
  pure (runner, report)

/--
Convenience: instantiate a task and train on a `Dataset`.

Returns both the `Runner` and the train report.
-/
def trainDatasetTask {σ τ : Spec.Shape} {_task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α] [API.Runtime.Scalar α]
    (task : Task σ τ) (cfg : TrainConfig)
    (dataset : _root_.Runtime.Autograd.Train.Dataset (API.TorchLean.TensorPack α [σ, τ]))
    (backend : API.TorchLean.Backend := .eager) :
    IO (Runner α task × TrainReport α) := do
  let runner ← instantiate (task := task) (α := α) backend
  let report ← trainDataset runner cfg dataset
  pure (runner, report)

/--
Convenience: instantiate a task and train using a `DataLoader`.

Returns the `Runner`, the train report, and the updated loader state (shuffled epoch cursor).
-/
def trainLoaderTask {σ τ : Spec.Shape} {_task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α] [API.Runtime.Scalar α]
    (task : Task σ τ) (cfg : LoaderTrainConfig)
    (loader : _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ]))
    (backend : API.TorchLean.Backend := .eager) :
    IO (Runner α task × TrainReport α ×
      _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ])) := do
  let runner ← instantiate (task := task) (α := α) backend
  let (report, loader') ← trainLoader runner cfg loader
  pure (runner, report, loader')

end Trainer

/-
The remaining exports expose the imperative session interface.

Most code should start from `Trainer.new` and `trainer.train`. Use these exports when a file needs:
- interactive/debug workflows that want mutable tape control, or
- advanced tooling that needs the low-level session primitives.
-/

/-- Execution config parsed from CLI flags (dtype/backend/fast-kernels). -/
abbrev ExecConfig := Module.ExecConfig

namespace ExecConfig
/-- Parse and strip execution flags, returning `(config, remainingArgs)`. -/
abbrev parseAndStrip := Module.ExecConfig.parseAndStrip
/-- Log the chosen execution config to stdout for reproducible runs. -/
abbrev log := Module.ExecConfig.log
end ExecConfig

namespace ScalarTrainer
/-!
Re-export of the low-level imperative scalar trainer interface.

This exposes `forwardT`/`backwardT`/`stepT` from `Runtime.Autograd.TorchLean.ScalarTrainer`.
Use the higher-level `TorchLean.Trainer` facade unless a file needs these lower-level training hooks.
-/
export _root_.Runtime.Autograd.TorchLean.ScalarTrainer (forwardT backwardT stepT)
end ScalarTrainer

namespace Session
/-!
Imperative session API: a tape-backed interface that can run in eager or compiled mode.

This is approximately analogous to using PyTorch "eager tensors", except TorchLean makes the tape/session
explicit. The `Session` surface is useful for:
- interactive experiments in `IO`,
- debugging (inspect intermediate values),
- building higher-level runners.
-/
export _root_.Runtime.Autograd.TorchLean.Session
  (new resetTape param use input inputNat getNat setNat inputNatVec getNatVec setNatVec const
    getValue
   withFreshTape sgdStepScalarGraph
   add sub mul scale abs sqrt clamp max min
   broadcastTo reshape transpose2d transpose3dFirstToLast transpose3dLastToFirst
     transpose3dLastTwo
   swapAdjacentAtDepth
   reduceSum reduceMean
   gatherScalar gatherRow gatherScalarRef gatherRowRef gatherVecRef gatherRowsRef
   gatherScalarNat gatherVecNat gatherRowsNat scatterAddVec scatterAddRow
   matmul bmm concatVectors concatDim0 sliceRange0 maxPool2d smoothMaxPool2d avgPool2d
   relu sigmoid tanh softmax softplus exp log safeLog sum flatten
   linear mseLoss layerNorm conv2d multiHeadAttention
   backwardDenseAll backwardScalarDenseAll grad sgdStepAll)
end Session

end TorchLean
end API
end NN
