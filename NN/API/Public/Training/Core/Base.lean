/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Adapters
public import NN.API.Common
public import NN.API.Data
public import NN.API.Macros
public import NN.API.Public.TensorPack
public import NN.API.Rand
public import NN.API.RL
public import NN.API.Runtime
public import NN.API.Samples
public import NN.API.SelfSupervised
public import NN.API.Text
public import NN.API.Text.Bpe
public import NN.Spec.Layers.PositionalEncoding

import Mathlib.Algebra.Order.Algebra

@[expose] public section

namespace NN
namespace API

/-!
# Public optimizers, losses, metrics, and training tools

This module contains the executable training API: optimizer configs, loss exports, metrics,
callbacks, loaders, and module-level training loops.
-/

namespace optim

/-!
Optimizer configs for the public training APIs.

These mirror common PyTorch optimizers (by name and default hyperparameters), but they produce a
TorchLean trainer config rather than a mutable optimizer object.

PyTorch references:
- `torch.optim`: `https://pytorch.org/docs/stable/optim.html`
-/

@[inherit_doc TorchLean.Trainer.Optimizer]
abbrev Optimizer := TorchLean.Trainer.Optimizer

/-- SGD optimizer configuration. -/
structure SgdConfig where
  /-- Learning rate. -/
  lr : Float
  /-- Momentum coefficient. -/
  momentum : Float := 0.0
deriving Repr

/-- AdaGrad optimizer configuration. -/
structure AdaGradConfig where
  /-- Learning rate. -/
  lr : Float
  /-- Numerical stabilizer. -/
  epsilon : Float := 1e-10
deriving Repr

/-- RMSProp optimizer configuration. -/
structure RMSPropConfig where
  /-- Learning rate. -/
  lr : Float
  /-- Decay coefficient for the running average of squared gradients. -/
  decay : Float := 0.99
  /-- Numerical stabilizer. -/
  epsilon : Float := 1e-8
deriving Repr

/-- Adam optimizer configuration. -/
structure AdamConfig where
  /-- Learning rate. -/
  lr : Float
  /-- First moment coefficient. -/
  beta1 : Float := 0.9
  /-- Second moment coefficient. -/
  beta2 : Float := 0.999
  /-- Numerical stabilizer. -/
  epsilon : Float := 1e-8
deriving Repr

/-- AdamW optimizer configuration. -/
structure AdamWConfig extends AdamConfig where
  /-- Decoupled weight decay. -/
  weightDecay : Float := 0.01
deriving Repr

/-- Adadelta optimizer configuration. -/
structure AdadeltaConfig where
  /-- Learning rate. -/
  lr : Float := 1.0
  /-- Decay coefficient for gradient/update accumulators. -/
  rho : Float := 0.9
  /-- Numerical stabilizer. -/
  epsilon : Float := 1e-6
deriving Repr

/-- SGD optimizer config, written `optim.sgd { lr := 0.05 }`. -/
def sgd (cfg : SgdConfig) : Optimizer :=
  TorchLean.Trainer.sgd cfg.lr cfg.momentum

@[inherit_doc TorchLean.Trainer.momentumSGD]
def momentumSGD (cfg : SgdConfig) : Optimizer :=
  TorchLean.Trainer.sgd cfg.lr
    (if cfg.momentum == 0.0 then 0.9 else cfg.momentum)

/-- AdaGrad optimizer config, written `optim.adagrad { lr := 0.05 }`. -/
def adagrad (cfg : AdaGradConfig) : Optimizer :=
  TorchLean.Trainer.adagrad cfg.lr cfg.epsilon

/-- RMSProp optimizer config, written `optim.rmsprop { lr := 1e-3 }`. -/
def rmsprop (cfg : RMSPropConfig) : Optimizer :=
  TorchLean.Trainer.rmsprop cfg.lr cfg.decay cfg.epsilon

/-- Adam optimizer config, written `optim.adam { lr := 1e-3 }`. -/
def adam (cfg : AdamConfig) : Optimizer :=
  TorchLean.Trainer.adam cfg.lr cfg.beta1 cfg.beta2 cfg.epsilon

/-- AdamW optimizer config, written `optim.adamw { lr := 1e-3, weightDecay := 0.01 }`. -/
def adamw (cfg : AdamWConfig) : Optimizer :=
  TorchLean.Trainer.adamw cfg.lr cfg.weightDecay cfg.beta1 cfg.beta2 cfg.epsilon

/-- Adadelta optimizer config, written `optim.adadelta {}`. -/
def adadelta (cfg : AdadeltaConfig) : Optimizer :=
  TorchLean.Trainer.adadelta cfg.lr cfg.rho cfg.epsilon

@[inherit_doc TorchLean.Trainer.OptimizerKind]
abbrev Kind := TorchLean.Trainer.OptimizerKind

namespace Kind

@[inherit_doc TorchLean.Trainer.OptimizerKind.parse]
abbrev parse := TorchLean.Trainer.OptimizerKind.parse

@[inherit_doc TorchLean.Trainer.OptimizerKind.name]
def name (kind : Kind) : String :=
  TorchLean.Trainer.OptimizerKind.name kind

@[inherit_doc TorchLean.Trainer.OptimizerKind.toOptimizer]
def toOptimizer (kind : Kind) (lr : Float) : Optimizer :=
  TorchLean.Trainer.OptimizerKind.toOptimizer kind lr

end Kind

end optim

namespace loss

/-
Loss functions are re-exported from the TorchLean runtime.

PyTorch references:
- `torch.nn.functional` loss docs: `https://pytorch.org/docs/stable/nn.functional.html`
-/

@[inherit_doc TorchLean.Loss.Reduction]
abbrev Reduction := TorchLean.Loss.Reduction

export TorchLean.Loss
  (mse
   nllOneHot crossEntropyOneHot
   nllIndex nllNat crossEntropyIndex crossEntropyNat
   rowTargetFlatIndices nllRowsNat crossEntropyRowsNat
   bceWithLogits bce)

end loss

namespace metrics

/- Small classification metrics: argmax, one-hot decoding, and correctness checks. -/
export TorchLean.Metrics (argmax? classOfOneHot? correctOneHot?)

end metrics

namespace train

/-!
Public training tools.

This namespace is the public training API: it wires together
- a model (`nn.Sequential`)
- a loss (regression or classification)
- an optimizer config (`API.optim`)
- optional LR schedules

The API exposes a small set of reusable building blocks, so model commands can share the same
training path while still making the model, loss, optimizer, and logging choices explicit.

Importantly, this module sits around the root public trainer facade:
- use `TorchLean.Trainer.RunConfig` for persistent runtime settings,
- use `TorchLean.Trainer.TrainOptions` for one training call,
- use `Trainer.new ...` followed by `trainer.train ...` as the normal quickstart path,
- use `API.train.Manual` only when the dependent runner API is genuinely needed.

### PyTorch Mapping

These definitions correspond to the training loop code you would typically write around:
- `torch.optim.*`
- forward pass + loss
- `loss.backward()` + optimizer step
- batching via `torch.utils.data.DataLoader`
-/

/-!
Manual training layer underneath the `Trainer` facade.

New examples should prefer `Trainer.new`, `trainer.train`, and trained-handle methods. This namespace
remains available for runtime code that really does need direct steppers, epochs, or manual
reporting, but it is no longer the API we teach first.
-/

export TorchLean.Trainer
  (steps epochs
   constantLR stepLR exponentialLR
   constantEpochLR stepEpochLR exponentialEpochLR)

/-!
## Metric Artifacts

The public training API also exposes TorchLean's metric artifact format.  This is
the local equivalent of “log scalars during a run, then inspect them later”: write a JSON
`TrainLog`, view it with the training widgets, or adapt the JSON to an external tracker such as
Weights & Biases.
-/

export _root_.Runtime.Training
  (Series TrainLog Curve MetricHistory ConfigEntry Artifact RunInfo ExperimentLog LogDestination)
export _root_.Runtime.Training.Curve (push toTrainLog)
export _root_.Runtime.Training.MetricHistory (empty push toTrainLog)
export _root_.Runtime.Training.ExperimentLog (init log logRow addArtifact toTrainLog)
export _root_.Runtime.Training.LogDestination (disabled json isEnabled path? writeTrainLog)
export _root_.Runtime.Training.TrainLog (writeJson readJson)

end train
end API
end NN
