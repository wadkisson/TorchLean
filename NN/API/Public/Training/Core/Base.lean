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
# Manual training foundations

This module exposes the schedules and metric artifacts used by direct runner loops. Public
optimizer configurations live in `TorchLean.optim`, while the ordinary model-training interface
lives in `TorchLean.Trainer`.
-/

namespace train

/-!
Low-level training support beneath `TorchLean.Trainer`.

The exported schedules and artifact types are useful to callback loops, custom streams, and
verification workflows. Model code should normally use `Trainer.new`, `trainer.train`, and the
returned trained handle.
-/

export TorchLean.Trainer
  (steps epochs
   constantLR stepLR exponentialLR
   constantEpochLR stepEpochLR exponentialEpochLR)

/-!
## Metric Artifacts

`TrainLog`, `Curve`, and `ExperimentLog` retain the data produced by a run without coupling the
runtime loop to a particular dashboard.
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
