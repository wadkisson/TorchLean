/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.NN
public import NN.API.Public.TensorPack
public import NN.API.Public.Seeded
public import NN.API.Public.Autograd
public import NN.API.Data
public import NN.API.Data.Transforms
public import NN.API.Runtime
public import NN.API.Models
public import NN.API.Public.NN.Transformer
public import NN.API.RL
public import NN.API.Rand
public import NN.API.Samples.Bands
public import NN.API.Text.Bpe
public import NN.MLTheory.CROWN.Flatbox
public import NN.MLTheory.CROWN.Graph
public import NN.Verification.TorchLean.Compile

/-!
# TorchLean Public Names

Short root names and training-log types used by `import NN`.
-/

@[expose] public section

namespace TorchLean

@[inherit_doc Spec.Shape]
abbrev Shape := Spec.Shape

@[inherit_doc TorchLean.Sample.Supervised]
abbrev SupervisedSample := TorchLean.Sample.Supervised

@[inherit_doc NN.API.TorchLean.Options]
abbrev Options := NN.API.TorchLean.Options

@[inherit_doc NN.API.TorchLean.TensorPack]
abbrev TensorPack := NN.API.TorchLean.TensorPack

namespace Training

@[inherit_doc _root_.Runtime.Training.Curve]
abbrev Curve := _root_.Runtime.Training.Curve

@[inherit_doc _root_.Runtime.Training.TrainLog]
abbrev TrainLog := _root_.Runtime.Training.TrainLog

@[inherit_doc _root_.Runtime.Training.ExperimentLog]
abbrev ExperimentLog := _root_.Runtime.Training.ExperimentLog

@[inherit_doc _root_.Runtime.Training.LogDestination]
abbrev LogDestination := _root_.Runtime.Training.LogDestination

@[inherit_doc _root_.Runtime.Training.MetricHistory]
abbrev MetricHistory := _root_.Runtime.Training.MetricHistory

@[inherit_doc _root_.Runtime.Autograd.Train.Dataset]
abbrev Dataset := _root_.Runtime.Autograd.Train.Dataset

@[inherit_doc _root_.Runtime.Autograd.Train.DataLoader]
abbrev DataLoader := _root_.Runtime.Autograd.Train.DataLoader

namespace MetricHistory

export _root_.Runtime.Training.MetricHistory (empty)

end MetricHistory

end Training

end TorchLean
