/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.NN

/-!
# TorchLean Optimizers

Public optimizer configuration records and runtime optimizer constructors.
-/

@[expose] public section

namespace TorchLean

namespace optim

@[inherit_doc NN.API.TorchLean.Trainer.Optimizer]
abbrev Optimizer := NN.API.TorchLean.Trainer.Optimizer

/-- Public SGD optimizer configuration. -/
structure SgdConfig where
  /-- Learning rate. -/
  lr : Float
  /-- Momentum coefficient. -/
  momentum : Float := 0.0
deriving Repr

/-- Public Adam optimizer configuration. -/
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

/-- Public AdamW optimizer configuration. -/
structure AdamWConfig extends AdamConfig where
  /-- Decoupled weight decay. -/
  weightDecay : Float := 0.01
deriving Repr

/-- SGD optimizer config, written `optim.sgd { lr := 0.05 }`. -/
def sgd (cfg : SgdConfig) : Optimizer :=
  NN.API.TorchLean.Trainer.sgd cfg.lr cfg.momentum

@[inherit_doc NN.API.TorchLean.Trainer.momentumSGD]
def momentumSGD (cfg : SgdConfig) : Optimizer :=
  NN.API.TorchLean.Trainer.sgd cfg.lr
    (if cfg.momentum == 0.0 then 0.9 else cfg.momentum)

/-- Adam optimizer config, written `optim.adam { lr := 1e-3 }`. -/
def adam (cfg : AdamConfig) : Optimizer :=
  NN.API.TorchLean.Trainer.adam cfg.lr cfg.beta1 cfg.beta2 cfg.epsilon

/-- AdamW optimizer config, written `optim.adamw { lr := 1e-3, weightDecay := 0.01 }`. -/
def adamw (cfg : AdamWConfig) : Optimizer :=
  NN.API.TorchLean.Trainer.adamw cfg.lr cfg.weightDecay cfg.beta1 cfg.beta2 cfg.epsilon

@[inherit_doc NN.API.TorchLean.Trainer.OptimizerKind]
abbrev Kind := NN.API.TorchLean.Trainer.OptimizerKind

namespace Kind

@[inherit_doc NN.API.TorchLean.Trainer.OptimizerKind.parse]
abbrev parse := NN.API.TorchLean.Trainer.OptimizerKind.parse

@[inherit_doc NN.API.TorchLean.Trainer.OptimizerKind.name]
def name (kind : Kind) : String :=
  NN.API.TorchLean.Trainer.OptimizerKind.name kind

@[inherit_doc NN.API.TorchLean.Trainer.OptimizerKind.toOptimizer]
def toOptimizer (kind : Kind) (lr : Float) : Optimizer :=
  NN.API.TorchLean.Trainer.OptimizerKind.toOptimizer kind lr

end Kind

/-- Runtime Adam optimizer for module-level training. -/
def runtimeAdam {α : Type} [Runtime.TensorScalar α]
    (lr beta1 beta2 epsilon : α) {paramShapes : List Shape} :
    NN.API.TorchLean.Optim.Optimizer α paramShapes :=
  NN.API.TorchLean.Optim.adam (α := α) lr beta1 beta2 epsilon (paramShapes := paramShapes)

/-- Runtime AdamW optimizer for module-level training. -/
def runtimeAdamW {α : Type} [Runtime.TensorScalar α]
    (lr weightDecay beta1 beta2 epsilon : α) {paramShapes : List Shape} :
    NN.API.TorchLean.Optim.Optimizer α paramShapes :=
  NN.API.TorchLean.Optim.adamw (α := α) lr weightDecay beta1 beta2 epsilon
    (paramShapes := paramShapes)

/-- Runtime SGD optimizer for module-level training. -/
def runtimeSGD {α : Type} [Runtime.TensorScalar α]
    (lr : α) {paramShapes : List Shape} :
    NN.API.TorchLean.Optim.Optimizer α paramShapes :=
  NN.API.TorchLean.Optim.sgd (α := α) lr (paramShapes := paramShapes)

/-- Runtime momentum-SGD optimizer for module-level training. -/
def runtimeMomentumSGD {α : Type} [Runtime.TensorScalar α]
    (lr momentum : α) {paramShapes : List Shape} :
    NN.API.TorchLean.Optim.Optimizer α paramShapes :=
  NN.API.TorchLean.Optim.momentumSGD (α := α) lr momentum (paramShapes := paramShapes)

export NN.API.TorchLean.Optim
  (handle)

end optim


end TorchLean
