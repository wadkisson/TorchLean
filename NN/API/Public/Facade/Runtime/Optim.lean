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

The default trainer config exposes self-contained core update rules for SGD, momentum SGD,
AdaGrad, RMSProp, Adam, AdamW, and Adadelta. Runtime-only extension points live here too:

- Muon is an optimizer, but the public runtime constructor requires an explicit orthogonalization
  backend.  The identity backend is available for proofs and fallback behavior.
- GaLore is exposed as gradient-projection machinery around a base update.  The public name is
  therefore `optim.galore.projectedSGD`, which says exactly which update rule owns the state.
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

/-- Public AdaGrad optimizer configuration. -/
structure AdaGradConfig where
  /-- Learning rate. -/
  lr : Float
  /-- Numerical stabilizer. -/
  epsilon : Float := 1e-10
deriving Repr

/-- Public RMSProp optimizer configuration. -/
structure RMSPropConfig where
  /-- Learning rate. -/
  lr : Float
  /-- Decay coefficient for the running average of squared gradients. -/
  decay : Float := 0.99
  /-- Numerical stabilizer. -/
  epsilon : Float := 1e-8
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

/-- Public Adadelta optimizer configuration. -/
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
  NN.API.TorchLean.Trainer.sgd cfg.lr cfg.momentum

@[inherit_doc NN.API.TorchLean.Trainer.momentumSGD]
def momentumSGD (cfg : SgdConfig) : Optimizer :=
  NN.API.TorchLean.Trainer.sgd cfg.lr
    (if cfg.momentum == 0.0 then 0.9 else cfg.momentum)

/-- AdaGrad optimizer config, written `optim.adagrad { lr := 0.05 }`. -/
def adagrad (cfg : AdaGradConfig) : Optimizer :=
  NN.API.TorchLean.Trainer.adagrad cfg.lr cfg.epsilon

/-- RMSProp optimizer config, written `optim.rmsprop { lr := 1e-3 }`. -/
def rmsprop (cfg : RMSPropConfig) : Optimizer :=
  NN.API.TorchLean.Trainer.rmsprop cfg.lr cfg.decay cfg.epsilon

/-- Adam optimizer config, written `optim.adam { lr := 1e-3 }`. -/
def adam (cfg : AdamConfig) : Optimizer :=
  NN.API.TorchLean.Trainer.adam cfg.lr cfg.beta1 cfg.beta2 cfg.epsilon

/-- AdamW optimizer config, written `optim.adamw { lr := 1e-3, weightDecay := 0.01 }`. -/
def adamw (cfg : AdamWConfig) : Optimizer :=
  NN.API.TorchLean.Trainer.adamw cfg.lr cfg.weightDecay cfg.beta1 cfg.beta2 cfg.epsilon

/-- Adadelta optimizer config, written `optim.adadelta {}`. -/
def adadelta (cfg : AdadeltaConfig) : Optimizer :=
  NN.API.TorchLean.Trainer.adadelta cfg.lr cfg.rho cfg.epsilon

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

/-- Runtime AdaGrad optimizer for module-level training. -/
def runtimeAdaGrad {α : Type} [Runtime.TensorScalar α]
    (lr epsilon : α) {paramShapes : List Shape} :
    NN.API.TorchLean.Optim.Optimizer α paramShapes :=
  NN.API.TorchLean.Optim.adagrad (α := α) lr epsilon (paramShapes := paramShapes)

/-- Runtime RMSProp optimizer for module-level training. -/
def runtimeRMSProp {α : Type} [Runtime.TensorScalar α]
    (lr decay epsilon : α) {paramShapes : List Shape} :
    NN.API.TorchLean.Optim.Optimizer α paramShapes :=
  NN.API.TorchLean.Optim.rmsprop (α := α) lr decay epsilon (paramShapes := paramShapes)

/-- Runtime Adadelta optimizer for module-level training. -/
def runtimeAdadelta {α : Type} [Runtime.TensorScalar α]
    (lr rho epsilon : α) {paramShapes : List Shape} :
    NN.API.TorchLean.Optim.Optimizer α paramShapes :=
  NN.API.TorchLean.Optim.adadelta (α := α) lr rho epsilon (paramShapes := paramShapes)

@[inherit_doc _root_.Optim.Muon.Orthogonalizer]
abbrev MuonOrthogonalizer := _root_.Optim.Muon.Orthogonalizer

@[inherit_doc _root_.Optim.Muon.identityOrthogonalizer]
def identityMuonOrthogonalizer {α : Type} {s : Shape} :
    MuonOrthogonalizer α s :=
  _root_.Optim.Muon.identityOrthogonalizer (α := α) (s := s)

/--
Runtime Muon-style optimizer for module-level training.

Muon is public at the runtime layer because a meaningful Muon run needs an orthogonalization
backend. The default identity backend supports proofs and fallback behavior; production Muon should
pass a matrix-shaped orthogonalizer.
-/
def runtimeMuon {α : Type} [Runtime.TensorScalar α]
    (lr momentum : α)
    (orthogonalizer : {s : Shape} → MuonOrthogonalizer α s :=
      fun {s} => identityMuonOrthogonalizer (α := α) (s := s))
    {paramShapes : List Shape} :
    NN.API.TorchLean.Optim.Optimizer α paramShapes :=
  NN.API.TorchLean.Optim.muon (α := α) lr momentum orthogonalizer (paramShapes := paramShapes)

namespace galore

@[inherit_doc _root_.Optim.GaLore.Projector]
abbrev Projector := _root_.Optim.GaLore.Projector

@[inherit_doc _root_.Optim.GaLore.identityProjector]
def identityProjector {α : Type} {s : Shape} :
    Projector α s s :=
  _root_.Optim.GaLore.identityProjector (α := α) (s := s)

/--
Projected-SGD runtime constructor for GaLore-style gradient projection.

This is a projection strategy wrapped around an SGD update.  Full GaLore also needs a policy that
constructs and refreshes low-rank projectors for matrix parameters; this constructor exposes the
verified update boundary once a same-shape projector is supplied.
-/
def projectedSGD {α : Type} [Runtime.TensorScalar α]
    (lr : α)
    (projector : {s : Shape} → Projector α s s :=
      fun {s} => identityProjector (α := α) (s := s))
    {paramShapes : List Shape} :
    NN.API.TorchLean.Optim.Optimizer α paramShapes :=
  NN.API.TorchLean.Optim.projectedSGD (α := α) lr projector (paramShapes := paramShapes)

end galore

export NN.API.TorchLean.Optim
  (handle)

end optim


end TorchLean
