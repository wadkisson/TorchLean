/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.NN
public import NN.API.Public.Facade.Runtime.RL

/-!
# TorchLean Module Runtime Facade

Executable module operations for advanced runtime and example code.
-/

@[expose] public section

namespace TorchLean

namespace Module

/-- Executable module instance with mutable runtime parameters and optimizer state. -/
abbrev ScalarModule := NN.API.TorchLean.Module.ScalarModule

/-- The module-definition type used by `Module` runtime operations. -/
abbrev ScalarModuleDef := NN.API.TorchLean.Module.ScalarModuleDef

export NN.API.TorchLean.Module
  (instantiateWithOptions forward backward step initOptim stepWith
   params setParams trainSGD trainWith meanLoss run)

/--
Instantiate an executable runtime module from a public `ScalarModuleDef`.

This is the generic public entrypoint for custom runtime tasks outside the standard supervised
module constructors such as `Module.instantiateMse` or `Module.instantiateCrossEntropyOneHot`.
-/
def instantiate
    {α : Type} [Runtime.SemanticScalar α] [DecidableEq Shape] [Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    {paramShapes inputShapes : List Shape}
    (opts : Options)
    (defn : ScalarModuleDef paramShapes inputShapes)
    (cast : Float → α := Runtime.ofFloat) :
    IO (ScalarModule α paramShapes inputShapes) :=
  instantiateWithOptions (α := α) defn cast opts

/--
Run one inference step through a supervised runtime module.

This is the public sibling of the direct runtime pattern
`nn.eval1 opts model m.trainer.params x`.
-/
def predict1 {σ τ : Shape} {α : Type}
    [Runtime.TensorScalar α] [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (opts : Options)
    (model : nn.Sequential σ τ)
    (m : ScalarModule α (nn.paramShapes model) [σ, τ])
    (x : Tensor.T α σ) : IO (Tensor.T α τ) :=
  nn.eval1 (α := α) opts model m.trainer.params x

/--
Run one no-grad inference step through a supervised runtime module.

Use this for sampling/reporting paths that should not build an autograd tape.
-/
def predict1NoGrad {σ τ : Shape} {α : Type}
    [Runtime.TensorScalar α] [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (opts : Options)
    (model : nn.Sequential σ τ)
    (m : ScalarModule α (nn.paramShapes model) [σ, τ])
    (x : Tensor.T α σ) : IO (Tensor.T α τ) :=
  nn.eval1NoGrad (α := α) opts model m.trainer.params x

/--
Instantiate a supervised MSE module directly from a sequential model.
-/
def instantiateMse {σ τ : Shape} {α : Type}
    [Runtime.SemanticScalar α] [DecidableEq Shape] [Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (opts : Options)
    (model : nn.Sequential σ τ)
    (reduction : LossReduction := .mean)
    (cast : Float → α := Runtime.ofFloat) :
    IO (ScalarModule α (nn.paramShapes model) [σ, τ]) :=
  instantiate (α := α) opts
    (nn.mseScalarModuleDef model (reduction := reduction)) cast

/--
Instantiate a supervised one-hot cross-entropy module directly from a sequential model.
-/
def instantiateCrossEntropyOneHot {σ τ : Shape} {α : Type}
    [Runtime.SemanticScalar α] [DecidableEq Shape] [Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (opts : Options)
    (model : nn.Sequential σ τ)
    (reduction : LossReduction := .mean)
    (cast : Float → α := Runtime.ofFloat) :
    IO (ScalarModule α (nn.paramShapes model) [σ, τ]) :=
  instantiate (α := α) opts
    (nn.crossEntropyOneHotScalarModuleDef model (reduction := reduction))
    cast

/--
Instantiate a custom supervised runtime module directly from a sequential model.

Use this when a public example keeps the ordinary `nn.Sequential` model surface but needs a custom
loss/module definition instead of the standard MSE or cross-entropy module constructors.
-/
def instantiateModuleDefModel
    {σ τ : Shape} {α : Type}
    [Runtime.SemanticScalar α] [DecidableEq Shape] [Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (opts : Options)
    (model : nn.Sequential σ τ)
    (moduleDefOf : (model : nn.Sequential σ τ) →
      ScalarModuleDef (nn.paramShapes model) [σ, τ])
    (cast : Float → α := Runtime.ofFloat) :
    IO (ScalarModule α (nn.paramShapes model) [σ, τ]) :=
  instantiate (α := α) opts (moduleDefOf model) cast

/--
Instantiate the standard PPO actor-critic supervised runtime module from rollout-shaped actor and
critic networks.
-/
def instantiatePpoActorCritic
    {stateShape : Shape} {batch nActions : Nat} {α : Type}
    [Fact (0 < batch)] [Fact (0 < nActions)]
    [Runtime.SemanticScalar α] [DecidableEq Shape] [Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (opts : Options)
    (actor : nn.Sequential stateShape (.dim batch (.dim nActions .scalar)))
    (critic : nn.Sequential stateShape (.dim batch (.dim 1 .scalar)))
    (cast : Float → α := Runtime.ofFloat) :
    IO (ScalarModule α
      (nn.paramShapes actor ++ nn.paramShapes critic)
      [stateShape, (.dim batch (.dim nActions .scalar)), (.dim batch .scalar), (.dim batch .scalar),
        (.dim batch (.dim 1 .scalar))]) :=
  instantiate (α := α) opts
    (rl.policy.autograd.ppoActorCriticScalarModuleDef
      (batch := batch) (nActions := nActions) actor critic)
    cast

/--
Build a sequential model, instantiate a one-hot cross-entropy runtime module for it, and continue
with both values.

This packages the common public example pattern
`nn.withModel mkModel fun model => let m ← Module.instantiateCrossEntropyOneHot ...`.
-/
def withCrossEntropyOneHotModel
    {σ τ : Shape} {α : Type} {β : Type}
    [Runtime.SemanticScalar α] [DecidableEq Shape] [Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (mkModel : nn.M (nn.Sequential σ τ))
    (opts : Options)
    (reduction : LossReduction := .mean)
    (cast : Float → α := Runtime.ofFloat)
    (k : (model : nn.Sequential σ τ) →
      ScalarModule α (nn.paramShapes model) [σ, τ] → IO β) : IO β :=
  nn.withModel mkModel fun model => do
    let m ← instantiateCrossEntropyOneHot
      (α := α) opts model (reduction := reduction) (cast := cast)
    k model m

/--
Build a sequential model, instantiate an MSE runtime module for it, and continue with both values.
-/
def withMseModel
    {σ τ : Shape} {α : Type} {β : Type}
    [Runtime.SemanticScalar α] [DecidableEq Shape] [Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (mkModel : nn.M (nn.Sequential σ τ))
    (opts : Options)
    (reduction : LossReduction := .mean)
    (cast : Float → α := Runtime.ofFloat)
    (k : (model : nn.Sequential σ τ) →
      ScalarModule α (nn.paramShapes model) [σ, τ] → IO β) : IO β :=
  nn.withModel mkModel fun model => do
    let m ← instantiateMse
      (α := α) opts model (reduction := reduction) (cast := cast)
    k model m

/--
Build a sequential model, instantiate a custom supervised runtime module for it, and continue with
both values.

This packages the common public example pattern
`nn.withModel mkModel fun model => let m ← Module.instantiate ... (moduleDefOf model)`.
-/
def withModuleDefModel
    {σ τ : Shape} {α : Type} {β : Type}
    [Runtime.SemanticScalar α] [DecidableEq Shape] [Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (mkModel : nn.M (nn.Sequential σ τ))
    (opts : Options)
    (moduleDefOf : (model : nn.Sequential σ τ) →
      ScalarModuleDef (nn.paramShapes model) [σ, τ])
    (cast : Float → α := Runtime.ofFloat)
    (k : (model : nn.Sequential σ τ) →
      ScalarModule α (nn.paramShapes model) [σ, τ] → IO β) : IO β :=
  nn.withModel mkModel fun model => do
    let m ← instantiateModuleDefModel
      (α := α) opts model moduleDefOf (cast := cast)
    k model m

/--
Build a sequential model, instantiate a runtime module for a custom scalar loss program, and
continue with both values.

This is the custom-loss sibling of `withMseModel` / `withCrossEntropyOneHotModel`. Use it when the
model is ordinary `nn.Sequential`, but the loss needs task-specific logic beyond the standard MSE or
cross-entropy module constructors.
-/
def withScalarLossModel
    {σ τ : Shape} {α : Type} {β : Type}
    [Runtime.SemanticScalar α] [DecidableEq Shape] [Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (mkModel : nn.M (nn.Sequential σ τ))
    (opts : Options)
    (loss : ∀ {α : Type}, [Runtime.TensorScalar α] → [DecidableEq Shape] →
      _root_.Runtime.Autograd.TorchLean.Program α [τ, τ] Shape.scalar)
    (cast : Float → α := Runtime.ofFloat)
    (k : (model : nn.Sequential σ τ) →
      ScalarModule α (nn.paramShapes model) [σ, τ] → IO β) : IO β :=
  withModuleDefModel
    (α := α) (mkModel := mkModel) (opts := opts)
    (moduleDefOf := fun model => nn.scalarModuleDef model (loss := loss))
    (cast := cast) k

/--
Evaluate one supervised sample through a runtime module and return the scalar loss value.

This packages the common public example pattern `Module.forward ...; Tensor.toScalar`.
-/
def lossScalar {σ τ : Shape} {α : Type}
    [Runtime.TensorScalar α] [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (model : nn.Sequential σ τ)
    (m : ScalarModule α (nn.paramShapes model) [σ, τ])
    (sample : SupervisedSample α σ τ) : IO α := do
  let loss ← forward (α := α) m sample
  pure (Tensor.toScalar loss)

/--
Create an Adam optimizer handle bound to a concrete runtime module.

This packages the common public example pattern `optim.runtimeAdam ...; optim.handle m opt`.
-/
def adamHandle {α : Type}
    [Runtime.TensorScalar α] [DecidableEq Shape]
    {paramShapes inputShapes : List Shape}
    (m : ScalarModule α paramShapes inputShapes)
    (lr beta1 beta2 epsilon : α) := do
  let opt := NN.API.TorchLean.Optim.adam (α := α) lr beta1 beta2 epsilon
    (paramShapes := paramShapes)
  NN.API.TorchLean.Optim.handle (α := α) m opt

/--
Create an AdamW optimizer handle bound to a concrete runtime module.
-/
def adamWHandle {α : Type}
    [Runtime.TensorScalar α] [DecidableEq Shape]
    {paramShapes inputShapes : List Shape}
    (m : ScalarModule α paramShapes inputShapes)
    (lr weightDecay beta1 beta2 epsilon : α) := do
  let opt := NN.API.TorchLean.Optim.adamw (α := α)
    (paramShapes := paramShapes)
    lr weightDecay beta1 beta2 epsilon
  NN.API.TorchLean.Optim.handle (α := α) m opt

/--
Create an SGD optimizer handle bound to a concrete runtime module.
-/
def sgdHandle {α : Type}
    [Runtime.TensorScalar α] [DecidableEq Shape]
    {paramShapes inputShapes : List Shape}
    (m : ScalarModule α paramShapes inputShapes)
    (lr : α) := do
  let opt := NN.API.TorchLean.Optim.sgd (α := α) lr (paramShapes := paramShapes)
  NN.API.TorchLean.Optim.handle (α := α) m opt

/--
Create a one-step update function for any typed module input pack from the public optimizer config
used by the trainer API.

This is the generic bridge for custom training loops: richer examples can keep their own control
flow while still choosing SGD/Adam/AdamW through the same `optim.*` config surface as
`Trainer.RunConfig`.
-/
def optimizerInputs {α : Type}
    [Runtime.TensorScalar α] [Runtime.Scalar α] [DecidableEq Shape]
    {paramShapes inputShapes : List Shape}
    (m : ScalarModule α paramShapes inputShapes)
    (cfg : NN.API.TorchLean.Trainer.Optimizer) :
    IO (TensorPack α inputShapes → IO Unit) := do
  match cfg with
  | .sgd lr momentum =>
      if momentum == 0.0 then
        let optH ← sgdHandle m (Runtime.ofFloat lr)
        pure optH.step
      else
        let opt := NN.API.TorchLean.Optim.momentumSGD
          (α := α)
          (Runtime.ofFloat lr)
          (Runtime.ofFloat momentum)
          (paramShapes := paramShapes)
        let optH ← NN.API.TorchLean.Optim.handle (α := α) m opt
        pure optH.step
  | .adam lr beta1 beta2 epsilon =>
      let optH ← adamHandle m
        (Runtime.ofFloat lr)
        (Runtime.ofFloat beta1)
        (Runtime.ofFloat beta2)
        (Runtime.ofFloat epsilon)
      pure optH.step
  | .adamw lr weightDecay beta1 beta2 epsilon =>
      let optH ← adamWHandle m
        (Runtime.ofFloat lr)
        (Runtime.ofFloat weightDecay)
        (Runtime.ofFloat beta1)
        (Runtime.ofFloat beta2)
        (Runtime.ofFloat epsilon)
      pure optH.step

/--
Create a sample-step function from the public optimizer config used by the trainer API.

This is the bridge for custom training loops: richer examples can keep their own control flow while
still choosing SGD/Adam/AdamW through the same `optim.*` config surface as `Trainer.RunConfig`.
-/
def optimizerStep {α : Type} {σ τ : Shape}
    [Runtime.TensorScalar α] [Runtime.Scalar α] [DecidableEq Shape]
    {paramShapes : List Shape}
    (m : ScalarModule α paramShapes [σ, τ])
    (cfg : NN.API.TorchLean.Trainer.Optimizer) : IO (SupervisedSample α σ τ → IO Unit) := do
  optimizerInputs m cfg

end Module


end TorchLean
