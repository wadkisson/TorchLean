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

import Mathlib.Algebra.Order.Algebra
import NN.Spec.Autograd.AutogradSpec

@[expose] public section

namespace NN
namespace API
namespace TorchLean

/-!
# Runtime Core

Core exports for `NN.API.TorchLean`: execution options, tensor operations, losses, optimizers,
RL helpers, sequential model types, and deterministic runtime RNG helpers.
-/

/-!
### Core Exports

Most of this namespace is a curated re-export of `_root_.Runtime.Autograd.TorchLean.*`, so users can
`import NN.API.Runtime` and get a stable API surface without chasing implementation modules.

The exported names fall into these groups:
- execution control: `Backend`, `Options`
- program interface: `Ops`, `RefTy`, `Program`, `CompiledGraph`, `CompiledScalar`, ...
- primitive tensor ops: `add`, `matmul`, `reshape`, elementwise activations, pooling, ...
- training utilities: `trainCycle*`, `meanLoss`
-/

export _root_.Runtime.Autograd.TorchLean (Backend Options TensorRef Param AnyParam)
export _root_.Runtime.Autograd.TorchLean (CompiledScalar compileScalar)
export _root_.Runtime.Autograd.TorchLean (CompiledGraph compileGraph)
export _root_.Runtime.Autograd.TorchLean (ParamList ScalarTrainer scalarTrainer)
export _root_.Runtime.Autograd.TorchLean (TList Ops Ref RefList CurriedRef RefTy Program)
export _root_.Runtime.Autograd.TorchLean.Curried (Fn curry uncurry)
export _root_.Runtime.Autograd.TorchLean.CurriedRef (uncurry applyVarList)
export _root_.Runtime.Autograd.TorchLean.RefList (append)
export _root_.Runtime.Autograd.TorchLean
  (const add sub mul scale abs sqrt clamp max min
   broadcastTo reshape transpose2d swapAdjacentAtDepth reduceSum reduceMean
   gatherScalar gatherRow gatherScalarNat gatherVecNat gatherRowsNat scatterAddVec
     scatterAddRow
   matmul concatVectors
   maxPool avgPool smoothMaxPool
   maxPool2d maxPool2dPad smoothMaxPool2d avgPool2d avgPool2dPad
   relu silu gelu sigmoid tanh softmax softplus exp log inv safeLog logSoftmax
   sum flatten
   linear mseLoss layerNorm batchnormChannelFirst multiHeadAttention multiHeadAttentionOutputBias
   conv convTranspose conv2d)
export _root_.Runtime.Autograd.TorchLean
  (scalarOf tlistSingleton tlistPair tlistTriple tlistQuad trainCycleSGD trainCycleOptim meanLoss)

/-- Public name for TorchLean's shape-indexed tensor-pack / typed tuple representation. -/
abbrev TensorPack (α : Type) (shapes : List Spec.Shape) := TList α shapes

/-- Construct a one-tensor pack. -/
abbrev tensorpackSingleton {α : Type} {s : Spec.Shape} (x : Spec.Tensor α s) :
    TensorPack α [s] :=
  tlistSingleton x

/-- Construct a two-tensor pack. -/
abbrev tensorpackPair {α : Type} {s₁ s₂ : Spec.Shape}
    (x₁ : Spec.Tensor α s₁) (x₂ : Spec.Tensor α s₂) :
    TensorPack α [s₁, s₂] :=
  tlistPair x₁ x₂

/-- Construct a three-tensor pack. -/
abbrev tensorpackTriple {α : Type} {s₁ s₂ s₃ : Spec.Shape}
    (x₁ : Spec.Tensor α s₁) (x₂ : Spec.Tensor α s₂) (x₃ : Spec.Tensor α s₃) :
    TensorPack α [s₁, s₂, s₃] :=
  tlistTriple x₁ x₂ x₃

/-- Construct a four-tensor pack. -/
abbrev tensorpackQuad {α : Type} {s₁ s₂ s₃ s₄ : Spec.Shape}
    (x₁ : Spec.Tensor α s₁) (x₂ : Spec.Tensor α s₂)
    (x₃ : Spec.Tensor α s₃) (x₄ : Spec.Tensor α s₄) :
    TensorPack α [s₁, s₂, s₃, s₄] :=
  tlistQuad x₁ x₂ x₃ x₄

/-
`TList` is a *typed list of tensors* whose shape list lives in the type.

It is great for safety (the compiler tracks parameter order/shapes), but raw destructuring
with `.cons ... .nil` is noisy in examples.

For tuple-like constructors/accessors (`tensorpack.unpackPair`, `tensorpack.second`, etc.), see
`NN/API/Public/TensorPack.lean`.
-/

namespace RefList

/-- Unpack a two-element `RefList` into a pair. -/
def unpackPair {Ref : Spec.Shape → Type} {s₁ s₂ : Spec.Shape} :
    TorchLean.RefList Ref [s₁, s₂] → (Ref s₁ × Ref s₂)
  | .cons x₁ (.cons x₂ .nil) => (x₁, x₂)

end RefList

namespace F
/- Functional tensor helpers mirroring `torch.nn.functional`-style building blocks. -/
export _root_.Runtime.Autograd.TorchLean.F
  (square checkpoint
   addB mulB
   embedding embeddingRowsNat embeddingBatchSeqNat mean
   detach
   dropoutSeeded)
end F

namespace Loss
/- Loss helpers mirroring the usual `torch.nn.functional` loss family. -/
export _root_.Runtime.Autograd.TorchLean.Loss
  (Reduction mse nllOneHot crossEntropyOneHot nllIndex nllNat crossEntropyIndex crossEntropyNat
    rowTargetFlatIndices nllRowsNat crossEntropyRowsNat
    bceWithLogits bce)
end Loss

namespace Norm
/- Normalization helpers exported by the runtime API. -/
export _root_.Runtime.Autograd.TorchLean.Norm
  (rmsNormLast instanceNorm2dNchw groupNorm2dNchw
   batchNorm2dNchwTrain batchNorm2dNchwTrainStats batchNormRunningUpdate
   batchNorm2dNchwEval batchNorm2dChwEval)
end Norm

namespace Autodiff
/- Autodiff entrypoints for compiled and eager runtime programs. -/
export _root_.Runtime.Autograd.TorchLean.Autodiff
  (compileLoss compileGraph
   gradParams gradInputs
   vjpOutParams vjpOutInputs
   jacrevOutParams jacrevOutInputs
   jacfwdInput
   hessianInput
   jvpLossParams jvpLossInputs
   hvpParams hvpInputs)
end Autodiff

namespace Metrics
/- Small post-processing metrics such as argmax and classification correctness. -/
export _root_.Runtime.Autograd.TorchLean.Metrics
  (argmax? classOfOneHot? correctOneHot?)
end Metrics

namespace Optim
/- Optimizer constructors used by the runtime trainer API. -/
export _root_.Runtime.Autograd.TorchLean.Optim (StateList Optimizer)
export _root_.Runtime.Autograd.TorchLean.Optim
  (sgd momentumSGD adagrad rmsprop adam adamw adadelta projectedSGD muon)

/-!
### Optimizer Handles (PyTorch-Like)

TorchLean optimizers are purely functional in their state: `opt.step` returns a new state.

This small wrapper stores the optimizer state in an `IO.Ref` so users can write:

```
let h ← API.TorchLean.Optim.handle m (TorchLean.Optim.sgd lr)
h.step sample
```

without manually threading the optimizer state through the training loop.
-/

/--
A mutable optimizer handle bound to a concrete TorchLean `ScalarModule`.

The optimizer state is stored in an `IO.Ref` and updated when you call `h.step sample`.
-/
structure Handle (α : Type) [Context α] [DecidableEq Spec.Shape]
    (paramShapes inputShapes : List Spec.Shape) (State : Type) where
  /-- The module whose parameters will be updated in-place. -/
  module : _root_.Runtime.Autograd.TorchLean.Module.ScalarModule α paramShapes inputShapes
  /-- Mutable optimizer state. -/
  state : IO.Ref State
  /-- One training step on a single sample, updating the optimizer state. -/
  step : TList α inputShapes → IO Unit

/--
Create an optimizer handle for a module by initializing optimizer state from the module's current
parameters.
-/
def handle {α : Type} [Context α] [DecidableEq Spec.Shape]
    {paramShapes inputShapes : List Spec.Shape}
    (m : _root_.Runtime.Autograd.TorchLean.Module.ScalarModule α paramShapes inputShapes)
    (opt : Optimizer α paramShapes) :
    IO (Handle α paramShapes inputShapes opt.State) := do
  let st0 ← _root_.Runtime.Autograd.TorchLean.Module.ScalarModule.initOptim (m := m) opt
  let stRef ← IO.mkRef st0
  let step (sample : TList α inputShapes) : IO Unit := do
    let st ← stRef.get
    let st' ← _root_.Runtime.Autograd.TorchLean.Module.ScalarModule.stepWith (m := m) opt st sample
    stRef.set st'
  pure { module := m, state := stRef, step := step }
end Optim

namespace RL
/- Reinforcement-learning helpers spanning bandits, tabular control, value learning, and policy
objectives. -/
  export _root_.Spec.RL
    (AdvantageStep
     continueMask discountedBackup tdTarget tdResidual
     discountedReturns discountedReturnsFrom discountedReturnsDone
     generalizedAdvantageEstimation returnsFromAdvantages
     ValueFunction Policy FiniteMDP
     valueAt stateActionValue actionValues
     bellmanPolicy bellmanOptimality)
  export _root_.Spec.RL.FiniteMDP (toEnv)
  export _root_.Spec.RL.Markov
    (ValueFunction Policy MDP Valid
     transitionMeasure
     expectedNextValue actionValue
     bellmanPolicy bellmanOptimality)
  export _root_.Spec.RL.FiniteStochastic
    (MDP Valid
     expectedNextValue actionValue actionValues
     bellmanPolicy bellmanOptimality)
  export _root_.Runtime.RL.Core
    (Transition IndexedTransition
     discountedReturnsVecFrom discountedReturnsVec discountedReturnsVecDone
     generalizedAdvantageEstimationVec returnsFromAdvantagesVec
     squaredError huberLoss)
  export _root_.Runtime.RL.Bandits
    (ValueState PreferenceState
     greedyAction? epsilonGreedyAction?
     sampleAverageStep totalPulls
     ucb1Bonus ucb1Scores ucb1Action?
     gradientPolicy gradientBanditStep)
  export _root_.Runtime.RL.Bandits.ValueState (init)
  export _root_.Runtime.RL.Bandits.PreferenceState (init)
export _root_.Runtime.RL.Tabular
  (actionRow maxActionValue greedyAction? expectedActionValue
   td0Update
   sarsaTarget expectedSarsaTarget qLearningTarget doubleQTarget
   sarsaUpdate expectedSarsaUpdate qLearningUpdate
   doubleQUpdateLeft doubleQUpdateRight)
  export _root_.Runtime.RL.ValueLearning
    (chosenActionValue maxQValue
     dqnTarget doubleDqnTarget
     dqnResidual dqnMSELoss dqnHuberLoss doubleDqnResidual
     ddpgActorObjective ddpgCriticTarget td3Target
     sacTarget sacActorObjective)
  export _root_.Runtime.RL.PolicyGradient
    (actionPolicy actionProbability actionLogProbability entropyBonus
     reinforceLoss actorLoss criticLoss actorCriticLoss
     importanceRatio ppoClippedObjective ppoLoss)

  namespace Autograd
  /- Differentiable (autograd-capable) policy-gradient objectives over TorchLean `Ops`. -/
  export _root_.Runtime.RL.PolicyGradient.Autograd
    (actionLogProbOneHotBatch
     entropyMean
     ppoClippedObjectiveBatch
     ppoLossBatch
     ppoActorCriticScalarModuleDef)
  end Autograd
  end RL

namespace LayerCore
/- Neural-network layer constructors and sequential-model helpers. -/
export _root_.Runtime.Autograd.TorchLean.NN
  (Mode LayerDef Seq
   linear rnn gru mamba lstm
   relu silu gelu sigmoid tanh softmax square sum flatten dropout
   layerNorm rmsNorm
   batchnormChannelFirst batchnormChannelFirstEval batchnormChannelFirstMode
   instanceNorm2dNchw groupNorm2dNchw batchNorm2dNchw batchNorm2dNchwMode
   multiHeadAttention multiHeadAttentionOutputBias conv2d
   maxPool2d maxPool2dPad avgPool2d avgPool2dPad
   singleLayer)

/-
To keep example code "PyTorch-like", the `seq!` macro supports stacking either:
- a single layer (`LayerDef σ τ`), or
- an already-sequential model (`Seq σ τ`)
in the same `seq! ...` expression.

Lean's coercion insertion is not always reliable in partially-applied situations, so we provide an
explicit, typeclass-driven adapter that `seq!` can use.
-/
universe u v

/--
Adapter typeclass used by the `seq!` macro to treat both layers and already-sequential models as
composable building blocks.

This exists purely for ergonomics: it lets examples mix `LayerDef` and `Seq` in the same `seq!`
expression without relying on Lean's coercion insertion heuristics.
-/
class AsSeqK (F : Spec.Shape → Spec.Shape → Sort u) where
  /-- Convert a layer-like thing into a `Seq` so `seq!` can compose it. -/
  asSeq : {σ τ : Spec.Shape} → F σ τ → Seq σ τ

/-- A single `LayerDef` can always be viewed as a 1-layer sequential model (`singleLayer`). -/
instance : AsSeqK LayerDef where
  asSeq := fun {_σ _τ} layer => singleLayer layer

/-- A sequential model is already a sequential model (identity). -/
instance : AsSeqK Seq where
  asSeq := fun {_σ _τ} s => s

/--
Compose either layers or sequential models without relying on coercions.

This is the helper used by the `seq! ...` macro so examples can write
`seq! layer1, model2, layer3` while still mirroring PyTorch's "stack layers" style.
-/
def compAny {σ τ υ : Spec.Shape}
    {F : Spec.Shape → Spec.Shape → Sort u} {G : Spec.Shape → Spec.Shape → Sort v}
    [AsSeqK F] [AsSeqK G] (f : F σ τ) (g : G τ υ) : Seq σ υ :=
  _root_.Runtime.Autograd.TorchLean.NN.Seq.comp (AsSeqK.asSeq f) (AsSeqK.asSeq g)

namespace Seq
export _root_.Runtime.Autograd.TorchLean.NN.Seq
  (paramShapes paramRequiresGrad initParams runtimeInit? hasBufferUpdates comp updateBuffers
   programWithMode forwardProgram
   scalarModuleDefWithMode scalarModuleDef
   mseScalarModuleDefWithMode mseScalarModuleDef
   crossEntropyOneHotScalarModuleDefWithMode crossEntropyOneHotScalarModuleDef
   compileForwardWithMode compileForward
   forward predict forwardArtifact)
end Seq
end LayerCore

namespace Random
/-!
Deterministic RNG helpers re-exported for runtime callers.

These are deterministic utilities used by examples and training loops (`keyOf`, `nextSeed`,
`uniform`, `mask`).
-/
export _root_.Runtime.Autograd.TorchLean.Random (keyOf nextSeed uniform mask)
end Random

end TorchLean
end API
end NN
