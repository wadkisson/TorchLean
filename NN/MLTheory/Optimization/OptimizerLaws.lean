/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Optim.Optimizers

/-!
# Optimizer Law Interface

This module gives TorchLean optimizers a small proof layer interface.

Runtime optimizers live in `NN.Runtime.Optim.Optimizers` as executable tensor equations.  The
definitions below package those equations as shape-polymorphic optimizers and expose the update
specs that a new optimizer should prove against.

The pattern for adding an optimizer is:

1. define a pure per-tensor `init` and `update` equation;
2. package it as a `TensorOptimizer`;
3. register a `StepSpec` giving the proof layer next-state and next-parameter equations;
4. prove optimizer-specific algebraic facts as consequences of that generic interface.

The one-step registrations are often definitional.  The reusable content is the generic layer:
higher-level trainer proofs can quantify over any `TensorOptimizer`, reason about whole gradient
streams via `runSteps`, and specialize to SGD/Adam/Muon only at the boundary.
-/

@[expose] public section

namespace Optim

open Spec
open Tensor

variable {α : Type} [Context α]

/-- A shape-polymorphic per-tensor optimizer. -/
structure TensorOptimizer (α : Type) [Context α] where
  /-- Per-parameter optimizer state for a tensor of shape `s`. -/
  State : Shape → Type
  /-- Initialize optimizer state from the current parameter tensor. -/
  init : {s : Shape} → Tensor α s → State s
  /-- One update from state, parameters, and gradients. -/
  update : {s : Shape} → State s → Tensor α s → Tensor α s → State s × Tensor α s

namespace TensorOptimizer

section ConcreteOptimizers

variable [DecidableRel ((· > ·) : α → α → Prop)]

/-- Package plain SGD as a `TensorOptimizer`. -/
def sgd (lr : α) : TensorOptimizer α :=
  { State := SGD.State α
    init := fun {s} p => SGD.init (α := α) (s := s) lr p
    update := fun {_s} st p g => (st, SGD.update (α := α) st p g) }

/-- Package momentum SGD as a `TensorOptimizer`. -/
def momentumSGD (lr momentum : α) : TensorOptimizer α :=
  { State := MomentumSGD.State α
    init := fun {s} p => MomentumSGD.init (α := α) (s := s) lr momentum p
    update := fun {_s} st p g => MomentumSGD.update (α := α) st p g }

/-- Package AdaGrad as a `TensorOptimizer`. -/
def adagrad (lr epsilon : α) : TensorOptimizer α :=
  { State := AdaGrad.State α
    init := fun {s} p => AdaGrad.init (α := α) (s := s) lr epsilon p
    update := fun {_s} st p g => AdaGrad.update (α := α) st p g }

/-- Package RMSProp as a `TensorOptimizer`. -/
def rmsprop (lr decay epsilon : α) : TensorOptimizer α :=
  { State := RMSProp.State α
    init := fun {s} p => RMSProp.init (α := α) (s := s) lr decay epsilon p
    update := fun {_s} st p g => RMSProp.update (α := α) st p g }

/-- Package Adam as a `TensorOptimizer`. -/
def adam (lr beta1 beta2 epsilon : α) : TensorOptimizer α :=
  { State := Adam.State α
    init := fun {s} p => Adam.init (α := α) (s := s) lr beta1 beta2 epsilon p
    update := fun {_s} st p g => Adam.update (α := α) st p g }

/-- Package AdamW as a `TensorOptimizer`. -/
def adamw (lr weightDecay beta1 beta2 epsilon : α) : TensorOptimizer α :=
  { State := AdamW.State α
    init := fun {s} p => AdamW.init (α := α) (s := s) lr weightDecay beta1 beta2 epsilon p
    update := fun {_s} st p g => AdamW.update (α := α) st p g }

/-- Package Adadelta as a `TensorOptimizer`. -/
def adadelta (lr rho epsilon : α) : TensorOptimizer α :=
  { State := Adadelta.State α
    init := fun {s} p => Adadelta.init (α := α) (s := s) lr rho epsilon p
    update := fun {_s} st p g => Adadelta.update (α := α) st p g }

/-- Package Muon-style orthogonalized momentum as a `TensorOptimizer`. -/
def muon (lr momentum : α)
    (orthogonalizer : {s : Shape} → Muon.Orthogonalizer α s :=
      fun {s} => Muon.identityOrthogonalizer (α := α) (s := s)) :
    TensorOptimizer α :=
  { State := Muon.State α
    init := fun {s} p => Muon.init (α := α) (s := s) lr momentum (orthogonalizer (s := s)) p
    update := fun {_s} st p g => Muon.update (α := α) st p g }

end ConcreteOptimizers

/-- State/parameter pair threaded by an optimizer for one fixed tensor shape. -/
abbrev Step (opt : TensorOptimizer α) (s : Shape) :=
  opt.State s × Tensor α s

/-- Run one optimizer step on a state/parameter pair. -/
def step (opt : TensorOptimizer α) {s : Shape}
    (current : Step opt s) (grads : Tensor α s) : Step opt s :=
  opt.update current.1 current.2 grads

/-- Run a finite stream of gradients through an optimizer. -/
def runSteps (opt : TensorOptimizer α) {s : Shape} :
    Step opt s → List (Tensor α s) → Step opt s
  | current, [] => current
  | current, grads :: rest => runSteps opt (opt.step current grads) rest

/-- Empty gradient streams leave optimizer state and parameters unchanged. -/
theorem runSteps_nil (opt : TensorOptimizer α) {s : Shape}
    (current : Step opt s) :
    opt.runSteps current [] = current := by
  rfl

/-- Cons form of `runSteps`. -/
theorem runSteps_cons (opt : TensorOptimizer α) {s : Shape}
    (current : Step opt s) (grads : Tensor α s) (rest : List (Tensor α s)) :
    opt.runSteps current (grads :: rest) = opt.runSteps (opt.step current grads) rest := by
  rfl

/--
Splitting a gradient stream and running the two pieces sequentially gives the same state and
parameters as running the concatenated stream.
-/
theorem runSteps_append (opt : TensorOptimizer α) {s : Shape}
    (current : Step opt s) (left right : List (Tensor α s)) :
    opt.runSteps current (left ++ right) = opt.runSteps (opt.runSteps current left) right := by
  induction left generalizing current with
  | nil =>
      rfl
  | cons grads rest ih =>
      simp [runSteps, step, ih]

/-- Optimizer state after a finite gradient stream. -/
def stateAfter (opt : TensorOptimizer α) {s : Shape}
    (current : Step opt s) (grads : List (Tensor α s)) : opt.State s :=
  (opt.runSteps current grads).1

/-- Optimizer parameters after a finite gradient stream. -/
def paramsAfter (opt : TensorOptimizer α) {s : Shape}
    (current : Step opt s) (grads : List (Tensor α s)) : Tensor α s :=
  (opt.runSteps current grads).2

/-- State projection of `runSteps_append`. -/
theorem stateAfter_append (opt : TensorOptimizer α) {s : Shape}
    (current : Step opt s) (left right : List (Tensor α s)) :
    opt.stateAfter current (left ++ right) =
      opt.stateAfter (opt.runSteps current left) right := by
  exact congrArg Prod.fst (opt.runSteps_append current left right)

/-- Parameter projection of `runSteps_append`. -/
theorem paramsAfter_append (opt : TensorOptimizer α) {s : Shape}
    (current : Step opt s) (left right : List (Tensor α s)) :
    opt.paramsAfter current (left ++ right) =
      opt.paramsAfter (opt.runSteps current left) right := by
  exact congrArg Prod.snd (opt.runSteps_append current left right)

end TensorOptimizer

/-! ## Generic step specifications -/

/--
Proof-facing specification of one optimizer step.

An optimizer-specific file only has to identify the next-state and next-parameter equations once.
The generic theorems below then lift that one-step fact to whole finite gradient streams.
-/
structure StepSpec (opt : TensorOptimizer α) where
  /-- Spec equation for the next optimizer state. -/
  nextState : {s : Shape} → opt.State s → Tensor α s → Tensor α s → opt.State s
  /-- Spec equation for the next parameter tensor. -/
  nextParams : {s : Shape} → opt.State s → Tensor α s → Tensor α s → Tensor α s
  /-- The executable optimizer update agrees with the stated step equations. -/
  update_eq : ∀ {s : Shape} (state : opt.State s) (params grads : Tensor α s),
    opt.update state params grads = (nextState state params grads, nextParams state params grads)

namespace StepSpec

variable {opt : TensorOptimizer α}

/-- Run one step through the proof layer equations. -/
def step (law : StepSpec opt) {s : Shape}
    (current : TensorOptimizer.Step opt s) (grads : Tensor α s) :
    TensorOptimizer.Step opt s :=
  (law.nextState current.1 current.2 grads, law.nextParams current.1 current.2 grads)

/-- Run a finite stream of gradients through the proof layer equations. -/
def runSteps (law : StepSpec opt) {s : Shape} :
    TensorOptimizer.Step opt s → List (Tensor α s) → TensorOptimizer.Step opt s
  | current, [] => current
  | current, grads :: rest => runSteps law (law.step current grads) rest

/-- A registered step spec agrees with the executable optimizer for one step. -/
theorem step_eq_optimizer_step (law : StepSpec opt) {s : Shape}
    (current : TensorOptimizer.Step opt s) (grads : Tensor α s) :
    law.step current grads = opt.step current grads := by
  cases current with
  | mk state params =>
      simp [step, TensorOptimizer.step, law.update_eq]

/--
A registered one-step optimizer spec agrees with the executable optimizer over any finite gradient
stream.  This is the general theorem optimizer-specific registrations feed into.
-/
theorem runSteps_eq_optimizer_runSteps (law : StepSpec opt) {s : Shape}
    (current : TensorOptimizer.Step opt s) (grads : List (Tensor α s)) :
    law.runSteps current grads = opt.runSteps current grads := by
  induction grads generalizing current with
  | nil =>
      rfl
  | cons grads rest ih =>
      simp [runSteps, TensorOptimizer.runSteps, step_eq_optimizer_step, ih]

/--
The proof layer equations compose over concatenated gradient streams just like the executable
optimizer.
-/
theorem runSteps_append (law : StepSpec opt) {s : Shape}
    (current : TensorOptimizer.Step opt s)
    (left right : List (Tensor α s)) :
    law.runSteps current (left ++ right) = law.runSteps (law.runSteps current left) right := by
  induction left generalizing current with
  | nil =>
      rfl
  | cons grads rest ih =>
      simp [runSteps, step, ih]

end StepSpec

/-! ## Named update specs -/

variable [DecidableRel ((· > ·) : α → α → Prop)]

namespace SGD

/-- The proof layer SGD update equation. -/
def updateSpec {s : Shape} (state : State α s) (params grads : Tensor α s) :
    State α s × Tensor α s :=
  (state, subSpec params (scaleSpec grads state.lr))

/-- The executable SGD optimizer follows `updateSpec`. -/
theorem update_eq_spec {s : Shape} (state : State α s) (params grads : Tensor α s) :
    (state, update state params grads) = updateSpec state params grads := by
  rfl

/-- Register SGD under the generic optimizer-step interface. -/
def stepSpec (lr : α) : StepSpec (TensorOptimizer.sgd (α := α) lr) where
  nextState := fun state _params _grads => state
  nextParams := fun state params grads => subSpec params (scaleSpec grads state.lr)
  update_eq := by
    intro _s _state _params _grads
    rfl

/-- SGD's registered step spec agrees with executable SGD over any finite gradient stream. -/
theorem runSteps_eq_stepSpec (lr : α) {s : Shape}
    (current : TensorOptimizer.Step (TensorOptimizer.sgd (α := α) lr) s)
    (grads : List (Tensor α s)) :
    (stepSpec (α := α) lr).runSteps current grads =
      (TensorOptimizer.sgd (α := α) lr).runSteps current grads :=
  StepSpec.runSteps_eq_optimizer_runSteps (stepSpec (α := α) lr) current grads

end SGD

namespace MomentumSGD

/-- The proof layer momentum-SGD update equation. -/
def updateSpec {s : Shape} (state : State α s) (params grads : Tensor α s) :
    State α s × Tensor α s :=
  let newBuf := OptimizerUtils.updateMomentumBuf state.buf state.momentum grads
  ({ state with buf := newBuf }, subSpec params (scaleSpec newBuf state.lr))

/-- The executable momentum-SGD optimizer follows `updateSpec`. -/
theorem update_eq_spec {s : Shape} (state : State α s) (params grads : Tensor α s) :
    update state params grads = updateSpec state params grads := by
  rfl

/-- Register momentum SGD under the generic optimizer-step interface. -/
def stepSpec (lr momentum : α) :
    StepSpec (TensorOptimizer.momentumSGD (α := α) lr momentum) where
  nextState := fun state _params grads =>
    { state with buf := OptimizerUtils.updateMomentumBuf state.buf state.momentum grads }
  nextParams := fun state params grads =>
    let newBuf := OptimizerUtils.updateMomentumBuf state.buf state.momentum grads
    subSpec params (scaleSpec newBuf state.lr)
  update_eq := by
    intro _s _state _params _grads
    rfl

/-- Momentum SGD's registered step spec agrees with executable momentum SGD over streams. -/
theorem runSteps_eq_stepSpec (lr momentum : α) {s : Shape}
    (current : TensorOptimizer.Step (TensorOptimizer.momentumSGD (α := α) lr momentum) s)
    (grads : List (Tensor α s)) :
    (stepSpec (α := α) lr momentum).runSteps current grads =
      (TensorOptimizer.momentumSGD (α := α) lr momentum).runSteps current grads :=
  StepSpec.runSteps_eq_optimizer_runSteps (stepSpec (α := α) lr momentum) current grads

end MomentumSGD

namespace AdaGrad

/-- The proof layer AdaGrad update equation. -/
def updateSpec {s : Shape} (state : State α s) (params grads : Tensor α s) :
    State α s × Tensor α s :=
  let squaredGrads := squareSpec grads
  let newAccumulator := addSpec state.accumulator squaredGrads
  let adaptiveLR := OptimizerUtils.mkAdaptiveLR state.lr state.epsilon newAccumulator
  ({ state with accumulator := newAccumulator }, subSpec params (mulSpec adaptiveLR grads))

/-- The executable AdaGrad optimizer follows `updateSpec`. -/
theorem update_eq_spec {s : Shape} (state : State α s) (params grads : Tensor α s) :
    update state params grads = updateSpec state params grads := by
  rfl

/-- Register AdaGrad under the generic optimizer-step interface. -/
def stepSpec (lr epsilon : α) : StepSpec (TensorOptimizer.adagrad (α := α) lr epsilon) where
  nextState := fun state _params grads =>
    { state with accumulator := addSpec state.accumulator (squareSpec grads) }
  nextParams := fun state params grads =>
    let squaredGrads := squareSpec grads
    let newAccumulator := addSpec state.accumulator squaredGrads
    let adaptiveLR := OptimizerUtils.mkAdaptiveLR state.lr state.epsilon newAccumulator
    subSpec params (mulSpec adaptiveLR grads)
  update_eq := by
    intro _s _state _params _grads
    rfl

/-- AdaGrad's registered step spec agrees with executable AdaGrad over streams. -/
theorem runSteps_eq_stepSpec (lr epsilon : α) {s : Shape}
    (current : TensorOptimizer.Step (TensorOptimizer.adagrad (α := α) lr epsilon) s)
    (grads : List (Tensor α s)) :
    (stepSpec (α := α) lr epsilon).runSteps current grads =
      (TensorOptimizer.adagrad (α := α) lr epsilon).runSteps current grads :=
  StepSpec.runSteps_eq_optimizer_runSteps (stepSpec (α := α) lr epsilon) current grads

end AdaGrad

namespace RMSProp

/-- The proof layer RMSProp update equation. -/
def updateSpec {s : Shape} (state : State α s) (params grads : Tensor α s) :
    State α s × Tensor α s :=
  let squaredGrads := squareSpec grads
  let newAccumulator :=
    addSpec (scaleSpec state.accumulator state.decay)
      (scaleSpec squaredGrads (1 - state.decay))
  let adaptiveLR := OptimizerUtils.mkAdaptiveLR state.lr state.epsilon newAccumulator
  ({ state with accumulator := newAccumulator }, subSpec params (mulSpec adaptiveLR grads))

/-- The executable RMSProp optimizer follows `updateSpec`. -/
theorem update_eq_spec {s : Shape} (state : State α s) (params grads : Tensor α s) :
    update state params grads = updateSpec state params grads := by
  rfl

/-- Register RMSProp under the generic optimizer-step interface. -/
def stepSpec (lr decay epsilon : α) :
    StepSpec (TensorOptimizer.rmsprop (α := α) lr decay epsilon) where
  nextState := fun state _params grads =>
    let squaredGrads := squareSpec grads
    let newAccumulator :=
      addSpec (scaleSpec state.accumulator state.decay)
        (scaleSpec squaredGrads (1 - state.decay))
    { state with accumulator := newAccumulator }
  nextParams := fun state params grads =>
    let squaredGrads := squareSpec grads
    let newAccumulator :=
      addSpec (scaleSpec state.accumulator state.decay)
        (scaleSpec squaredGrads (1 - state.decay))
    let adaptiveLR := OptimizerUtils.mkAdaptiveLR state.lr state.epsilon newAccumulator
    subSpec params (mulSpec adaptiveLR grads)
  update_eq := by
    intro _s _state _params _grads
    rfl

/-- RMSProp's registered step spec agrees with executable RMSProp over streams. -/
theorem runSteps_eq_stepSpec (lr decay epsilon : α) {s : Shape}
    (current : TensorOptimizer.Step (TensorOptimizer.rmsprop (α := α) lr decay epsilon) s)
    (grads : List (Tensor α s)) :
    (stepSpec (α := α) lr decay epsilon).runSteps current grads =
      (TensorOptimizer.rmsprop (α := α) lr decay epsilon).runSteps current grads :=
  StepSpec.runSteps_eq_optimizer_runSteps (stepSpec (α := α) lr decay epsilon) current grads

end RMSProp

namespace Adam

/-- The proof layer Adam update equation. -/
def updateSpec {s : Shape} (state : State α s) (params grads : Tensor α s) :
    State α s × Tensor α s :=
  let t' := state.t + 1
  let m' := addSpec (scaleSpec state.m state.beta1) (scaleSpec grads (1 - state.beta1))
  let v' := addSpec (scaleSpec state.v state.beta2) (scaleSpec (squareSpec grads) (1 - state.beta2))
  let mHat := scaleSpec m' (1 / (1 - scalarPowNat state.beta1 t'))
  let vHat := scaleSpec v' (1 / (1 - scalarPowNat state.beta2 t'))
  let adaptiveLR := OptimizerUtils.mkAdaptiveLR state.lr state.epsilon vHat
  ({ state with m := m', v := v', t := t' }, subSpec params (mulSpec adaptiveLR mHat))

/-- The executable Adam optimizer follows `updateSpec`. -/
theorem update_eq_spec {s : Shape} (state : State α s) (params grads : Tensor α s) :
    update state params grads = updateSpec state params grads := by
  rfl

/-- Register Adam under the generic optimizer-step interface. -/
def stepSpec (lr beta1 beta2 epsilon : α) :
    StepSpec (TensorOptimizer.adam (α := α) lr beta1 beta2 epsilon) where
  nextState := fun state _params grads =>
    let t' := state.t + 1
    let m' := addSpec (scaleSpec state.m state.beta1) (scaleSpec grads (1 - state.beta1))
    let v' := addSpec (scaleSpec state.v state.beta2)
      (scaleSpec (squareSpec grads) (1 - state.beta2))
    { state with m := m', v := v', t := t' }
  nextParams := fun state params grads =>
    let t' := state.t + 1
    let m' := addSpec (scaleSpec state.m state.beta1) (scaleSpec grads (1 - state.beta1))
    let v' := addSpec (scaleSpec state.v state.beta2)
      (scaleSpec (squareSpec grads) (1 - state.beta2))
    let mHat := scaleSpec m' (1 / (1 - scalarPowNat state.beta1 t'))
    let vHat := scaleSpec v' (1 / (1 - scalarPowNat state.beta2 t'))
    let adaptiveLR := OptimizerUtils.mkAdaptiveLR state.lr state.epsilon vHat
    subSpec params (mulSpec adaptiveLR mHat)
  update_eq := by
    intro _s _state _params _grads
    rfl

/-- Adam's registered step spec agrees with executable Adam over streams. -/
theorem runSteps_eq_stepSpec (lr beta1 beta2 epsilon : α) {s : Shape}
    (current : TensorOptimizer.Step (TensorOptimizer.adam (α := α) lr beta1 beta2 epsilon) s)
    (grads : List (Tensor α s)) :
    (stepSpec (α := α) lr beta1 beta2 epsilon).runSteps current grads =
      (TensorOptimizer.adam (α := α) lr beta1 beta2 epsilon).runSteps current grads :=
  StepSpec.runSteps_eq_optimizer_runSteps
    (stepSpec (α := α) lr beta1 beta2 epsilon) current grads

end Adam

namespace AdamW

/-- The proof layer AdamW update equation. -/
def updateSpec {s : Shape} (state : State α s) (params grads : Tensor α s) :
    State α s × Tensor α s :=
  let t' := state.t + 1
  let m' := addSpec (scaleSpec state.m state.beta1) (scaleSpec grads (1 - state.beta1))
  let v' := addSpec (scaleSpec state.v state.beta2) (scaleSpec (squareSpec grads) (1 - state.beta2))
  let mHat := scaleSpec m' (1 / (1 - scalarPowNat state.beta1 t'))
  let vHat := scaleSpec v' (1 / (1 - scalarPowNat state.beta2 t'))
  let adaptiveLR := OptimizerUtils.mkAdaptiveLR state.lr state.epsilon vHat
  let decayedParams := subSpec params (scaleSpec params (state.lr * state.weight_decay))
  ({ state with m := m', v := v', t := t' }, subSpec decayedParams (mulSpec adaptiveLR mHat))

/-- The executable AdamW optimizer follows `updateSpec`. -/
theorem update_eq_spec {s : Shape} (state : State α s) (params grads : Tensor α s) :
    update state params grads = updateSpec state params grads := by
  rfl

/-- Register AdamW under the generic optimizer-step interface. -/
def stepSpec (lr weightDecay beta1 beta2 epsilon : α) :
    StepSpec (TensorOptimizer.adamw (α := α) lr weightDecay beta1 beta2 epsilon) where
  nextState := fun state _params grads =>
    let t' := state.t + 1
    let m' := addSpec (scaleSpec state.m state.beta1) (scaleSpec grads (1 - state.beta1))
    let v' := addSpec (scaleSpec state.v state.beta2)
      (scaleSpec (squareSpec grads) (1 - state.beta2))
    { state with m := m', v := v', t := t' }
  nextParams := fun state params grads =>
    let t' := state.t + 1
    let m' := addSpec (scaleSpec state.m state.beta1) (scaleSpec grads (1 - state.beta1))
    let v' := addSpec (scaleSpec state.v state.beta2)
      (scaleSpec (squareSpec grads) (1 - state.beta2))
    let mHat := scaleSpec m' (1 / (1 - scalarPowNat state.beta1 t'))
    let vHat := scaleSpec v' (1 / (1 - scalarPowNat state.beta2 t'))
    let adaptiveLR := OptimizerUtils.mkAdaptiveLR state.lr state.epsilon vHat
    let decayedParams := subSpec params (scaleSpec params (state.lr * state.weight_decay))
    subSpec decayedParams (mulSpec adaptiveLR mHat)
  update_eq := by
    intro _s _state _params _grads
    rfl

/-- AdamW's registered step spec agrees with executable AdamW over streams. -/
theorem runSteps_eq_stepSpec (lr weightDecay beta1 beta2 epsilon : α) {s : Shape}
    (current :
      TensorOptimizer.Step
        (TensorOptimizer.adamw (α := α) lr weightDecay beta1 beta2 epsilon) s)
    (grads : List (Tensor α s)) :
    (stepSpec (α := α) lr weightDecay beta1 beta2 epsilon).runSteps current grads =
      (TensorOptimizer.adamw (α := α) lr weightDecay beta1 beta2 epsilon).runSteps
        current grads :=
  StepSpec.runSteps_eq_optimizer_runSteps
    (stepSpec (α := α) lr weightDecay beta1 beta2 epsilon) current grads

end AdamW

namespace Adadelta

/-- The proof layer Adadelta update equation. -/
def updateSpec {s : Shape} (state : State α s) (params grads : Tensor α s) :
    State α s × Tensor α s :=
  let squaredGrads := squareSpec grads
  let newV := addSpec (scaleSpec state.v state.rho) (scaleSpec squaredGrads (1 - state.rho))
  let epsT : Tensor α s := fill state.epsilon s
  let rmsV := sqrtSpec (addSpec newV epsT)
  let rmsU := sqrtSpec (addSpec state.u epsT)
  let ratio := divSpec rmsU rmsV
  let delta := scaleSpec (mulSpec ratio grads) (-state.lr)
  let newParams := addSpec params delta
  let newU := addSpec (scaleSpec state.u state.rho) (scaleSpec (squareSpec delta) (1 - state.rho))
  ({ state with v := newV, u := newU }, newParams)

/-- The executable Adadelta optimizer follows `updateSpec`. -/
theorem update_eq_spec {s : Shape} (state : State α s) (params grads : Tensor α s) :
    update state params grads = updateSpec state params grads := by
  rfl

/-- Register Adadelta under the generic optimizer-step interface. -/
def stepSpec (lr rho epsilon : α) :
    StepSpec (TensorOptimizer.adadelta (α := α) lr rho epsilon) where
  nextState := fun state _params grads =>
    let squaredGrads := squareSpec grads
    let newV := addSpec (scaleSpec state.v state.rho) (scaleSpec squaredGrads (1 - state.rho))
    let epsT : Tensor α _ := fill state.epsilon _
    let rmsV := sqrtSpec (addSpec newV epsT)
    let rmsU := sqrtSpec (addSpec state.u epsT)
    let ratio := divSpec rmsU rmsV
    let delta := scaleSpec (mulSpec ratio grads) (-state.lr)
    let newU := addSpec (scaleSpec state.u state.rho) (scaleSpec (squareSpec delta) (1 - state.rho))
    { state with v := newV, u := newU }
  nextParams := fun state params grads =>
    let squaredGrads := squareSpec grads
    let newV := addSpec (scaleSpec state.v state.rho) (scaleSpec squaredGrads (1 - state.rho))
    let epsT : Tensor α _ := fill state.epsilon _
    let rmsV := sqrtSpec (addSpec newV epsT)
    let rmsU := sqrtSpec (addSpec state.u epsT)
    let ratio := divSpec rmsU rmsV
    let delta := scaleSpec (mulSpec ratio grads) (-state.lr)
    addSpec params delta
  update_eq := by
    intro _s _state _params _grads
    rfl

/-- Adadelta's registered step spec agrees with executable Adadelta over streams. -/
theorem runSteps_eq_stepSpec (lr rho epsilon : α) {s : Shape}
    (current : TensorOptimizer.Step (TensorOptimizer.adadelta (α := α) lr rho epsilon) s)
    (grads : List (Tensor α s)) :
    (stepSpec (α := α) lr rho epsilon).runSteps current grads =
      (TensorOptimizer.adadelta (α := α) lr rho epsilon).runSteps current grads :=
  StepSpec.runSteps_eq_optimizer_runSteps (stepSpec (α := α) lr rho epsilon) current grads

end Adadelta

namespace Muon

/-- The proof layer Muon initialization equation. -/
def initSpec {s : Shape} (lr momentum : α) (orthogonalizer : Orthogonalizer α s)
    (_params : Tensor α s) : State α s :=
  { lr := lr, momentum := momentum, buf := fill 0 s, orthogonalizer := orthogonalizer }

/-- The executable Muon initializer follows `initSpec`. -/
theorem init_eq_spec {s : Shape} (lr momentum : α) (orthogonalizer : Orthogonalizer α s)
    (params : Tensor α s) :
    init lr momentum orthogonalizer params = initSpec lr momentum orthogonalizer params := by
  rfl

/-- Muon initialization stores the requested learning rate. -/
theorem init_lr_eq {s : Shape} (lr momentum : α) (orthogonalizer : Orthogonalizer α s)
    (params : Tensor α s) :
    (init lr momentum orthogonalizer params).lr = lr := by
  rfl

/-- Muon initialization stores the requested momentum coefficient. -/
theorem init_momentum_eq {s : Shape} (lr momentum : α) (orthogonalizer : Orthogonalizer α s)
    (params : Tensor α s) :
    (init lr momentum orthogonalizer params).momentum = momentum := by
  rfl

/-- Muon initialization starts from the all-zero momentum buffer. -/
theorem init_buffer_eq_zero {s : Shape} (lr momentum : α) (orthogonalizer : Orthogonalizer α s)
    (params : Tensor α s) :
    (init lr momentum orthogonalizer params).buf = fill 0 s := by
  rfl

/-- Muon initialization stores exactly the orthogonalizer backend supplied by the caller. -/
theorem init_orthogonalizer_eq {s : Shape} (lr momentum : α)
    (orthogonalizer : Orthogonalizer α s) (params : Tensor α s) :
    (init lr momentum orthogonalizer params).orthogonalizer = orthogonalizer := by
  rfl

/-- After initialization, Muon's first step stores the momentum update from the zero buffer. -/
theorem init_update_buffer_eq_spec {s : Shape}
    (lr momentum : α) (orthogonalizer : Orthogonalizer α s)
    (params grads : Tensor α s) :
    (update (init lr momentum orthogonalizer params) params grads).1.buf =
      OptimizerUtils.updateMomentumBuf (fill 0 s) momentum grads := by
  rfl

/--
After initialization, Muon's first parameter update applies the supplied backend to the fresh
momentum buffer computed from the zero buffer.
-/
theorem init_update_params_eq_spec {s : Shape}
    (lr momentum : α) (orthogonalizer : Orthogonalizer α s)
    (params grads : Tensor α s) :
    (update (init lr momentum orthogonalizer params) params grads).2 =
      let newBuf := OptimizerUtils.updateMomentumBuf (fill 0 s) momentum grads
      subSpec params (scaleSpec (orthogonalizer.apply newBuf) lr) := by
  rfl

/--
The proof layer Muon update equation.

Muon-style training first updates the momentum buffer, then applies the configured
orthogonalization backend to that buffer, and finally subtracts the scaled direction from the
parameters.
-/
def updateSpec {s : Shape} (state : State α s) (params grads : Tensor α s) :
    State α s × Tensor α s :=
  let newBuf := OptimizerUtils.updateMomentumBuf state.buf state.momentum grads
  let direction := state.orthogonalizer.apply newBuf
  ({ state with buf := newBuf }, subSpec params (scaleSpec direction state.lr))

/-- The executable Muon optimizer follows `updateSpec`. -/
theorem update_eq_spec {s : Shape} (state : State α s) (params grads : Tensor α s) :
    update state params grads = updateSpec state params grads := by
  rfl

/-- Register Muon under the generic optimizer-step interface. -/
def stepSpec (lr momentum : α)
    (orthogonalizer : {s : Shape} → Orthogonalizer α s :=
      fun {s} => identityOrthogonalizer (α := α) (s := s)) :
    StepSpec (TensorOptimizer.muon (α := α) lr momentum orthogonalizer) where
  nextState := fun state _params grads =>
    { state with buf := OptimizerUtils.updateMomentumBuf state.buf state.momentum grads }
  nextParams := fun state params grads =>
    let newBuf := OptimizerUtils.updateMomentumBuf state.buf state.momentum grads
    subSpec params (scaleSpec (state.orthogonalizer.apply newBuf) state.lr)
  update_eq := by
    intro _s _state _params _grads
    rfl

/-- Muon's registered step spec agrees with executable Muon over streams. -/
theorem runSteps_eq_stepSpec (lr momentum : α)
    (orthogonalizer : {s : Shape} → Orthogonalizer α s :=
      fun {s} => identityOrthogonalizer (α := α) (s := s))
    {s : Shape}
    (current : TensorOptimizer.Step (TensorOptimizer.muon (α := α) lr momentum orthogonalizer) s)
    (grads : List (Tensor α s)) :
    (stepSpec (α := α) lr momentum orthogonalizer).runSteps current grads =
      (TensorOptimizer.muon (α := α) lr momentum orthogonalizer).runSteps current grads :=
  StepSpec.runSteps_eq_optimizer_runSteps
    (stepSpec (α := α) lr momentum orthogonalizer) current grads

/-- Muon's next state is the old state with only the momentum buffer replaced. -/
theorem update_state_eq_spec {s : Shape} (state : State α s) (params grads : Tensor α s) :
    (update state params grads).1 =
      { state with buf := OptimizerUtils.updateMomentumBuf state.buf state.momentum grads } := by
  rfl

/-- After initialization, Muon's first next state has the requested scalars/backend and fresh buffer. -/
theorem init_update_state_eq_spec {s : Shape}
    (lr momentum : α) (orthogonalizer : Orthogonalizer α s)
    (params grads : Tensor α s) :
    (update (init lr momentum orthogonalizer params) params grads).1 =
      ({ lr := lr, momentum := momentum,
         buf := OptimizerUtils.updateMomentumBuf (fill 0 s) momentum grads,
         orthogonalizer := orthogonalizer } : State α s) := by
  rfl

/-- Muon's next buffer is the momentum update `momentum * buf + grad`. -/
theorem update_buffer_eq_spec {s : Shape} (state : State α s) (params grads : Tensor α s) :
    (update state params grads).1.buf =
      OptimizerUtils.updateMomentumBuf state.buf state.momentum grads := by
  rfl

/-- A Muon update preserves the learning rate stored in the optimizer state. -/
theorem update_lr_eq {s : Shape} (state : State α s) (params grads : Tensor α s) :
    (update state params grads).1.lr = state.lr := by
  rfl

/-- A Muon update preserves the momentum coefficient stored in the optimizer state. -/
theorem update_momentum_eq {s : Shape} (state : State α s) (params grads : Tensor α s) :
    (update state params grads).1.momentum = state.momentum := by
  rfl

/-- A Muon update preserves the configured orthogonalizer backend. -/
theorem update_orthogonalizer_eq {s : Shape} (state : State α s) (params grads : Tensor α s) :
    (update state params grads).1.orthogonalizer = state.orthogonalizer := by
  rfl

/-- Muon's parameter direction is exactly the orthogonalizer applied to the fresh buffer. -/
theorem update_params_eq_spec {s : Shape} (state : State α s) (params grads : Tensor α s) :
    (update state params grads).2 =
      let newBuf := OptimizerUtils.updateMomentumBuf state.buf state.momentum grads
      subSpec params (scaleSpec (state.orthogonalizer.apply newBuf) state.lr) := by
  rfl

/--
For any orthogonalizer backend, Muon's stored momentum buffer evolves exactly like momentum SGD.
The backend changes the parameter direction, not the buffer recurrence.
-/
theorem update_buffer_eq_momentumSGD {s : Shape}
    (state : State α s) (params grads : Tensor α s) :
    (update state params grads).1.buf =
      (MomentumSGD.update
        ({ lr := state.lr, momentum := state.momentum, buf := state.buf } :
          MomentumSGD.State α s)
        params grads).1.buf := by
  rfl

/--
Starting from initialized states, Muon's first stored momentum buffer agrees with momentum SGD for
any orthogonalizer backend.
-/
theorem init_update_buffer_eq_momentumSGD {s : Shape}
    (lr momentum : α) (orthogonalizer : Orthogonalizer α s)
    (params grads : Tensor α s) :
    (update (init lr momentum orthogonalizer params) params grads).1.buf =
      (MomentumSGD.update (MomentumSGD.init lr momentum params) params grads).1.buf := by
  rfl

/--
If a Muon backend returns the fresh momentum buffer unchanged on this step, then the parameter
update agrees with momentum SGD for this step.
-/
theorem update_params_eq_momentumSGD_of_apply_eq {s : Shape}
    (state : State α s) (params grads : Tensor α s)
    (happly :
      state.orthogonalizer.apply
        (OptimizerUtils.updateMomentumBuf state.buf state.momentum grads) =
        OptimizerUtils.updateMomentumBuf state.buf state.momentum grads) :
    (update state params grads).2 =
      (MomentumSGD.update
        ({ lr := state.lr, momentum := state.momentum, buf := state.buf } :
          MomentumSGD.State α s)
        params grads).2 := by
  simp [update, MomentumSGD.update, happly]

/--
Initialized version of `update_params_eq_momentumSGD_of_apply_eq`.
-/
theorem init_update_params_eq_momentumSGD_of_apply_eq {s : Shape}
    (lr momentum : α) (orthogonalizer : Orthogonalizer α s)
    (params grads : Tensor α s)
    (happly :
      orthogonalizer.apply (OptimizerUtils.updateMomentumBuf (fill 0 s) momentum grads) =
        OptimizerUtils.updateMomentumBuf (fill 0 s) momentum grads) :
    (update (init lr momentum orthogonalizer params) params grads).2 =
      (MomentumSGD.update (MomentumSGD.init lr momentum params) params grads).2 := by
  exact update_params_eq_momentumSGD_of_apply_eq
    (state := init lr momentum orthogonalizer params)
    (params := params) (grads := grads) happly

/--
With the identity orthogonalizer, Muon has the same parameter update as momentum SGD.

This is the fallback law used by the public/runtime API: adding a real orthogonalization backend is
a separate obligation, but the identity backend cannot silently change the optimizer.
-/
theorem update_identity_params_eq_momentumSGD_spec {s : Shape}
    (lr momentum : α) (buf params grads : Tensor α s) :
    (update
        ({ lr := lr, momentum := momentum, buf := buf,
           orthogonalizer := identityOrthogonalizer (α := α) (s := s) } : State α s)
        params grads).2 =
      (MomentumSGD.update ({ lr := lr, momentum := momentum, buf := buf } :
        MomentumSGD.State α s) params grads).2 := by
  rfl

/--
Starting from initialized states, identity-backend Muon stores the same next buffer as momentum SGD
after one step.
-/
theorem init_identity_update_buffer_eq_momentumSGD {s : Shape}
    (lr momentum : α) (params grads : Tensor α s) :
    (update
        (init lr momentum (identityOrthogonalizer (α := α) (s := s)) params)
        params grads).1.buf =
      (MomentumSGD.update (MomentumSGD.init lr momentum params) params grads).1.buf := by
  exact init_update_buffer_eq_momentumSGD
    (lr := lr) (momentum := momentum)
    (orthogonalizer := identityOrthogonalizer (α := α) (s := s))
    (params := params) (grads := grads)

/--
Starting from initialized states, identity-backend Muon has the same parameter update as momentum
SGD after one step.
-/
theorem init_identity_update_params_eq_momentumSGD {s : Shape}
    (lr momentum : α) (params grads : Tensor α s) :
    (update
        (init lr momentum (identityOrthogonalizer (α := α) (s := s)) params)
        params grads).2 =
      (MomentumSGD.update (MomentumSGD.init lr momentum params) params grads).2 := by
  rfl

end Muon

end Optim
