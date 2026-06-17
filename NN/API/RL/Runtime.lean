/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.RL.Core
public import NN.Runtime.RL.Boundary
public import NN.Runtime.RL.Gymnasium
public import NN.Runtime.RL.Numerics
public import NN.Runtime.RL.PPO
public import NN.Runtime.RL.Session

/-!
# Public RL Runtime API

Runtime-facing RL tools under `NN.API.rl.*`: rollout boundary checks, Gymnasium sessions,
Float32/interval numerics, and PPO actor-critic wiring.
-/

@[expose] public section

namespace NN
namespace API

namespace rl

namespace boundary
export _root_.Runtime.RL.Boundary
  (isFiniteFloat tensorAll tensorFinite tensorInClosedInterval
   Contract Transition
   checkAction
   checkObservation checkReward checkDoneFlags
   checkTransitionFin checkTransition
   parseTransitionJson loadRollout)
export _root_.Runtime.RL.Boundary.Transition (done)

/-!
## Casting to Other Scalar Backends

The trust-boundary checker validates rollout JSON in host `Float`, because that is the interchange
format. The functions below cast accepted rollouts into the runtime scalar chosen for the proof or
training path.
-/

/-- Cast a `Float` observation tensor into a runtime scalar backend `α`. -/
def castObs {α : Type} [Runtime.Scalar α] {obsShape : _root_.Spec.Shape}
    (t : _root_.Spec.Tensor Float obsShape) : _root_.Spec.Tensor α obsShape :=
  _root_.Spec.mapTensor (Runtime.ofFloat (α := α)) t

/-- Cast a validated `Float` transition into a runtime scalar backend `α`. -/
def castTransition {α : Type} [Runtime.Scalar α]
    {obsShape : _root_.Spec.Shape} {nActions : Nat}
    (tr : _root_.Runtime.RL.Boundary.Transition obsShape nActions) :
    _root_.Spec.RL.ObservedTransition (_root_.Spec.Tensor α obsShape) (Fin nActions) α :=
  { observation := castObs (α := α) tr.observation
    action := tr.action
    reward := Runtime.ofFloat (α := α) tr.reward
    nextObservation := castObs (α := α) tr.nextObservation
    terminated := tr.terminated
    truncated := tr.truncated }

/-- Cast a whole rollout into a runtime scalar backend `α`. -/
def castRollout {α : Type} [Runtime.Scalar α]
    {obsShape : _root_.Spec.Shape} {nActions : Nat}
    (xs : Array (_root_.Runtime.RL.Boundary.Transition obsShape nActions)) :
    Array (_root_.Spec.RL.ObservedTransition (_root_.Spec.Tensor α obsShape) (Fin nActions) α) :=
  xs.map (castTransition (α := α) (obsShape := obsShape) (nActions := nActions))

/-- Load a rollout JSON file, validate it with the boundary contract, then cast to scalar `α`. -/
def loadRolloutCast {α : Type} [Runtime.Scalar α]
    {obsShape : _root_.Spec.Shape} {nActions : Nat}
    (path : String)
    (c : _root_.Runtime.RL.Boundary.Contract obsShape nActions) :
    IO (Array (_root_.Spec.RL.ObservedTransition (_root_.Spec.Tensor α obsShape) (Fin nActions) α)) := do
  let xs ← _root_.Runtime.RL.Boundary.loadRollout (obsShape := obsShape) (nActions := nActions) path c
  pure (castRollout (α := α) (obsShape := obsShape) (nActions := nActions) xs)

end boundary

namespace numerics
namespace float32
export _root_.Runtime.RL.Numerics.Float32
  (Float32Exec Interval32
   ofFloatIEEE32ExecChecked castTensorIEEE32ExecChecked castTransitionIEEE32ExecChecked
   discountedBackupIEEE32ExecChecked discountedReturnsVecFromIEEE32ExecChecked
   tdResidualIEEE32ExecChecked
   generalizedAdvantageEstimationVecIEEE32ExecChecked
   normalizeZScoreIEEE32ExecChecked
   importanceRatioIEEE32ExecChecked
   ppoClippedObjectiveFromRatioIEEE32ExecChecked
   discountedBackupInterval32 tdResidualInterval32
   ppoClippedObjectiveFromRatioInterval32
   discountedReturnsIntervals32 generalizedAdvantageEstimationIntervals32
   returnsWithinIntervals32)
end float32
end numerics

namespace session
export _root_.Runtime.RL.Session (CheckedSession)
export _root_.Runtime.RL.Session.CheckedSession (gymnasium ofEnv)
end session

namespace gym
export _root_.Runtime.RL.Gymnasium (Client Session)

namespace client
-- Only export the stable high-level entry points. The JSON request/response protocol and raw-step
-- protocol remain behind `NN.Runtime.RL.Gymnasium`.
export _root_.Runtime.RL.Gymnasium.Client (spawn reset close withClient)
end client

namespace session
export _root_.Runtime.RL.Gymnasium.Session (start reset stepChecked close withSession)
end session

end gym

namespace ppo
export _root_.Runtime.RL.PPO
  (StateBatchShape LogitsBatchShape ScalarBatchShape ValueBatchShape
   Step Rollout
   collectRolloutSessionWith collectRolloutCheckedSessionWith collectRolloutWith)
export _root_.Runtime.RL.PPO.Rollout (toActorCriticSample)

/-- Instantiate the standard PPO actor-critic runtime module. -/
def instantiateActorCritic
    {stateShape : _root_.Spec.Shape} {batch nActions : Nat} {α : Type}
    [Fact (0 < batch)] [Fact (0 < nActions)]
    [API.Semantics.Scalar α] [DecidableEq _root_.Spec.Shape] [API.Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (opts : _root_.Runtime.Autograd.Torch.Options)
    (actor : API.TorchLean.NN.Seq stateShape (.dim batch (.dim nActions .scalar)))
    (critic : API.TorchLean.NN.Seq stateShape (.dim batch (.dim 1 .scalar)))
    (cast : Float → α := API.Runtime.ofFloat) :
    IO (API.TorchLean.Module.ScalarModule α
      (API.TorchLean.NN.Seq.paramShapes actor ++ API.TorchLean.NN.Seq.paramShapes critic)
      [stateShape, (.dim batch (.dim nActions .scalar)), (.dim batch .scalar), (.dim batch .scalar),
        (.dim batch (.dim 1 .scalar))]) :=
  API.TorchLean.Module.instantiateWithOptions (α := α)
    (_root_.Runtime.RL.PolicyGradient.Autograd.ppoActorCriticScalarModuleDef
      (batch := batch) (nActions := nActions) actor critic)
    cast opts

/-- Create a PPO actor-critic update function from the public optimizer config. -/
def optimizerInputs {α : Type}
    [API.Semantics.Scalar α] [API.Runtime.Scalar α]
    {paramShapes inputShapes : List _root_.Spec.Shape}
    (m : API.TorchLean.Module.ScalarModule α paramShapes inputShapes)
    (cfg : API.TorchLean.Trainer.Optimizer) :
    IO (_root_.Runtime.Autograd.Torch.TList α inputShapes → IO Unit) := do
  match cfg with
  | .sgd lr momentum =>
      if momentum == 0.0 then
        let opt := API.TorchLean.Optim.sgd
          (α := α) (API.Runtime.ofFloat lr) (paramShapes := paramShapes)
        let optH ← API.TorchLean.Optim.handle (α := α) m opt
        pure optH.step
      else
        let opt := API.TorchLean.Optim.momentumSGD
          (α := α)
          (API.Runtime.ofFloat lr)
          (API.Runtime.ofFloat momentum)
          (paramShapes := paramShapes)
        let optH ← API.TorchLean.Optim.handle (α := α) m opt
        pure optH.step
  | .adam lr beta1 beta2 epsilon =>
      let opt := API.TorchLean.Optim.adam
        (α := α)
        (API.Runtime.ofFloat lr)
        (API.Runtime.ofFloat beta1)
        (API.Runtime.ofFloat beta2)
        (API.Runtime.ofFloat epsilon)
        (paramShapes := paramShapes)
      let optH ← API.TorchLean.Optim.handle (α := α) m opt
      pure optH.step
  | .adamw lr weightDecay beta1 beta2 epsilon =>
      let opt := API.TorchLean.Optim.adamw
        (α := α)
        (API.Runtime.ofFloat lr)
        (API.Runtime.ofFloat weightDecay)
        (API.Runtime.ofFloat beta1)
        (API.Runtime.ofFloat beta2)
        (API.Runtime.ofFloat epsilon)
        (paramShapes := paramShapes)
      let optH ← API.TorchLean.Optim.handle (α := α) m opt
      pure optH.step

/-- Read the concatenated actor-critic parameter pack from a PPO runtime module. -/
def params {α : Type} [API.Semantics.Scalar α] {paramShapes inputShapes : List _root_.Spec.Shape}
    (m : API.TorchLean.Module.ScalarModule α paramShapes inputShapes) :
    IO (_root_.Runtime.Autograd.Torch.TList α paramShapes) :=
  API.TorchLean.Module.params m

/-- Split a concatenated actor-critic parameter pack into `(actorParams, criticParams)`. -/
def splitActorCriticParams
    {σ₁ τ₁ σ₂ τ₂ : _root_.Spec.Shape}
    (actor : _root_.Runtime.Autograd.TorchLean.NN.Seq σ₁ τ₁)
    (critic : _root_.Runtime.Autograd.TorchLean.NN.Seq σ₂ τ₂)
    {α : Type}
    (ps :
      _root_.Runtime.Autograd.Torch.TList α
        (_root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes actor ++
          _root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes critic)) :
    _root_.Runtime.Autograd.Torch.TList α (_root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes actor) ×
      _root_.Runtime.Autograd.Torch.TList α (_root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes critic) :=
  _root_.Runtime.Autograd.Torch.Proofs.Autograd.Algebra.TList.splitAppend (α := α)
    (ss₁ := _root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes actor)
    (ss₂ := _root_.Runtime.Autograd.TorchLean.NN.Seq.paramShapes critic)
    ps

/--
Build a compiled actor-policy predictor from an actor-critic parameter pack.

PPO trains actor and critic together with one rollout-shaped module, but rollout collection and
evaluation usually need the actor at the single-observation shape. The equality argument keeps the
shared-parameter assumption explicit.
-/
def actorPolicyFromParams
    {obsShape logitsShape rolloutStateShape rolloutLogitsShape rolloutValueShape : _root_.Spec.Shape}
    {α : Type} [API.Semantics.Scalar α]
    (actorObs : API.TorchLean.NN.Seq obsShape logitsShape)
    (actorCompiled :
      _root_.Runtime.Autograd.Torch.CompiledOut α
        (API.TorchLean.NN.Seq.paramShapes actorObs ++ [obsShape]) logitsShape)
    (actorRollout : API.TorchLean.NN.Seq rolloutStateShape rolloutLogitsShape)
    (criticRollout : API.TorchLean.NN.Seq rolloutStateShape rolloutValueShape)
    (psAll : _root_.Runtime.Autograd.Torch.TList α
      (API.TorchLean.NN.Seq.paramShapes actorRollout ++
        API.TorchLean.NN.Seq.paramShapes criticRollout))
    (sameActorParams :
      API.TorchLean.NN.Seq.paramShapes actorRollout =
        API.TorchLean.NN.Seq.paramShapes actorObs := by rfl) :
    _root_.Spec.Tensor α obsShape → _root_.Spec.Tensor α logitsShape :=
  let (psActor, _psCritic) := splitActorCriticParams actorRollout criticRollout psAll
  let psActorObs : _root_.Runtime.Autograd.Torch.TList α
      (API.TorchLean.NN.Seq.paramShapes actorObs) :=
    Eq.mp (by rw [← sameActorParams]) psActor
  fun obs => API.TorchLean.NN.Seq.predict1 actorObs actorCompiled psActorObs obs

/--
Build a compiled critic-value predictor from an actor-critic parameter pack.

The returned function evaluates the single-observation critic and reads the scalar from its
length-one output vector.
-/
def criticValueFromParams
    {obsShape rolloutStateShape rolloutLogitsShape rolloutValueShape : _root_.Spec.Shape}
    {α : Type} [API.Semantics.Scalar α]
    (criticObs : API.TorchLean.NN.Seq obsShape (.dim 1 .scalar))
    (criticCompiled :
      _root_.Runtime.Autograd.Torch.CompiledOut α
        (API.TorchLean.NN.Seq.paramShapes criticObs ++ [obsShape]) (.dim 1 .scalar))
    (actorRollout : API.TorchLean.NN.Seq rolloutStateShape rolloutLogitsShape)
    (criticRollout : API.TorchLean.NN.Seq rolloutStateShape rolloutValueShape)
    (psAll : _root_.Runtime.Autograd.Torch.TList α
      (API.TorchLean.NN.Seq.paramShapes actorRollout ++
        API.TorchLean.NN.Seq.paramShapes criticRollout))
    (sameCriticParams :
      API.TorchLean.NN.Seq.paramShapes criticRollout =
        API.TorchLean.NN.Seq.paramShapes criticObs := by rfl) :
    _root_.Spec.Tensor α obsShape → α :=
  let (_psActor, psCritic) := splitActorCriticParams actorRollout criticRollout psAll
  let psCriticObs : _root_.Runtime.Autograd.Torch.TList α
      (API.TorchLean.NN.Seq.paramShapes criticObs) :=
    Eq.mp (by rw [← sameCriticParams]) psCritic
  fun obs =>
    _root_.Spec.Tensor.vecGet
      (API.TorchLean.NN.Seq.predict1 criticObs criticCompiled psCriticObs obs)
      ⟨0, by decide⟩

end ppo

end rl

end API
end NN
